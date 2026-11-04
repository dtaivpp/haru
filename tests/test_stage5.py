"""Stage 5 tests: backend TTS streaming/cancel/echo, and the Pi speaker's
playback of audio.output streams through a fake `aplay`."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from haru_gateway import tts
from haru_protocol import (
    AUDIO_CANCEL, AUDIO_START, AUDIO_STOP, PCM_S16LE, FrameKind, Message, MessageType,
    decode_frame, encode_frame,
)
from haru_terminal.audio_output import Speaker
from haru_terminal.config import AudioOutputConfig

RATE = 24000


# --- backend TTS ---------------------------------------------------------------------

class FakeSpeech(BaseHTTPRequestHandler):
    requests: list[dict] = []
    delay_s = 0.0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeSpeech.requests.append(body)
        time.sleep(FakeSpeech.delay_s)
        # 50 ms of audio per character, tagged with the character so tests can tell sentences apart.
        pcm = b"".join(bytes([ord(ch) % 256, 0]) * (RATE // 20) for ch in body["input"])
        self.send_response(200)
        self.send_header("Content-Length", str(len(pcm)))
        self.end_headers()
        self.wfile.write(pcm)

    def log_message(self, *a):
        pass


class FakeTerminals(tts.Terminals):
    def __init__(self):
        self.messages, self.frames = [], []

    def output_format(self, device_id):
        return {"encoding": PCM_S16LE, "sample_rate": RATE, "channels": 1} if device_id == "pi" else None

    async def send(self, device_id, type_, data):
        self.messages.append((device_id, str(type_), data))
        return True

    async def send_frame(self, device_id, frame):
        self.frames.append(decode_frame(frame))
        return True


@pytest.fixture
def speech_server():
    FakeSpeech.requests, FakeSpeech.delay_s = [], 0.0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeSpeech)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


async def wait_for(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


def engine(url, terms):
    return tts.TextToSpeech(tts.TTSConfig(base_url=url, model="kokoro", voice="af_heart"), terms)


def test_reply_becomes_one_stream(speech_server):
    async def body():
        terms = FakeTerminals()
        t = engine(speech_server, terms)
        t.say("pi", "**Hi** there.", time.monotonic())   # markdown stripped before synthesis
        t.say("pi", "Bye.", time.monotonic())
        t.end("pi")
        await wait_for(lambda: any(d.get("event") == AUDIO_STOP for _, _, d in terms.messages))
        events = [d["event"] for _, _, d in terms.messages]
        assert events == [AUDIO_START, AUDIO_STOP]
        start = terms.messages[0][2]
        assert (start["sample_rate"], start["encoding"]) == (RATE, PCM_S16LE)
        assert [r["input"] for r in FakeSpeech.requests] == ["Hi there.", "Bye."]
        assert FakeSpeech.requests[0]["sample_rate"] == RATE and FakeSpeech.requests[0]["response_format"] == "pcm"
        assert [f.seq for f in terms.frames] == list(range(len(terms.frames)))
        assert all(f.kind == FrameKind.AUDIO_OUTPUT and f.stream_id == start["stream_id"] for f in terms.frames)
        audio = b"".join(f.payload for f in terms.frames)
        assert len(audio) == (len("Hi there.") + len("Bye.")) * RATE // 10

        # No speaker -> nothing sent.
        t.say("cli", "Hello.", time.monotonic())
        await asyncio.sleep(0.1)
        assert all(dev == "pi" for dev, _, _ in terms.messages)

        # A second reply gets a new stream.
        t.say("pi", "Again.", time.monotonic())
        t.end("pi")
        await wait_for(lambda: len(terms.messages) == 4)
        assert terms.messages[2][2]["stream_id"] != start["stream_id"]
        await t.close()

    asyncio.run(body())


def test_cancel_stops_queue_and_speaker(speech_server):
    FakeSpeech.delay_s = 0.2

    async def body():
        terms = FakeTerminals()
        t = engine(speech_server, terms)
        for s in ("One.", "Two.", "Three.", "Four."):
            t.say("pi", s, time.monotonic())
        t.end("pi")
        await wait_for(lambda: terms.frames)  # first sentence out
        await t.cancel("pi")
        assert terms.messages[-1][2]["event"] == AUDIO_CANCEL
        await asyncio.sleep(0.6)
        assert len(FakeSpeech.requests) <= 2  # the rest of the queue was dropped
        assert not any(d["event"] == AUDIO_STOP for _, _, d in terms.messages)
        # And speaking still works afterwards.
        t.say("pi", "Five.", time.monotonic())
        t.end("pi")
        await wait_for(lambda: terms.messages[-1][2]["event"] == AUDIO_STOP)
        await t.close()

    asyncio.run(body())


def test_echo_detection(speech_server):
    async def body():
        t = engine(speech_server, FakeTerminals())
        t.say("pi", "The capital of France is Paris.", time.monotonic())
        t.end("pi")
        await wait_for(lambda: t._voices["pi"].spoken)
        assert t.is_echo("pi", "the capital of France is Paris")
        assert t.is_echo("pi", "capital of France")
        assert not t.is_echo("pi", "what about Germany then")
        assert not t.is_echo("pi", "Paris")        # one word isn't enough to call it echo
        assert not t.is_echo("other", "the capital of France is Paris")
        t._voices["pi"].spoken[0] = (time.monotonic() - 60, t._voices["pi"].spoken[0][1])
        assert not t.is_echo("pi", "the capital of France is Paris")  # too long ago
        await t.close()

    asyncio.run(body())


# --- Pi speaker --------------------------------------------------------------------

FAKE_APLAY = textwrap.dedent(f"""\
    #!{sys.executable}
    # Fake aplay: consume stdin in real time, record everything written.
    import os, sys
    out = open(os.environ["FAKE_APLAY_OUT"], "wb")
    while True:
        chunk = sys.stdin.buffer.read(960)
        if not chunk:
            break
        out.write(chunk); out.flush()
""")


class FakeLink:
    def __init__(self):
        self.handlers, self.frame_handlers, self.sent = {}, [], []

    def add_heartbeat_fields(self, fn): pass
    def add_hello_fields(self, fn): self.hello = fn()
    def on(self, type_, fn): self.handlers[str(type_)] = fn
    def on_frame(self, fn): self.frame_handlers.append(fn)

    async def send(self, type_, data):
        self.sent.append(data)
        return True

    async def control(self, **data):
        await self.handlers[str(MessageType.AUDIO_OUTPUT)](Message(str(MessageType.AUDIO_OUTPUT), data))

    def frame(self, stream_id, seq, payload):
        for fh in self.frame_handlers:
            fh(encode_frame(FrameKind.AUDIO_OUTPUT, stream_id, seq, 0, payload))


@pytest.fixture
def fake_aplay(tmp_path, monkeypatch):
    path = tmp_path / "aplay"
    path.write_text(FAKE_APLAY)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    out = tmp_path / "played.raw"
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_APLAY_OUT", str(out))
    return out


def test_speaker_plays_streams_and_reports(fake_aplay):
    async def body():
        link = FakeLink()
        spk = Speaker(link, AudioOutputConfig(enabled=True, device="fake", sample_rate=RATE))
        assert link.hello == {"audio_output": {"encoding": PCM_S16LE, "sample_rate": RATE, "channels": 1}}
        runner = asyncio.create_task(spk.run_forever())
        await asyncio.sleep(0.2)  # silence while idle

        fmt = {"encoding": PCM_S16LE, "sample_rate": RATE, "channels": 1}
        await link.control(event=AUDIO_START, stream_id=7, **fmt)
        speech = b"\x11\x22" * (RATE // 2)  # 0.5 s
        for i in range(5):
            link.frame(7, i, speech[i * 4800:(i + 1) * 4800])
        link.frame(99, 0, b"\xff\xff" * 100)  # other stream: ignored
        await link.control(event=AUDIO_STOP, stream_id=7)
        await wait_for(lambda: any(d["playback"]["event"] == "finished" for d in link.sent))
        events = [d["playback"]["event"] for d in link.sent]
        assert events == ["started", "finished"]
        assert 480 <= link.sent[-1]["playback"]["played_ms"] <= 520
        await asyncio.sleep(0.1)
        played = fake_aplay.read_bytes()
        assert speech in played and b"\xff\xff" * 100 not in played

        # Cancel mid-stream: the rest never plays.
        await link.control(event=AUDIO_START, stream_id=8, **fmt)
        long = b"\x33\x44" * (RATE * 3)  # 3 s
        link.frame(8, 0, long)
        await wait_for(lambda: link.sent[-1]["playback"]["event"] == "started")
        await asyncio.sleep(0.2)
        await link.control(event=AUDIO_CANCEL, stream_id=8)
        assert link.sent[-1]["playback"]["event"] == "cancelled"
        assert link.sent[-1]["playback"]["played_ms"] < 1000
        assert spk.status()["audio_out"]["buffered_ms"] == 0

        # Wrong format is refused.
        await link.control(event=AUDIO_START, stream_id=9, encoding=PCM_S16LE, sample_rate=16000, channels=1)
        assert spk._stream_id is None
        runner.cancel()

    asyncio.run(body())
