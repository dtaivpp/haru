"""Stage 2 tests: mic streaming from terminal to gateway, stream bookkeeping
and the stats/recorder sinks. Capture runs through the real MicStreamer and
a fake `arecord` on PATH that emits a 440 Hz tone in real time."""

from __future__ import annotations

import asyncio
import math
import os
import stat
import sys
import textwrap
import time
import wave

import pytest

from haru_gateway import audio
from haru_protocol import FrameKind, Frame, ProtocolError
from haru_terminal.config import AudioInputConfig
from haru_terminal.audio_input import MicStreamer
from haru_terminal.transport import BackendLink
from test_stage1 import RunningGateway, free_port, terminal_config, until

FAKE_ARECORD = textwrap.dedent(f"""\
    #!{sys.executable}
    # Fake arecord: 440 Hz at -6 dBFS, 16 kHz mono s16le, paced in real time.
    import math, struct, sys, time
    rate, chunk = 16000, 320
    n, t0 = 0, time.monotonic()
    while True:
        buf = b"".join(struct.pack("<h", int(16384 * math.sin(2 * math.pi * 440 * (n + i) / rate)))
                       for i in range(chunk))
        n += chunk
        try:
            sys.stdout.buffer.write(buf); sys.stdout.buffer.flush()
        except BrokenPipeError:
            break
        time.sleep(max(0, t0 + n / rate - time.monotonic()))
""")


@pytest.fixture
def fake_arecord(tmp_path, monkeypatch):
    path = tmp_path / "arecord"
    path.write_text(FAKE_ARECORD)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    return path


def frame(stream_id, seq, payload=b"\0\0" * 320):
    return Frame(FrameKind.AUDIO_INPUT, stream_id, seq, time.time_ns() // 1000, payload)


class Recorder:
    def __init__(self):
        self.events = []

    def on_start(self, stream): self.events.append(("start", stream.stream_id))
    def on_audio(self, stream, f, received_at_us): self.events.append(("audio", f.seq))
    def on_stop(self, stream, reason): self.events.append(("stop", reason))


def test_router_tracks_gaps_and_isolates_broken_sinks():
    class Broken:
        def on_start(self, s): raise RuntimeError("boom")
        on_audio = on_stop = on_start

    rec = Recorder()
    router = audio.AudioRouter("pi", [Broken(), rec])
    router.handle_control({"event": "start", "stream_id": 1, "encoding": "pcm_s16le"})
    for seq in (0, 1, 4):
        router.handle_frame(frame(1, seq))
    router.handle_frame(frame(99, 0))  # unknown stream: ignored
    assert router.streams[1].lost_frames == 2
    router.close("disconnected")
    assert rec.events == [("start", 1), ("audio", 0), ("audio", 1), ("audio", 4), ("stop", "disconnected")]

    with pytest.raises(ProtocolError):
        router.handle_control({"event": "start", "stream_id": 2, "encoding": "opus"})
    with pytest.raises(ProtocolError):
        router.handle_control({"event": "start"})


def test_stats_sink_levels():
    sink = audio.StatsSink(interval_s=3600)
    stream = audio.AudioStream("pi", 1, "pcm_s16le", 16000, 1, 20)
    sink.on_start(stream)
    full_scale = (b"\xff\x7f\x00\x80") * 160  # +32767 / -32768 square wave
    sink.on_audio(stream, frame(1, 0, full_scale), time.time_ns() // 1000)
    sink.on_stop(stream, "done")
    r = sink.last_report["pi/1"]
    assert r["peak_dbfs"] == 0.0 and r["rms_dbfs"] > -0.1 and r["clipped"] == 320


def test_wav_recorder_rotates_and_prunes(tmp_path):
    sink = audio.WavRecorderSink(tmp_path, segment_s=0, keep=2)  # new segment every frame
    stream = audio.AudioStream("pi", 1, "pcm_s16le", 16000, 1, 20)
    sink.on_start(stream)
    for seq in range(4):
        sink.on_audio(stream, frame(1, seq), 0)
        time.sleep(1.01)  # filenames have 1 s resolution
    sink.on_stop(stream, "done")
    files = sorted(tmp_path.glob("*.wav"))
    assert len(files) == 2
    with wave.open(str(files[-1])) as wf:
        assert (wf.getframerate(), wf.getnchannels(), wf.getnframes()) == (16000, 1, 320)


def test_mic_streams_end_to_end(fake_arecord, tmp_path):
    async def body():
        port = free_port()
        gw = RunningGateway(port)
        stats = audio.StatsSink(interval_s=0.5)
        gw.gateway.audio_sinks = [stats, audio.WavRecorderSink(tmp_path / "rec")]
        await gw.start()

        cfg = terminal_config(port)
        cfg.audio_input = AudioInputConfig(enabled=True, device="fake")
        link = BackendLink(cfg, capabilities=["audio.input"])
        mic = MicStreamer(link, cfg.audio_input)
        tasks = [asyncio.create_task(link.run_forever()), asyncio.create_task(mic.run_forever())]

        await until(lambda: "test-pi" in gw.gateway.sessions)
        router = gw.gateway.sessions["test-pi"].audio
        await until(lambda: router.streams and next(iter(router.streams.values())).frames >= 50)
        await until(lambda: stats.last_report)
        report = next(iter(stats.last_report.values()))
        assert report["lost"] == 0
        assert -10 < report["rms_dbfs"] < -8          # -6 dBFS peak sine => -9 dBFS rms
        assert report["latency_ms_p95"] < 100
        assert 45 <= report["fps"] <= 55
        first_stream = next(iter(router.streams))

        # Backend restart: a fresh stream starts on the new connection.
        await gw.shutdown()
        await asyncio.sleep(0.3)
        gw = RunningGateway(port)
        gw.gateway.audio_sinks = [audio.StatsSink()]
        await gw.start()
        await until(lambda: "test-pi" in gw.gateway.sessions and gw.gateway.sessions["test-pi"].audio.streams)
        router = gw.gateway.sessions["test-pi"].audio
        assert next(iter(router.streams)) != first_stream
        await until(lambda: next(iter(router.streams.values())).frames >= 10)
        assert next(iter(router.streams.values())).lost_frames == 0
        assert mic.status()["audio_in"]["streaming"]

        await mic.stop("test over")
        await until(lambda: not router.streams)
        await asyncio.sleep(0.2)
        assert not router.streams  # stays stopped; capture continuing must not reopen it
        mic.resume()
        await until(lambda: router.streams)
        await link.close()
        for t in tasks:
            t.cancel()
        await gw.shutdown()

    asyncio.run(body())
    wavs = list((tmp_path / "rec").glob("*.wav"))
    assert wavs
    with wave.open(str(wavs[0])) as wf:
        assert wf.getnframes() >= 16000 * 0.9  # about a second or more of audio, header finalized
