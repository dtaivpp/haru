"""Stage 3 tests: Silero VAD segmentation on real synthesized speech, and the
STT client against a fake OpenAI-compatible transcription server.

Speech clips come from macOS `say` (skipped elsewhere), so no personal
recordings live in the repo. The real Whisper path is verified live against
the `stt` container instead; see README.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from haru_gateway import stt
from haru_gateway.audio import AudioStream
from haru_protocol import Frame, FrameKind

ROOT = Path(__file__).resolve().parent.parent
VAD_MODEL = ROOT / ".cache" / "silero_vad.onnx"
VAD_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx"
FRAME = 320  # 20 ms at 16 kHz


@pytest.fixture(scope="session")
def vad_model() -> Path:
    if not VAD_MODEL.exists():
        VAD_MODEL.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(VAD_URL, VAD_MODEL)
    return VAD_MODEL


def say(text: str, tmp: Path) -> np.ndarray:
    if not (shutil.which("say") and shutil.which("afconvert")):
        pytest.skip("needs macOS say/afconvert")
    aiff, wav = tmp / "s.aiff", tmp / "s.wav"
    try:
        # macOS's speech service occasionally wedges; skip rather than hang the suite.
        subprocess.run(["say", "-o", str(aiff), text], check=True, timeout=30)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
                       check=True, timeout=30)
    except subprocess.TimeoutExpired:
        pytest.skip("macOS say/afconvert timed out")
    import wave
    with wave.open(str(wav)) as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")


def silence(seconds: float) -> np.ndarray:
    # Faint noise rather than exact zeros, like the Jabra's gated output.
    rng = np.random.default_rng(0)
    return rng.integers(-2, 3, int(16000 * seconds)).astype("<i2")


def frames(samples: np.ndarray):
    t_us = time.time_ns() // 1000
    for i, start in enumerate(range(0, len(samples) - FRAME + 1, FRAME)):
        yield Frame(FrameKind.AUDIO_INPUT, 1, i, t_us + i * 20_000, samples[start:start + FRAME].tobytes())


def segment(vad_model, samples, **vad_kw):
    cfg = stt.VADConfig(model_path=str(vad_model), **vad_kw)
    seg = stt.Segmenter(stt.SileroVAD(cfg.model_path), cfg, "pi/1")
    out = []
    for f in frames(samples):
        out += seg.feed(f)
    return out


def test_vad_splits_two_phrases_and_ignores_silence(vad_model, tmp_path):
    a = say("Hello there, can you hear me?", tmp_path)
    b = say("The weather is lovely today.", tmp_path)
    audio = np.concatenate([silence(1.0), a, silence(1.5), b, silence(1.5)])
    utts = segment(vad_model, audio)
    assert len(utts) == 2, [u.speech_ms for u in utts]
    for u, src in zip(utts, (a, b)):
        src_ms = len(src) * 1000 // 16000
        got_ms = len(u.pcm) * 1000 // 32000
        # Covers the phrase (plus pre-roll), but without the long gaps around it.
        assert src_ms * 0.8 <= got_ms <= src_ms + 1000, (src_ms, got_ms)
        assert not u.forced

    assert segment(vad_model, silence(5.0)) == []


def test_vad_forces_cut_on_long_speech(vad_model, tmp_path):
    long = say("This is a long sentence that just keeps going and going " * 3, tmp_path)
    utts = segment(vad_model, np.concatenate([long, silence(1.0)]), max_utterance_s=2.0)
    assert len(utts) >= 2 and utts[0].forced
    assert len(utts[0].pcm) // 32 <= 2100  # ~2 s cap


# --- STT client vs a fake Speaches -------------------------------------------------

class FakeSpeaches(BaseHTTPRequestHandler):
    models: set[str] = set()
    transcriptions: list[bytes] = []

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json({"data": [{"id": m} for m in sorted(self.models)]})
        else:
            self._json({}, 404)

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path.startswith("/v1/models/"):
            self.models.add(self.path.removeprefix("/v1/models/"))
            self._json({})
        elif self.path == "/v1/audio/transcriptions":
            assert b"RIFF" in body and b"verbose_json" in body and b"vad_filter" in body
            FakeSpeaches.transcriptions.append(body)
            if len(FakeSpeaches.transcriptions) == 1:  # the warm-up silence
                self._json({"text": "", "segments": []})
            else:
                self._json({"text": "Hello there. Thank you.", "segments": [
                    {"text": " Hello there.", "no_speech_prob": 0.05, "avg_logprob": -0.2},
                    {"text": " Thank you.", "no_speech_prob": 0.9, "avg_logprob": -1.4},  # hallucination
                ]})

    def log_message(self, *a):
        pass


def test_stt_downloads_warms_transcribes_and_filters(vad_model, tmp_path):
    FakeSpeaches.models, FakeSpeaches.transcriptions = set(), []
    srv = HTTPServer(("127.0.0.1", 0), FakeSpeaches)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    speech = say("Hello there.", tmp_path)

    async def body():
        statuses = []

        async def on_status(s):
            statuses.append(s)

        cfg = stt.STTConfig(base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="whisper-x",
                            language="en", vad=stt.VADConfig(model_path=str(vad_model)))
        engine = stt.SpeechToText(cfg, on_status=on_status)
        heard = []
        engine.on_transcript(heard.append)
        runner = asyncio.create_task(engine.run())
        deadline = time.monotonic() + 5
        while engine.status != "ok":
            assert time.monotonic() < deadline, statuses
            await asyncio.sleep(0.02)
        assert statuses == ["downloading", "loading", "ok"]
        assert "whisper-x" in FakeSpeaches.models

        stream = AudioStream("pi", 1, "pcm_s16le", 16000, 1, 20)
        engine.on_start(stream)
        for f in frames(np.concatenate([silence(0.5), speech, silence(1.0)])):
            engine.on_audio(stream, f, 0)
        while not heard:
            assert time.monotonic() < deadline + 5
            await asyncio.sleep(0.02)
        tr = heard[0]
        assert tr.text == "Hello there."  # the hallucinated segment is filtered out
        assert tr.dropped_segments == 1 and tr.stream_label == "pi/1"
        assert tr.end_to_text_ms < 1000
        runner.cancel()
        await engine.close()

    asyncio.run(body())
    srv.shutdown()
