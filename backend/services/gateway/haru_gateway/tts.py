"""Stage 5: reply sentences -> Kokoro TTS -> audio.output stream to the terminal.

Each reply becomes one output stream per terminal: `say()` for every sentence
as the LLM produces it, `end()` when the reply is complete, and `cancel()` on
barge-in. Sentences are synthesized in order by a per-terminal worker. While
sentence N plays on the Pi, N+1 is already being synthesized, because Kokoro
runs about 25-35x faster than real time on the GPU.

TTS is the OpenAI-compatible /v1/audio/speech API (Speaches serving Kokoro in
the `stt` container today), so the engine is swappable by URL. Audio is
requested directly in the format the terminal advertised in its hello, so
nothing on either side resamples.

Echo guard: the Jabra has hardware echo cancellation, but if Haru's own voice
ever leaks into the mic, she'd answer herself forever. `is_echo()` compares a
transcript against what was just spoken, so the conversation can drop it
while real barge-in still works.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

from haru_protocol import (
    AUDIO_CANCEL,
    AUDIO_START,
    AUDIO_STOP,
    PCM_S16LE,
    FrameKind,
    MessageType,
    encode_frame,
)

log = logging.getLogger("haru.gateway.tts")

USER_AGENT = "haru-gateway/0.1"
FRAME_MS = 200            # audio.output frame size; small enough to cancel promptly
ECHO_WINDOW_S = 15.0      # how long spoken text is remembered for echo detection
MIN_REALTIME_FACTOR = 8.0  # below this at warm-up, Kokoro is probably on the CPU

# Markdown/emoji the persona tells the model not to use, but belt and braces:
# anything left here gets read aloud literally or as odd noises.
_STRIP = re.compile(r"[*_#`~>|]|[\U0001F000-\U0001FAFF☀-➿️]")
_WORD = re.compile(r"[a-z0-9']+")
# gpt-oss likes typographic characters (narrow no-break spaces, non-breaking
# hyphens, curly quotes). Give the synthesizer plain ones.
_PLAIN = str.maketrans({"\u202f": " ", "\u00a0": " ", "\u2009": " ", "\u2011": "-", "\u2010": "-",
                        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


@dataclass
class TTSConfig:
    base_url: str
    model: str
    voice: str
    speed: float = 1.0
    health_interval_s: float = 30.0

    @classmethod
    def from_env(cls) -> TTSConfig | None:
        base_url = os.environ.get("HARU_TTS_BASE_URL", "").rstrip("/")
        if not base_url:
            return None
        e = os.environ.get
        return cls(
            base_url=base_url,
            model=e("HARU_TTS_MODEL", "speaches-ai/Kokoro-82M-v1.0-ONNX"),
            voice=e("HARU_TTS_VOICE", "af_heart"),
            speed=float(e("HARU_TTS_SPEED", "1.0")),
            health_interval_s=float(e("HARU_TTS_HEALTH_INTERVAL_S", "30")),
        )


class Terminals:
    """What the TTS needs from the gateway: who has a speaker, and how to reach it."""

    def output_format(self, device_id: str) -> dict | None: ...
    async def send(self, device_id: str, type_: MessageType, data: dict) -> bool: ...
    async def send_frame(self, device_id: str, frame: bytes) -> bool: ...


@dataclass
class _Item:
    text: str | None          # None = end of reply
    reply_started: float      # monotonic, when the transcript that caused this reply arrived


@dataclass
class _Voice:
    """Per-terminal speaking state."""
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    worker: asyncio.Task | None = None
    stream_id: int = 0
    streaming: bool = False
    seq: int = 0
    spoken: deque = field(default_factory=lambda: deque(maxlen=20))  # (monotonic, words)
    reply_started: float = 0.0
    first_audio_sent: bool = False


# (device_id, stream_id, pcm, sample_rate, channels) for every synthesized sentence
AudioHandler = Callable[[str, int, bytes, int, int], Awaitable[None] | None]
# (device_id, "started" | "finished" | "cancelled", stream_id) as the terminal plays
PlaybackHandler = Callable[[str, str, int], Awaitable[None] | None]


class TextToSpeech:
    def __init__(self, config: TTSConfig, terminals: Terminals,
                 on_status: Callable[[str], Awaitable[None]] | None = None) -> None:
        self._audio_handlers: list[AudioHandler] = []
        self._playback_handlers: list[PlaybackHandler] = []
        self.config = config
        self.terminals = terminals
        self.status = "unknown"
        self._on_status = on_status
        self._voices: dict[str, _Voice] = {}
        self._client = httpx.AsyncClient(headers={"User-Agent": USER_AGENT},
                                         timeout=httpx.Timeout(30.0, connect=5.0))
        # stream_id -> (device_id, reply_started) for latency reporting when the Pi starts playing.
        self._pending_start: dict[tuple[str, int], float] = {}

    def on_audio(self, handler: AudioHandler) -> None:
        self._audio_handlers.append(handler)

    def on_playback_event(self, handler: PlaybackHandler) -> None:
        self._playback_handlers.append(handler)

    # --- conversation-facing API --------------------------------------------

    def say(self, device_id: str, sentence: str, reply_started: float) -> None:
        text = _STRIP.sub("", sentence).translate(_PLAIN).strip()
        if not text:
            return
        if self.terminals.output_format(device_id) is None:
            log.debug("%s has no speaker; not speaking %r", device_id, text)
            return
        self._voice(device_id).queue.put_nowait(_Item(text, reply_started))

    def end(self, device_id: str) -> None:
        if device_id in self._voices:
            self._voices[device_id].queue.put_nowait(_Item(None, 0.0))

    async def cancel(self, device_id: str) -> None:
        """Barge-in: stop synthesizing, drop the queue, silence the speaker."""
        v = self._voices.get(device_id)
        if v is None:
            return
        while not v.queue.empty():
            v.queue.get_nowait()
        if v.worker and not v.worker.done():
            v.worker.cancel()
            v.worker = None
        if v.streaming:
            v.streaming = False
            log.info("%s: cancelling speech (stream %d)", device_id, v.stream_id)
            await self.terminals.send(device_id, MessageType.AUDIO_OUTPUT,
                                      {"event": AUDIO_CANCEL, "stream_id": v.stream_id})

    def is_echo(self, device_id: str, text: str) -> bool:
        """True if `text` is mostly words Haru said in the last ECHO_WINDOW_S."""
        v = self._voices.get(device_id)
        words = _WORD.findall(text.lower())
        if v is None or not words:
            return False
        cutoff = time.monotonic() - ECHO_WINDOW_S
        recent = set()
        for t, spoken in v.spoken:
            if t >= cutoff:
                recent |= spoken
        if not recent:
            return False
        overlap = sum(1 for w in words if w in recent) / len(words)
        return overlap >= 0.7 and len(words) >= 2

    def on_playback(self, device_id: str, data: dict) -> None:
        """Playback report from a terminal (device.status "playback")."""
        key = (device_id, data.get("stream_id"))
        if data.get("event") in ("started", "finished", "cancelled") and isinstance(key[1], int):
            for h in self._playback_handlers:
                asyncio.create_task(_call(h, device_id, data["event"], key[1]))
        if data.get("event") == "started" and key in self._pending_start:
            started = self._pending_start.pop(key)
            log.info("%s: speaking (stream %s), %.0f ms after the transcript arrived",
                     device_id, key[1], (time.monotonic() - started) * 1000)
        elif data.get("event") in ("finished", "cancelled"):
            self._pending_start.pop(key, None)
            log.info("%s: playback %s (stream %s, %s ms)", device_id, data.get("event"),
                     key[1], data.get("played_ms"))

    # --- worker -------------------------------------------------------------

    def _voice(self, device_id: str) -> _Voice:
        v = self._voices.setdefault(device_id, _Voice())
        if v.worker is None or v.worker.done():
            v.worker = asyncio.create_task(self._speak_forever(device_id, v))
        return v

    async def _speak_forever(self, device_id: str, v: _Voice) -> None:
        while True:
            item = await v.queue.get()
            fmt = self.terminals.output_format(device_id)
            if fmt is None:
                v.streaming = False
                continue
            if item.text is None:
                if v.streaming:
                    v.streaming = False
                    await self.terminals.send(device_id, MessageType.AUDIO_OUTPUT,
                                              {"event": AUDIO_STOP, "stream_id": v.stream_id})
                continue
            if not v.streaming:
                v.stream_id = (v.stream_id + 1) & 0xFFFF
                v.seq = 0
                v.streaming = await self.terminals.send(device_id, MessageType.AUDIO_OUTPUT, {
                    "event": AUDIO_START, "stream_id": v.stream_id, "encoding": PCM_S16LE,
                    "sample_rate": fmt["sample_rate"], "channels": fmt["channels"]})
                if not v.streaming:
                    continue
                self._pending_start[(device_id, v.stream_id)] = item.reply_started
            try:
                t0 = time.monotonic()
                pcm = await self._synthesize(item.text, fmt["sample_rate"])
                synth_ms = (time.monotonic() - t0) * 1000
            except httpx.HTTPError as exc:
                log.warning("TTS failed for %r: %s", item.text, exc)
                await self._set_status("unavailable")
                continue
            await self._set_status("ok")
            v.spoken.append((time.monotonic(), set(_WORD.findall(item.text.lower()))))
            audio_ms = len(pcm) * 1000 // (2 * fmt["sample_rate"] * fmt["channels"])
            log.info("%s: synthesized %r (%d ms audio in %.0f ms)", device_id, item.text, audio_ms, synth_ms)
            for h in self._audio_handlers:
                await _call(h, device_id, v.stream_id, pcm, fmt["sample_rate"], fmt["channels"])
            step = fmt["sample_rate"] * fmt["channels"] * 2 * FRAME_MS // 1000
            for off in range(0, len(pcm), step):
                frame = encode_frame(FrameKind.AUDIO_OUTPUT, v.stream_id, v.seq,
                                     time.time_ns() // 1000, pcm[off:off + step])
                if not await self.terminals.send_frame(device_id, frame):
                    v.streaming = False
                    break
                v.seq += 1

    async def _synthesize(self, text: str, sample_rate: int) -> bytes:
        resp = await self._client.post(f"{self.config.base_url}/audio/speech", json={
            "model": self.config.model, "voice": self.config.voice, "input": text,
            "response_format": "pcm", "sample_rate": sample_rate, "speed": self.config.speed})
        return resp.raise_for_status().content

    # --- health -------------------------------------------------------------

    async def run(self) -> None:
        while True:
            try:
                await self._check_and_warm()
            except httpx.HTTPError as exc:
                log.debug("TTS health check failed: %s", exc)
                await self._set_status("unavailable")
            await asyncio.sleep(self.config.health_interval_s if self.status == "ok" else 5)

    async def _check_and_warm(self) -> None:
        base = self.config.base_url
        models = (await self._client.get(f"{base}/models")).raise_for_status().json()
        if self.config.model not in {m.get("id") for m in models.get("data", [])}:
            await self._set_status("downloading")
            await self._client.post(f"{base}/models/{self.config.model}", timeout=1800)
        if self.status != "ok":
            await self._set_status("loading")
            # The first synthesis after a fresh image can take ~85 s (CUDA JIT; see stt.env).
            warm = await self._client.post(f"{base}/audio/speech", timeout=600, json={
                "model": self.config.model, "voice": self.config.voice, "input": "Hello.",
                "response_format": "pcm", "sample_rate": 24000})
            warm.raise_for_status()
            # Second, timed run: is Kokoro actually on the GPU?
            text = "This is a quick check of how fast speech synthesis is running today."
            t0 = time.monotonic()
            pcm = await self._synthesize(text, 24000)
            factor = (len(pcm) / 48000) / max(1e-6, time.monotonic() - t0)
            (log.info if factor >= MIN_REALTIME_FACTOR else log.warning)(
                "TTS speed %.0fx realtime%s", factor,
                "" if factor >= MIN_REALTIME_FACTOR else
                " (slow: Kokoro may be on the CPU; check PYTHONHASHSEED in config/stt.env)")
        await self._set_status("ok")

    async def _set_status(self, status: str) -> None:
        if status == self.status:
            return
        log.log(logging.INFO if status in ("ok", "loading", "downloading") else logging.WARNING,
                "TTS %s (%s, voice %s)", status, self.config.model, self.config.voice)
        self.status = status
        if self._on_status:
            await self._on_status(status)

    async def close(self) -> None:
        for v in self._voices.values():
            if v.worker:
                v.worker.cancel()
        await self._client.aclose()


async def _call(fn, *args) -> None:
    try:
        r = fn(*args)
        if asyncio.iscoroutine(r):
            await r
    except Exception:
        log.exception("TTS event handler failed")
