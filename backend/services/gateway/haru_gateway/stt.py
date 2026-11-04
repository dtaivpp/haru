"""Stage 3: voice activity detection + speech-to-text.

The terminal streams the mic continuously. Here, a Silero VAD per stream cuts
that stream into utterances, and each utterance goes to an OpenAI-compatible
`/audio/transcriptions` server (Speaches / faster-whisper in the `stt`
container). Swapping STT engines means pointing HARU_STT_BASE_URL somewhere
else; the Pi never knows.

Only speech is sent to Whisper, never the gaps. Whisper fed silence
hallucinates ("Thank you.", "You"), and the model's own no_speech_prob /
avg_logprob per segment filters whatever noise still slips past the VAD.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import wave
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx
import numpy as np
import onnxruntime as ort

from haru_protocol import Frame

from haru_gateway.audio import AudioStream

log = logging.getLogger("haru.gateway.stt")

SAMPLE_RATE = 16000
VAD_CHUNK = 512    # samples per Silero v5 call at 16 kHz (32 ms)
VAD_CONTEXT = 64   # samples of the previous chunk Silero v5 expects prepended
CHUNK_MS = VAD_CHUNK * 1000 // SAMPLE_RATE

# Same User-Agent rule as the LLM client (Cloudflare 1010 on Python defaults),
# in case STT ever points at a proxied endpoint.
USER_AGENT = "haru-gateway/0.1"


@dataclass
class VADConfig:
    model_path: str = "/opt/silero/silero_vad.onnx"
    start_threshold: float = 0.5
    end_threshold: float = 0.35
    min_silence_ms: int = 500      # end-of-utterance wait: the main latency/cut-off tradeoff
    min_speech_ms: int = 250       # shorter blips (clicks, coughs) are discarded
    pre_roll_ms: int = 300         # audio kept from before speech onset, so first syllables survive
    max_utterance_s: float = 30.0


@dataclass
class STTConfig:
    base_url: str
    model: str
    language: str | None
    vad: VADConfig
    health_interval_s: float = 30.0
    no_speech_prob_max: float = 0.6
    avg_logprob_min: float = -1.0

    @classmethod
    def from_env(cls) -> STTConfig | None:
        base_url = os.environ.get("HARU_STT_BASE_URL", "").rstrip("/")
        if not base_url:
            return None
        e = os.environ.get
        return cls(
            base_url=base_url,
            model=e("HARU_STT_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2"),
            language=e("HARU_STT_LANGUAGE") or None,
            health_interval_s=float(e("HARU_STT_HEALTH_INTERVAL_S", "30")),
            vad=VADConfig(
                model_path=e("HARU_VAD_MODEL", VADConfig.model_path),
                start_threshold=float(e("HARU_VAD_START_THRESHOLD", "0.5")),
                end_threshold=float(e("HARU_VAD_END_THRESHOLD", "0.35")),
                min_silence_ms=int(e("HARU_VAD_MIN_SILENCE_MS", "500")),
                min_speech_ms=int(e("HARU_VAD_MIN_SPEECH_MS", "250")),
                pre_roll_ms=int(e("HARU_VAD_PRE_ROLL_MS", "300")),
                max_utterance_s=float(e("HARU_VAD_MAX_UTTERANCE_S", "30")),
            ),
        )


# --- VAD -------------------------------------------------------------------------

class SileroVAD:
    """Shared ONNX session; per-stream recurrent state lives in VADState."""

    def __init__(self, model_path: str) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def new_state(self) -> VADState:
        return VADState()

    def prob(self, st: VADState, chunk: np.ndarray) -> float:
        x = np.concatenate([st.context, chunk])[np.newaxis, :]
        out, st.state = self.session.run(None, {"input": x, "state": st.state, "sr": self._sr})
        st.context = chunk[-VAD_CONTEXT:]
        return float(out[0, 0])


@dataclass
class VADState:
    state: np.ndarray = field(default_factory=lambda: np.zeros((2, 1, 128), np.float32))
    context: np.ndarray = field(default_factory=lambda: np.zeros(VAD_CONTEXT, np.float32))


@dataclass
class Utterance:
    stream_label: str
    pcm: bytes                 # s16le mono 16 kHz, including pre-roll
    speech_ms: int
    last_captured_at_us: int   # terminal clock: capture time of the final frame
    ended_at: float            # monotonic: when the VAD decided it was over
    forced: bool = False       # cut at max_utterance_s rather than by silence


VADEvent = Callable[[str, str], None]  # (stream label, "speech_start" | "speech_end")


class Segmenter:
    """Turns one stream's 20 ms frames into utterances."""

    def __init__(self, vad: SileroVAD, cfg: VADConfig, label: str, on_event: VADEvent | None = None) -> None:
        self.on_event = on_event
        self.vad = vad
        self.cfg = cfg
        self.label = label
        self.state = vad.new_state()
        self.pending = np.zeros(0, np.float32)   # samples not yet fed to the VAD
        self.pre_roll: deque[np.ndarray] = deque(maxlen=max(1, cfg.pre_roll_ms // CHUNK_MS))
        self.speech: list[np.ndarray] | None = None
        self.speech_chunks = 0
        self.silence_ms = 0
        self.last_captured_at_us = 0

    def feed(self, frame: Frame) -> list[Utterance]:
        samples = np.frombuffer(frame.payload, dtype="<i2").astype(np.float32) / 32768.0
        self.pending = np.concatenate([self.pending, samples])
        self.last_captured_at_us = frame.captured_at_us
        out = []
        while len(self.pending) >= VAD_CHUNK:
            chunk, self.pending = self.pending[:VAD_CHUNK], self.pending[VAD_CHUNK:]
            if (u := self._step(chunk)) is not None:
                out.append(u)
        return out

    def _step(self, chunk: np.ndarray) -> Utterance | None:
        p = self.vad.prob(self.state, chunk)
        if self.speech is None:
            if p >= self.cfg.start_threshold:
                self.speech = list(self.pre_roll) + [chunk]
                self.speech_chunks = 1
                self.silence_ms = 0
                self._event("speech_start")
            else:
                self.pre_roll.append(chunk)
            return None

        self.speech.append(chunk)
        if p >= self.cfg.end_threshold:
            self.speech_chunks += 1 + self.silence_ms // CHUNK_MS  # brief dips count as speech
            self.silence_ms = 0
        else:
            self.silence_ms += CHUNK_MS
        if self.silence_ms >= self.cfg.min_silence_ms:
            return self._finish(forced=False)
        if len(self.speech) * CHUNK_MS >= self.cfg.max_utterance_s * 1000:
            return self._finish(forced=True)
        return None

    def _event(self, name: str) -> None:
        if self.on_event:
            try:
                self.on_event(self.label, name)
            except Exception:
                log.exception("VAD event handler failed")

    def _finish(self, forced: bool) -> Utterance | None:
        self._event("speech_end")
        speech, self.speech = self.speech, None
        self.pre_roll.clear()
        speech_ms = self.speech_chunks * CHUNK_MS
        if not forced:
            # Drop most of the trailing silence; keep ~150 ms so the last word isn't clipped.
            keep_tail = max(0, (self.silence_ms - 150) // CHUNK_MS)
            if keep_tail:
                speech = speech[:-keep_tail]
        self.silence_ms = 0
        if speech_ms < self.cfg.min_speech_ms:
            log.debug("%s: discarded %d ms blip", self.label, speech_ms)
            return None
        pcm = (np.clip(np.concatenate(speech), -1, 1) * 32767).astype("<i2").tobytes()
        return Utterance(self.label, pcm, speech_ms, self.last_captured_at_us,
                         time.monotonic(), forced)


# --- transcription ------------------------------------------------------------------

@dataclass
class Transcript:
    stream_label: str
    text: str
    audio_ms: int
    stt_ms: float               # HTTP round trip to the STT server
    end_to_text_ms: float       # VAD end-of-utterance -> text available
    speech_end_to_text_ms: float  # last captured frame -> text (terminal clock vs ours)
    dropped_segments: int = 0   # segments Whisper itself flagged as non-speech


TranscriptHandler = Callable[[Transcript], Awaitable[None] | None]


def wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


class SpeechToText:
    """Owns the VAD, the transcription queue and the STT server's health."""

    def __init__(self, config: STTConfig, on_status: Callable[[str], Awaitable[None]] | None = None) -> None:
        self.config = config
        self.vad = SileroVAD(config.vad.model_path)
        self.status = "unknown"
        self._on_status = on_status
        self._handlers: list[TranscriptHandler] = []
        self._vad_handlers: list[VADEvent] = []
        self._segmenters: dict[str, Segmenter] = {}
        self._queue: asyncio.Queue[Utterance] = asyncio.Queue(maxsize=20)
        self._client = httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=60)

    def on_transcript(self, handler: TranscriptHandler) -> None:
        self._handlers.append(handler)

    def on_vad(self, handler: VADEvent) -> None:
        """(stream label, "speech_start" | "speech_end") as the VAD opens and closes an utterance."""
        self._vad_handlers.append(handler)

    def _vad_event(self, label: str, event: str) -> None:
        for h in self._vad_handlers:
            h(label, event)

    # AudioSink interface (called synchronously from the gateway's event loop).

    def on_start(self, stream: AudioStream) -> None:
        if stream.sample_rate != SAMPLE_RATE or stream.channels != 1:
            log.error("STT needs 16 kHz mono; ignoring stream %s (%d Hz x%d)",
                      stream.label, stream.sample_rate, stream.channels)
            return
        self._segmenters[stream.label] = Segmenter(self.vad, self.config.vad, stream.label, self._vad_event)

    def on_audio(self, stream: AudioStream, frame: Frame, received_at_us: int) -> None:
        seg = self._segmenters.get(stream.label)
        if seg is None:
            return
        for utt in seg.feed(frame):
            try:
                self._queue.put_nowait(utt)
            except asyncio.QueueFull:
                log.warning("STT backlog full; dropping %d ms utterance from %s", utt.speech_ms, utt.stream_label)

    def on_stop(self, stream: AudioStream, reason: str) -> None:
        # An utterance cut off by a disconnect is dropped; half a sentence isn't worth transcribing.
        self._segmenters.pop(stream.label, None)

    # Background work.

    async def run(self) -> None:
        await asyncio.gather(self._health_forever(), self._transcribe_forever())

    async def close(self) -> None:
        await self._client.aclose()

    async def _set_status(self, status: str) -> None:
        if status == self.status:
            return
        log.log(logging.INFO if status in ("ok", "loading", "downloading") else logging.WARNING,
                "STT %s (%s at %s)", status, self.config.model, self.config.base_url)
        self.status = status
        if self._on_status:
            await self._on_status(status)

    async def _health_forever(self) -> None:
        while True:
            try:
                await self._check_and_warm()
            except httpx.HTTPError as exc:
                log.debug("STT health check failed: %s", exc)
                await self._set_status("unavailable")
            await asyncio.sleep(self.config.health_interval_s if self.status == "ok" else 5)

    async def _check_and_warm(self) -> None:
        base = self.config.base_url
        models = (await self._client.get(f"{base}/models")).raise_for_status().json()
        if self.config.model not in {m.get("id") for m in models.get("data", [])}:
            # Speaches downloads on request; this can take minutes the first time.
            await self._set_status("downloading")
            await self._client.post(f"{base}/models/{self.config.model}", timeout=1800)
        if self.status != "ok":
            # Transcribe a short silence to load the model into VRAM now, rather
            # than making the first real utterance wait (~25 s cold on this box).
            await self._set_status("loading")
            await self._post_transcription(bytes(SAMPLE_RATE // 2 * 2), timeout=300)
        await self._set_status("ok")

    async def _post_transcription(self, pcm: bytes, timeout: float = 60) -> dict:
        # vad_filter off: our Silero VAD already cut the silence; running it twice just costs time.
        data = {"model": self.config.model, "response_format": "verbose_json", "temperature": "0",
                "vad_filter": "false"}
        if self.config.language:
            data["language"] = self.config.language
        resp = await self._client.post(
            f"{self.config.base_url}/audio/transcriptions",
            data=data, files={"file": ("utterance.wav", wav_bytes(pcm), "audio/wav")}, timeout=timeout)
        return resp.raise_for_status().json()

    async def _transcribe_forever(self) -> None:
        while True:
            utt = await self._queue.get()
            try:
                await self._transcribe(utt)
            except httpx.HTTPError as exc:
                log.warning("transcription of %d ms utterance failed: %s", utt.speech_ms, exc)
                await self._set_status("unavailable")
            except Exception:
                log.exception("transcription failed")

    async def _transcribe(self, utt: Utterance) -> None:
        t0 = time.monotonic()
        result = await self._post_transcription(utt.pcm)
        stt_ms = (time.monotonic() - t0) * 1000
        kept, dropped = [], 0
        for seg in result.get("segments") or []:
            if (seg.get("no_speech_prob", 0) > self.config.no_speech_prob_max
                    and seg.get("avg_logprob", 0) < self.config.avg_logprob_min):
                dropped += 1
                log.info("dropping non-speech segment %r (no_speech %.2f, logprob %.2f)",
                         seg.get("text"), seg["no_speech_prob"], seg["avg_logprob"])
                continue
            kept.append(seg.get("text", ""))
        text = ("".join(kept) if result.get("segments") is not None else result.get("text", "")).strip()
        now_us = time.time_ns() // 1000
        tr = Transcript(
            stream_label=utt.stream_label,
            text=text,
            audio_ms=len(utt.pcm) * 1000 // (2 * SAMPLE_RATE),
            stt_ms=round(stt_ms, 1),
            end_to_text_ms=round((time.monotonic() - utt.ended_at) * 1000, 1),
            speech_end_to_text_ms=round((now_us - utt.last_captured_at_us) / 1000, 1),
            dropped_segments=dropped,
        )
        if not text:
            log.info("%s: no speech in %d ms utterance", tr.stream_label, tr.audio_ms)
            return
        log.info("heard %s: %r (%d ms audio%s; STT %.0f ms, end-of-speech->text %.0f ms)",
                 tr.stream_label, tr.text, tr.audio_ms, ", cut at max length" if utt.forced else "",
                 tr.stt_ms, tr.speech_end_to_text_ms)
        await self._set_status("ok")
        for h in self._handlers:
            try:
                r = h(tr)
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                log.exception("transcript handler failed")
