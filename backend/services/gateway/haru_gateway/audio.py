"""Inbound audio: stream bookkeeping plus pluggable consumers ("sinks").

The terminal brackets each binary stream with audio.input start/stop messages.
`AudioRouter` tracks those streams per terminal (format, sequence gaps) and
fans every frame out to the configured sinks. Stage 2 ships two sinks, stats
logging and an optional rolling WAV recorder. Stage 3's VAD/STT is just
another sink, so nothing on the Pi changes when it lands.
"""

from __future__ import annotations

import array
import logging
import math
import os
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from haru_protocol import AUDIO_START, AUDIO_STOP, PCM_S16LE, Frame, FrameKind, ProtocolError

log = logging.getLogger("haru.gateway.audio")

SUPPORTED_ENCODINGS = {PCM_S16LE}


@dataclass
class AudioStream:
    device_id: str
    stream_id: int
    encoding: str
    sample_rate: int
    channels: int
    frame_ms: int
    started_at: float = field(default_factory=time.time)
    next_seq: int = 0
    frames: int = 0
    lost_frames: int = 0

    @property
    def label(self) -> str:
        return f"{self.device_id}/{self.stream_id}"


class AudioSink(Protocol):
    def on_start(self, stream: AudioStream) -> None: ...
    def on_audio(self, stream: AudioStream, frame: Frame, received_at_us: int) -> None: ...
    def on_stop(self, stream: AudioStream, reason: str) -> None: ...


class AudioRouter:
    """One per terminal session."""

    def __init__(self, device_id: str, sinks: list[AudioSink]) -> None:
        self.device_id = device_id
        self.sinks = sinks
        self.streams: dict[int, AudioStream] = {}

    def handle_control(self, data: dict) -> None:
        event = data.get("event")
        stream_id = data.get("stream_id")
        if not isinstance(stream_id, int):
            raise ProtocolError("audio.input needs an integer stream_id")
        if event == AUDIO_START:
            encoding = data.get("encoding")
            if encoding not in SUPPORTED_ENCODINGS:
                raise ProtocolError(f"unsupported audio encoding {encoding!r}")
            if stream_id in self.streams:
                self.stop(stream_id, "restarted")
            stream = AudioStream(
                device_id=self.device_id,
                stream_id=stream_id,
                encoding=encoding,
                sample_rate=int(data.get("sample_rate", 16000)),
                channels=int(data.get("channels", 1)),
                frame_ms=int(data.get("frame_ms", 20)),
            )
            self.streams[stream_id] = stream
            log.info("audio stream %s started: %s %d Hz x%d, %d ms frames", stream.label,
                     stream.encoding, stream.sample_rate, stream.channels, stream.frame_ms)
            self._fan_out("on_start", stream)
        elif event == AUDIO_STOP:
            self.stop(stream_id, str(data.get("reason", "stopped")))
        else:
            raise ProtocolError(f"unknown audio.input event {event!r}")

    def handle_frame(self, frame: Frame) -> None:
        if frame.kind != FrameKind.AUDIO_INPUT:
            raise ProtocolError(f"unexpected {frame.kind.name} frame from a terminal")
        stream = self.streams.get(frame.stream_id)
        if stream is None:
            # E.g. frames racing a stop; nothing to attach them to.
            log.debug("frame for unknown stream %s/%d", self.device_id, frame.stream_id)
            return
        received_at_us = time.time_ns() // 1000
        if frame.seq != stream.next_seq:
            gap = (frame.seq - stream.next_seq) & 0xFFFFFFFF
            stream.lost_frames += gap
            log.warning("audio stream %s lost %d frame(s) before seq %d", stream.label, gap, frame.seq)
        stream.next_seq = (frame.seq + 1) & 0xFFFFFFFF
        stream.frames += 1
        self._fan_out("on_audio", stream, frame, received_at_us)

    def stop(self, stream_id: int, reason: str) -> None:
        stream = self.streams.pop(stream_id, None)
        if stream is None:
            return
        log.info("audio stream %s stopped (%s): %d frames, %d lost, %.0fs",
                 stream.label, reason, stream.frames, stream.lost_frames, time.time() - stream.started_at)
        self._fan_out("on_stop", stream, reason)

    def close(self, reason: str) -> None:
        for stream_id in list(self.streams):
            self.stop(stream_id, reason)

    def _fan_out(self, method: str, *args) -> None:
        for sink in self.sinks:
            try:
                getattr(sink, method)(*args)
            except Exception:
                # One broken consumer (say, a crashed STT) must not starve the others.
                log.exception("audio sink %s.%s failed", type(sink).__name__, method)


# --- sinks -----------------------------------------------------------------------

def pcm16_samples(payload: bytes) -> array.array:
    samples = array.array("h")
    samples.frombytes(payload[: len(payload) - len(payload) % 2])
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def dbfs(value: float) -> float:
    return 20 * math.log10(value / 32768) if value > 0 else -120.0


@dataclass
class _Window:
    started: float = field(default_factory=time.monotonic)
    frames: int = 0
    lost_at_start: int = 0
    sum_sq: float = 0.0
    samples: int = 0
    peak: int = 0
    clipped: int = 0
    latencies_ms: list[float] = field(default_factory=list)


class StatsSink:
    """Logs level, clipping, loss and latency per stream every `interval_s`.

    Latency is the terminal's capture timestamp compared with the gateway's
    arrival time. Both machines run NTP, so the figure includes whatever
    clock offset remains between them (typically a few ms on a LAN).
    """

    def __init__(self, interval_s: float = 10.0) -> None:
        self.interval_s = interval_s
        self.windows: dict[str, _Window] = {}
        self.last_report: dict[str, dict] = {}

    def on_start(self, stream: AudioStream) -> None:
        self.windows[stream.label] = _Window()

    def on_audio(self, stream: AudioStream, frame: Frame, received_at_us: int) -> None:
        w = self.windows.setdefault(stream.label, _Window())
        samples = pcm16_samples(frame.payload)
        w.frames += 1
        w.samples += len(samples)
        w.sum_sq += sum(s * s for s in samples)
        if samples:
            w.peak = max(w.peak, max(samples), -min(samples))
            w.clipped += sum(1 for s in samples if s >= 32767 or s <= -32768)
        w.latencies_ms.append((received_at_us - frame.captured_at_us) / 1000)
        if time.monotonic() - w.started >= self.interval_s:
            self._report(stream, w)
            self.windows[stream.label] = _Window(lost_at_start=stream.lost_frames)

    def on_stop(self, stream: AudioStream, reason: str) -> None:
        w = self.windows.pop(stream.label, None)
        if w is not None and w.frames:
            self._report(stream, w)

    def _report(self, stream: AudioStream, w: _Window) -> None:
        elapsed = time.monotonic() - w.started
        lat = sorted(w.latencies_ms)
        report = {
            "fps": round(w.frames / elapsed, 1) if elapsed else 0.0,
            "expected_fps": round(1000 / stream.frame_ms, 1),
            "lost": stream.lost_frames - w.lost_at_start,
            "rms_dbfs": round(dbfs(math.sqrt(w.sum_sq / w.samples)) if w.samples else -120.0, 1),
            "peak_dbfs": round(dbfs(w.peak), 1),
            "clipped": w.clipped,
            "latency_ms_p50": round(statistics.median(lat), 1) if lat else None,
            "latency_ms_p95": round(lat[int(len(lat) * 0.95) - 1], 1) if lat else None,
            "latency_ms_max": round(lat[-1], 1) if lat else None,
        }
        self.last_report[stream.label] = report
        log.info("audio %s: %.1f fps (expect %.0f), lost %d, level %.1f dBFS rms / %.1f peak, "
                 "clipped %d, latency p50 %s / p95 %s / max %s ms",
                 stream.label, report["fps"], report["expected_fps"], report["lost"],
                 report["rms_dbfs"], report["peak_dbfs"], report["clipped"],
                 report["latency_ms_p50"], report["latency_ms_p95"], report["latency_ms_max"])


class WavRecorderSink:
    """Debug recorder: rolling WAV segments, pruned to the newest `keep` files.

    Off by default because it writes an always-on mic to disk. Turn it on to
    check audio quality by ear, then turn it back off.
    """

    def __init__(self, directory: Path, segment_s: float = 60.0, keep: int = 10) -> None:
        self.directory = directory
        self.segment_s = segment_s
        self.keep = keep
        self.files: dict[str, tuple[wave.Wave_write, float, Path]] = {}
        directory.mkdir(parents=True, exist_ok=True)

    def on_start(self, stream: AudioStream) -> None:
        self._open(stream)

    def on_audio(self, stream: AudioStream, frame: Frame, received_at_us: int) -> None:
        entry = self.files.get(stream.label)
        if entry is None or time.monotonic() - entry[1] >= self.segment_s:
            self._open(stream)
            entry = self.files[stream.label]
        entry[0].writeframes(frame.payload)  # (not ...raw) keeps the header valid mid-segment

    def on_stop(self, stream: AudioStream, reason: str) -> None:
        self._close(stream.label)

    def _open(self, stream: AudioStream) -> None:
        self._close(stream.label)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = self.directory / f"{stream.device_id}-s{stream.stream_id}-{stamp}.wav"
        wf = wave.open(str(path), "wb")
        wf.setnchannels(stream.channels)
        wf.setsampwidth(2)
        wf.setframerate(stream.sample_rate)
        self.files[stream.label] = (wf, time.monotonic(), path)
        log.info("recording %s -> %s", stream.label, path.name)
        self._prune()

    def _close(self, label: str) -> None:
        entry = self.files.pop(label, None)
        if entry is not None:
            entry[0].close()  # rewrites the header with the final length

    def _prune(self) -> None:
        in_use = {p for _, _, p in self.files.values()}
        old = sorted(self.directory.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        for path in old[: max(0, len(old) - self.keep)]:
            if path not in in_use:
                path.unlink(missing_ok=True)


def sinks_from_env() -> list[AudioSink]:
    sinks: list[AudioSink] = [StatsSink(float(os.environ.get("HARU_AUDIO_STATS_INTERVAL_S", "10")))]
    record_dir = os.environ.get("HARU_AUDIO_RECORD_DIR", "")
    if record_dir:
        sinks.append(WavRecorderSink(
            Path(record_dir),
            segment_s=float(os.environ.get("HARU_AUDIO_RECORD_SEGMENT_S", "60")),
            keep=int(os.environ.get("HARU_AUDIO_RECORD_KEEP", "10")),
        ))
        log.warning("audio recording is ON -> %s (debug only)", record_dir)
    return sinks
