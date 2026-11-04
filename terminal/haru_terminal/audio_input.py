"""Microphone -> backend streaming.

Capture is `arecord` writing raw PCM to a pipe. That means no PortAudio,
numpy or wheels on the Pi, and ALSA does whatever device-level conversion is
needed. Audio is streamed continuously while the backend is reachable. The
Pi does no VAD or wake word; deciding what's speech is the backend's job.

Capture and sending are decoupled by a small bounded queue: if the network
stalls, the oldest audio is dropped (stale speech is useless and must not
pile up), and arecord never overruns because we stopped reading it.
"""

from __future__ import annotations

import asyncio
import logging
import time

from haru_protocol import (
    AUDIO_START,
    AUDIO_STOP,
    PCM_S16LE,
    FrameKind,
    MessageType,
    encode_frame,
)

from haru_terminal.config import AudioInputConfig
from haru_terminal.transport import BackendLink, LinkState

log = logging.getLogger("haru.terminal.audio_in")

RESTART_BACKOFF_MAX_S = 10.0


class MicStreamer:
    def __init__(self, link: BackendLink, config: AudioInputConfig) -> None:
        self.link = link
        self.config = config
        self.queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(
            maxsize=max(1, config.queue_ms // config.frame_ms))
        self._stream_id = 0
        self._streaming = False
        self._paused = False  # stop() was called; don't auto-start a new stream
        self._seq = 0
        # Counters for the heartbeat.
        self.captured = 0
        self.sent = 0
        self.dropped = 0
        self.capture_restarts = 0
        self.last_send_delay_ms = 0.0
        link.on_state(self._on_link_state)
        link.add_heartbeat_fields(self.status)

    def status(self) -> dict:
        return {"audio_in": {
            "streaming": self._streaming,
            "captured": self.captured,
            "sent": self.sent,
            "dropped": self.dropped,
            "capture_restarts": self.capture_restarts,
            "send_delay_ms": round(self.last_send_delay_ms, 1),
        }}

    async def run_forever(self) -> None:
        await asyncio.gather(self._capture_forever(), self._send_forever())

    # --- capture ------------------------------------------------------------

    async def _capture_forever(self) -> None:
        failures = 0
        while True:
            started = time.monotonic()
            try:
                await self._capture_once()
            except Exception:
                log.exception("capture crashed")
            if time.monotonic() - started > 30:
                failures = 0  # it ran fine for a while; treat this as a fresh failure
            delay = min(RESTART_BACKOFF_MAX_S, 2 ** failures)
            failures += 1
            self.capture_restarts += 1
            log.warning("mic capture stopped; restarting in %.0fs", delay)
            await asyncio.sleep(delay)

    async def _capture_once(self) -> None:
        c = self.config
        cmd = [
            "arecord", "-q", "-D", c.device, "-t", "raw", "-f", "S16_LE",
            "-r", str(c.sample_rate), "-c", str(c.channels),
            f"--period-time={c.frame_ms * 1000}", f"--buffer-time={c.frame_ms * 1000 * 5}",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        log.info("mic capture started: %s", " ".join(cmd))
        stderr_task = asyncio.create_task(self._log_stderr(proc))
        frame_us = c.frame_ms * 1000
        try:
            while True:
                try:
                    pcm = await proc.stdout.readexactly(c.frame_bytes)
                except asyncio.IncompleteReadError:
                    break
                # The frame just finished capturing; stamp when it *started*.
                captured_at_us = time.time_ns() // 1000 - frame_us
                self.captured += 1
                if self.queue.full():
                    self.queue.get_nowait()
                    self.dropped += 1
                    if self.dropped == 1 or self.dropped % 50 == 0:
                        log.warning("send queue full; dropped %d frame(s) so far", self.dropped)
                self.queue.put_nowait((captured_at_us, pcm))
        finally:
            if proc.returncode is None:
                proc.terminate()
            rc = await proc.wait()
            await stderr_task
            log.info("arecord exited with %s", rc)

    async def _log_stderr(self, proc: asyncio.subprocess.Process) -> None:
        async for line in proc.stderr:
            log.warning("arecord: %s", line.decode(errors="replace").rstrip())

    # --- sending ------------------------------------------------------------

    async def _send_forever(self) -> None:
        while True:
            captured_at_us, pcm = await self.queue.get()
            if self._paused:
                continue
            if self.link.state != LinkState.ONLINE:
                self.dropped += 1  # offline: nowhere to send it; don't buffer stale audio
                continue
            if not self._streaming and not await self._start_stream():
                continue
            frame = encode_frame(FrameKind.AUDIO_INPUT, self._stream_id, self._seq, captured_at_us, pcm)
            if await self.link.send_frame(frame):
                self._seq += 1
                self.sent += 1
                self.last_send_delay_ms = (time.time_ns() // 1000 - captured_at_us) / 1000
            else:
                self._streaming = False

    async def _start_stream(self) -> bool:
        self._stream_id = (self._stream_id + 1) & 0xFFFF
        self._seq = 0
        c = self.config
        self._streaming = await self.link.send(MessageType.AUDIO_INPUT, {
            "event": AUDIO_START,
            "stream_id": self._stream_id,
            "encoding": PCM_S16LE,
            "sample_rate": c.sample_rate,
            "channels": c.channels,
            "frame_ms": c.frame_ms,
        })
        if self._streaming:
            log.info("streaming mic to backend as stream %d", self._stream_id)
        return self._streaming

    async def stop(self, reason: str) -> None:
        """End the current stream and send nothing more until resume()."""
        self._paused = True
        if self._streaming:
            self._streaming = False
            await self.link.send(MessageType.AUDIO_INPUT,
                                 {"event": AUDIO_STOP, "stream_id": self._stream_id, "reason": reason})

    def resume(self) -> None:
        self._paused = False  # the next captured frame opens a new stream

    def _on_link_state(self, state: LinkState) -> None:
        if state != LinkState.ONLINE:
            self._streaming = False  # a new connection gets a new stream
