"""Speaker output: a permanently open `aplay` fed in real time.

The output stream is never closed. When the backend has nothing to say it
plays silence, and that matters: the Jabra Speak2 40 UC delivers **no
microphone audio unless its playback stream is open** (arecord fails with EIO
within milliseconds otherwise, likely because its echo canceller is clocked
off the output). An always-open output also means speech starts with no
device-open delay.

The backend sends each reply as an audio.output stream (start, binary frames,
stop), usually faster than real time. Frames are buffered here and played at
wall-clock pace with a small lead, so buffered silence never delays speech by
more than about `lead_ms`. `cancel` (barge-in) drops the buffer instantly.
The terminal only plays what it's given, in the format it advertised; it
never resamples or interprets it.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import struct
import termios
import time

from haru_protocol import (
    AUDIO_CANCEL,
    AUDIO_START,
    AUDIO_STOP,
    PCM_S16LE,
    PLAYBACK_CANCELLED,
    PLAYBACK_FINISHED,
    PLAYBACK_STARTED,
    FrameKind,
    Message,
    MessageType,
    ProtocolError,
    decode_frame,
)

from haru_terminal.config import AudioOutputConfig
from haru_terminal.transport import BackendLink

log = logging.getLogger("haru.terminal.audio_out")

RESTART_BACKOFF_MAX_S = 10.0
MAX_BUFFER_S = 120.0  # a runaway backend can't eat the Pi's RAM
PIPE_BYTES = 4096     # aplay's stdin pipe: rounded up to one page (16 KB on the Pi 5)


class Speaker:
    def __init__(self, link: BackendLink, config: AudioOutputConfig) -> None:
        self.link = link
        self.config = config
        self.frame_bytes = config.sample_rate * config.channels * 2 * config.frame_ms // 1000
        self.running = False
        self.restarts = 0
        self.underruns = 0
        # Current output stream.
        self._stream_id: int | None = None
        self._buffer = bytearray()
        self._ending = False     # got "stop": play out the buffer, then report finished
        self._started = False    # reported "started" for this stream yet
        self._played_bytes = 0
        self._expected_seq = 0
        self.streams_played = 0
        link.add_heartbeat_fields(self.status)
        link.add_hello_fields(lambda: {"audio_output": {
            "encoding": PCM_S16LE, "sample_rate": config.sample_rate, "channels": config.channels}})
        link.on(MessageType.AUDIO_OUTPUT, self._on_control)
        link.on_frame(self._on_frame)

    def status(self) -> dict:
        return {"audio_out": {"running": self.running, "restarts": self.restarts,
                              "underruns": self.underruns, "streams_played": self.streams_played,
                              "playing": self._stream_id is not None,
                              "buffered_ms": self._bytes_to_ms(len(self._buffer))}}

    def _bytes_to_ms(self, n: int) -> int:
        return n * 1000 // (self.config.sample_rate * self.config.channels * 2)

    # --- backend -> speaker -------------------------------------------------

    async def _on_control(self, msg: Message) -> None:
        event, stream_id = msg.data.get("event"), msg.data.get("stream_id")
        if event == AUDIO_START:
            fmt = (msg.data.get("encoding"), msg.data.get("sample_rate"), msg.data.get("channels"))
            want = (PCM_S16LE, self.config.sample_rate, self.config.channels)
            if fmt != want:
                log.error("refusing output stream %s: format %s, speaker is %s", stream_id, fmt, want)
                return
            if self._stream_id is not None:
                await self._finish(PLAYBACK_CANCELLED)  # a new reply replaces the old
            self._stream_id = stream_id
            self._buffer.clear()
            self._ending = self._started = False
            self._played_bytes = 0
            self._expected_seq = 0
        elif stream_id != self._stream_id:
            return  # stale control for a stream we already dropped
        elif event == AUDIO_STOP:
            self._ending = True
        elif event == AUDIO_CANCEL:
            await self._finish(PLAYBACK_CANCELLED)

    def _on_frame(self, raw: bytes) -> None:
        try:
            frame = decode_frame(raw)
        except ProtocolError as exc:
            log.warning("bad frame from backend: %s", exc)
            return
        if frame.kind != FrameKind.AUDIO_OUTPUT or frame.stream_id != self._stream_id:
            return
        if frame.seq != self._expected_seq:
            log.warning("output stream %s: expected frame %d, got %d", frame.stream_id,
                        self._expected_seq, frame.seq)
        self._expected_seq = frame.seq + 1
        if self._bytes_to_ms(len(self._buffer)) > MAX_BUFFER_S * 1000:
            log.warning("output buffer over %.0fs; dropping audio", MAX_BUFFER_S)
            return
        self._buffer += frame.payload

    async def _finish(self, event: str) -> None:
        stream_id, played = self._stream_id, self._bytes_to_ms(self._played_bytes)
        self._stream_id = None
        self._buffer.clear()
        self._ending = False
        if stream_id is None:
            return
        self.streams_played += 1
        log.info("playback %s: stream %s, %d ms played", event, stream_id, played)
        await self.link.send(MessageType.DEVICE_STATUS, {"playback": {
            "event": event, "stream_id": stream_id, "played_ms": played}})

    # --- aplay --------------------------------------------------------------

    async def run_forever(self) -> None:
        failures = 0
        while True:
            started = time.monotonic()
            try:
                await self._play_once()
            except Exception:
                log.exception("speaker output crashed")
            self.running = False
            if time.monotonic() - started > 30:
                failures = 0
            delay = min(RESTART_BACKOFF_MAX_S, 2 ** failures)
            failures += 1
            self.restarts += 1
            log.warning("speaker output stopped; restarting in %.0fs", delay)
            await asyncio.sleep(delay)

    def _next_chunk(self) -> tuple[bytes, bool]:
        """(20 ms of audio to write, whether it's real speech)."""
        n = self.frame_bytes
        if self._stream_id is not None and len(self._buffer) >= n:
            chunk = bytes(self._buffer[:n])
            del self._buffer[:n]
            return chunk, True
        if self._stream_id is not None and self._ending and self._buffer:
            chunk = bytes(self._buffer) + bytes(n - len(self._buffer))  # pad the tail
            self._buffer.clear()
            return chunk, True
        return bytes(n), False

    async def _play_once(self) -> None:
        c = self.config
        cmd = [
            "aplay", "-q", "-D", c.device, "-t", "raw", "-f", "S16_LE",
            "-r", str(c.sample_rate), "-c", str(c.channels),
            f"--period-time={c.frame_ms * 1000}", f"--buffer-time={c.frame_ms * 1000 * 5}",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        # Keep the audio queued between us and the speaker near lead_ms. We write at
        # wall-clock pace but the device plays at its own clock, a hair slower, and the
        # difference piled up in aplay's stdin pipe (256 KB on the Pi, ~5 s at 24 kHz)
        # over days of uptime: "started" (and the avatar's mouth) ran seconds ahead of the
        # voice. Now the loop skips a beat whenever the pipe holds more than the lead, and
        # a minimum-size pipe (one page) with no asyncio-side buffer is the hard cap.
        pipe_fd = proc.stdin.get_extra_info("pipe").fileno()
        try:
            fcntl.fcntl(pipe_fd, fcntl.F_SETPIPE_SZ, PIPE_BYTES)
        except (AttributeError, OSError) as exc:
            log.warning("can't shrink the speaker pipe (%s)", exc)
        proc.stdin.transport.set_write_buffer_limits(high=0)
        log.info("speaker output started: %s", " ".join(cmd))
        stderr_task = asyncio.create_task(self._watch_stderr(proc))
        frame_s = c.frame_ms / 1000
        lead_s = c.lead_ms / 1000
        lead_bytes = self.frame_bytes * c.lead_ms // c.frame_ms
        self.running = True
        next_t = time.monotonic()
        try:
            while proc.returncode is None:
                chunk, speech = self._next_chunk()
                if speech and not self._started:
                    self._started = True
                    await self.link.send(MessageType.DEVICE_STATUS, {"playback": {
                        "event": PLAYBACK_STARTED, "stream_id": self._stream_id, "played_ms": 0}})
                if speech:
                    self._played_bytes += len(chunk)
                elif self._stream_id is not None and self._ending and not self._buffer:
                    await self._finish(PLAYBACK_FINISHED)
                proc.stdin.write(chunk)
                await proc.stdin.drain()
                next_t += frame_s
                if _queued(pipe_fd) > lead_bytes + self.frame_bytes:
                    next_t += frame_s  # the device is behind our clock: let it catch up
                # Stay at most lead_s ahead of real time. If we fell behind
                # (the Pi stalled), resync instead of bursting to catch up.
                now = time.monotonic()
                if next_t < now:
                    next_t = now
                await asyncio.sleep(max(0.0, next_t - lead_s - now))
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.running = False
            if proc.returncode is None:
                proc.terminate()
            rc = await proc.wait()
            await stderr_task
            log.info("aplay exited with %s", rc)

    async def _watch_stderr(self, proc: asyncio.subprocess.Process) -> None:
        async for line in proc.stderr:
            text = line.decode(errors="replace").rstrip()
            if "underrun" in text:
                self.underruns += 1
                if self.underruns == 1 or self.underruns % 50 == 0:
                    log.warning("speaker underruns: %d", self.underruns)
            else:
                log.warning("aplay: %s", text)


def _queued(fd: int) -> int:
    """Bytes written to a pipe that its reader hasn't taken yet."""
    try:
        return struct.unpack("i", fcntl.ioctl(fd, termios.FIONREAD, b"\0\0\0\0"))[0]
    except OSError:
        return 0
