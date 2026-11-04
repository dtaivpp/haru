"""Haru terminal <-> backend wire protocol, version 1.

Pure stdlib so both the Pi terminal and the backend gateway can depend on it
without dragging anything else onto the Pi. PROTOCOL.md is the human-readable
spec; this module is the executable one — keep them in sync.

Two kinds of WebSocket frames:

* Text frames carry a JSON envelope: {"version", "type", "timestamp", "data"}.
* Binary frames carry streaming payloads (audio) behind a 16-byte big-endian
  header: version (u8), kind (u8), stream id (u16), sequence number (u32),
  capture time (u64, Unix microseconds, sender's clock).
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

PROTOCOL_VERSION = 1

# The WebSocket path the gateway serves terminals on.
TERMINAL_PATH = "/terminal"
# Display clients (the avatar page on a terminal's screen): JSON only, receive-only.
#   ws://<backend>:8765/display?device=<device_id>&token=<terminal token>
# Backend -> display:
#   avatar.state {"state": "idle"|"listening"|"thinking"|"speaking", "emotion": "<name>",
#                 "stream_id": <audio.output stream, when speaking>}
#   avatar.mouth {"stream_id": 3, "fps": 50, "values": [0..1, ...]}  appended per sentence,
#                 played from the moment that stream starts speaking
#   avatar.gesture {"name": "wave"}  a one-off body motion, played as she starts speaking
DISPLAY_PATH = "/display"

# Emotions the avatar understands (the same vocabulary as Project AIRI's ACT markers).
EMOTIONS = ("neutral", "happy", "sad", "angry", "surprised", "think", "awkward", "question", "curious")
GESTURES = ("wave", "celebrate", "shrug")


class MessageType(StrEnum):
    # Pi -> backend
    AUDIO_INPUT = "audio.input"
    BUTTON_PRESS = "button.press"
    BUTTON_RELEASE = "button.release"
    CAMERA_FRAME = "camera.frame"
    DEVICE_STATUS = "device.status"
    DEVICE_HEARTBEAT = "device.heartbeat"
    # Backend -> Pi
    AUDIO_OUTPUT = "audio.output"
    DISPLAY_STATE = "display.state"
    AVATAR_STATE = "avatar.state"
    LED_STATE = "led.state"
    DEVICE_COMMAND = "device.command"
    BACKEND_STATUS = "backend.status"
    AVATAR_MOUTH = "avatar.mouth"
    AVATAR_GESTURE = "avatar.gesture"


TERMINAL_TO_BACKEND = frozenset({
    MessageType.AUDIO_INPUT,
    MessageType.BUTTON_PRESS,
    MessageType.BUTTON_RELEASE,
    MessageType.CAMERA_FRAME,
    MessageType.DEVICE_STATUS,
    MessageType.DEVICE_HEARTBEAT,
})
BACKEND_TO_TERMINAL = frozenset(MessageType) - TERMINAL_TO_BACKEND


class CloseCode(IntEnum):
    """Application close codes (4000-4999 is the range RFC 6455 reserves for apps)."""

    UNSUPPORTED_VERSION = 4002
    BAD_HELLO = 4003
    REPLACED = 4004  # another connection claimed the same device_id


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Message:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    version: int = PROTOCOL_VERSION


def encode(type: str, data: dict[str, Any] | None = None) -> str:
    return json.dumps({
        "version": PROTOCOL_VERSION,
        "type": str(type),
        "timestamp": time.time(),
        "data": data or {},
    }, separators=(",", ":"))


def decode(raw: str | bytes) -> Message:
    """Parse a text-frame envelope. Unknown types are allowed through (forward
    compatibility — the receiver decides whether to ignore them); a wrong
    version or malformed envelope is not."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProtocolError(f"not JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError("envelope must be a JSON object")
    version = obj.get("version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version!r}")
    type_ = obj.get("type")
    if not isinstance(type_, str) or not type_:
        raise ProtocolError("missing 'type'")
    data = obj.get("data", {})
    if not isinstance(data, dict):
        raise ProtocolError("'data' must be an object")
    timestamp = obj.get("timestamp", 0.0)
    if not isinstance(timestamp, (int, float)):
        raise ProtocolError("'timestamp' must be a number")
    return Message(type=type_, data=data, timestamp=float(timestamp), version=version)


# --- Binary frames -----------------------------------------------------------

class FrameKind(IntEnum):
    AUDIO_INPUT = 1   # Pi -> backend, mic PCM
    AUDIO_OUTPUT = 2  # backend -> Pi, speaker PCM


_FRAME_HEADER = struct.Struct("!BBHIQ")
FRAME_HEADER_SIZE = _FRAME_HEADER.size


@dataclass(frozen=True)
class Frame:
    kind: FrameKind
    stream_id: int
    seq: int              # per-stream, starts at 0; a jump means frames were dropped
    captured_at_us: int   # sender's wall clock when the payload was captured
    payload: bytes


def encode_frame(kind: FrameKind, stream_id: int, seq: int, captured_at_us: int,
                 payload: bytes) -> bytes:
    return _FRAME_HEADER.pack(PROTOCOL_VERSION, kind, stream_id,
                              seq & 0xFFFFFFFF, captured_at_us) + payload


def decode_frame(raw: bytes) -> Frame:
    if len(raw) < FRAME_HEADER_SIZE:
        raise ProtocolError("binary frame shorter than header")
    version, kind, stream_id, seq, captured_at_us = _FRAME_HEADER.unpack_from(raw)
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported binary frame version {version}")
    try:
        kind = FrameKind(kind)
    except ValueError as exc:
        raise ProtocolError(f"unknown binary frame kind {kind}") from exc
    return Frame(kind, stream_id, seq, captured_at_us, raw[FRAME_HEADER_SIZE:])


# --- audio stream control ------------------------------------------------------
# JSON audio.input / audio.output messages bracket a binary stream:
#   {"event": "start", "stream_id": 3, "encoding": "pcm_s16le",
#    "sample_rate": 16000, "channels": 1, "frame_ms": 20}
#   {"event": "stop", "stream_id": 3, "reason": "..."}

#
# audio.output (backend -> terminal) uses the same start/stop, where stop means
# "that's all; finish playing what you have". It adds:
#   {"event": "cancel", "stream_id": 3}  -> drop everything queued, go quiet now
#
# The terminal reports playback progress with device.status:
#   {"playback": {"event": "started" | "finished" | "cancelled",
#                 "stream_id": 3, "played_ms": 1840}}
#
# A terminal with a speaker advertises its format in its hello (device.status):
#   "audio_output": {"encoding": "pcm_s16le", "sample_rate": 24000, "channels": 1}
# and the backend sends audio in exactly that format; the terminal never resamples.

AUDIO_START = "start"
AUDIO_STOP = "stop"
AUDIO_CANCEL = "cancel"
PCM_S16LE = "pcm_s16le"

PLAYBACK_STARTED = "started"
PLAYBACK_FINISHED = "finished"
PLAYBACK_CANCELLED = "cancelled"
