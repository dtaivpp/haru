"""Stage 7: avatar events for display clients (the face page on a terminal's screen).

A display is just a browser page on the Pi's monitor showing the Live2D avatar.
It holds no logic: it connects here, and we tell it what to look like. That covers
state (idle / listening / thinking / speaking), emotion, and a mouth-openness
timeline for each sentence, played in sync with the audio the terminal is playing.
"""

from __future__ import annotations

import array
import logging
import math
import sys
from collections import defaultdict

from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from haru_protocol import EMOTIONS, GESTURES, MessageType, encode

log = logging.getLogger("haru.gateway.display")

MOUTH_FPS = 50


def mouth_envelope(pcm: bytes, sample_rate: int, channels: int = 1, fps: int = MOUTH_FPS) -> list[float]:
    """Per-frame mouth openness (0..1) from s16le PCM: RMS, normalised with a soft knee."""
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    if sys.byteorder == "big":
        samples.byteswap()
    step = max(1, sample_rate * channels // fps)
    out = []
    for i in range(0, len(samples), step):
        chunk = samples[i:i + step]
        if not chunk:
            break
        rms = math.sqrt(sum(s * s for s in chunk) / len(chunk)) / 32768
        # Speech RMS sits around 0.02-0.25; map to 0..1 with a gentle curve so
        # quiet syllables still move the lips and loud ones don't just max out.
        out.append(round(min(1.0, max(0.0, (rms - 0.01) * 6.0) ** 0.7), 3))
    return out


class DisplayHub:
    def __init__(self) -> None:
        self.displays: dict[str, set[ServerConnection]] = defaultdict(set)
        self.state: dict[str, dict] = defaultdict(lambda: {"state": "idle", "emotion": "neutral"})

    async def attach(self, device_id: str, ws: ServerConnection) -> None:
        self.displays[device_id].add(ws)
        log.info("display for %s connected (%d attached)", device_id, len(self.displays[device_id]))
        try:
            await ws.send(encode(MessageType.AVATAR_STATE, dict(self.state[device_id])))
            async for _ in ws:
                pass  # displays don't talk back (yet)
        except ConnectionClosed:
            pass
        finally:
            self.displays[device_id].discard(ws)
            log.info("display for %s disconnected", device_id)

    async def _send(self, device_id: str, type_: MessageType, data: dict) -> None:
        for ws in list(self.displays.get(device_id, ())):
            try:
                await ws.send(encode(type_, data))
            except ConnectionClosed:
                self.displays[device_id].discard(ws)

    async def set_state(self, device_id: str, state: str | None = None, *,
                        emotion: str | None = None, stream_id: int | None = None) -> None:
        cur = self.state[device_id]
        before = (cur["state"], cur["emotion"])
        if emotion is not None:
            emotion = emotion.lower()
            cur["emotion"] = emotion if emotion in EMOTIONS else "neutral"
        if state is not None:
            cur["state"] = state
        cur.pop("stream_id", None)
        if stream_id is not None:
            cur["stream_id"] = stream_id
        if (cur["state"], cur["emotion"]) != before:
            log.info("avatar %s: %s / %s", device_id, cur["state"], cur["emotion"])
        await self._send(device_id, MessageType.AVATAR_STATE, dict(cur))

    async def gesture(self, device_id: str, name: str) -> None:
        name = name.lower()
        if name not in GESTURES:
            log.debug("ignoring unknown gesture %r", name)
            return
        log.info("avatar %s: gesture %s", device_id, name)
        await self._send(device_id, MessageType.AVATAR_GESTURE, {"name": name})

    async def mouth(self, device_id: str, stream_id: int, values: list[float]) -> None:
        await self._send(device_id, MessageType.AVATAR_MOUTH,
                         {"stream_id": stream_id, "fps": MOUTH_FPS, "values": values})
