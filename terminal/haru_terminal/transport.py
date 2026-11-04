"""Persistent, self-healing connection from the terminal to the backend.

Knows nothing about what messages *mean* — callers register handlers per
message type. The only things it owns are: connecting, the hello, heartbeats,
and reconnecting with exponential backoff when anything goes wrong.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import random
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

from haru_protocol import (
    PROTOCOL_VERSION,
    CloseCode,
    Message,
    MessageType,
    ProtocolError,
    decode,
    encode,
)

from haru_terminal import __version__
from haru_terminal.config import TerminalConfig

log = logging.getLogger("haru.terminal.transport")

Handler = Callable[[Message], Awaitable[None] | None]
FrameHandler = Callable[[bytes], Awaitable[None] | None]


class LinkState(StrEnum):
    CONNECTING = "connecting"
    ONLINE = "online"
    OFFLINE = "offline"


class BackendLink:
    def __init__(self, config: TerminalConfig, capabilities: list[str] | None = None) -> None:
        self.config = config
        self.capabilities = capabilities or []
        self.state = LinkState.OFFLINE
        self.started_at = time.monotonic()
        self._ws: ClientConnection | None = None
        self._handlers: dict[str, list[Handler]] = {}
        self._frame_handlers: list[FrameHandler] = []
        self._state_listeners: list[Callable[[LinkState], None]] = []
        self._heartbeat_fields: list[Callable[[], dict]] = []
        self._hello_fields: list[Callable[[], dict]] = []
        self._online = asyncio.Event()
        self._closing = False

    # --- registration -------------------------------------------------------

    def on(self, type_: str, handler: Handler) -> None:
        self._handlers.setdefault(str(type_), []).append(handler)

    def on_frame(self, handler: FrameHandler) -> None:
        self._frame_handlers.append(handler)

    def on_state(self, listener: Callable[[LinkState], None]) -> None:
        self._state_listeners.append(listener)

    def add_hello_fields(self, provider: Callable[[], dict]) -> None:
        """Peripherals describe themselves (e.g. speaker format) in the hello."""
        self._hello_fields.append(provider)

    def add_heartbeat_fields(self, provider: Callable[[], dict]) -> None:
        """Peripherals report their health by merging fields into every heartbeat."""
        self._heartbeat_fields.append(provider)

    # --- sending ------------------------------------------------------------

    async def wait_online(self, timeout: float | None = None) -> None:
        await asyncio.wait_for(self._online.wait(), timeout)

    async def send(self, type_: str, data: dict | None = None) -> bool:
        """Best-effort send. Returns False (never raises) while offline — the
        caller decides whether a dropped event matters."""
        ws = self._ws
        if ws is None or self.state != LinkState.ONLINE:
            return False
        try:
            await ws.send(encode(type_, data))
            return True
        except ConnectionClosed:
            return False

    async def send_frame(self, frame: bytes) -> bool:
        ws = self._ws
        if ws is None or self.state != LinkState.ONLINE:
            return False
        try:
            await ws.send(frame)
            return True
        except ConnectionClosed:
            return False

    # --- lifecycle ----------------------------------------------------------

    async def close(self) -> None:
        """Clean shutdown: tell the backend we're going away, stop reconnecting."""
        self._closing = True
        if self._ws is not None:
            await self._ws.close(1001, "terminal shutting down")

    async def run_forever(self) -> None:
        attempt = 0
        while not self._closing:
            self._set_state(LinkState.CONNECTING)
            try:
                async with await self._connect() as ws:
                    attempt = 0  # handshake + hello succeeded
                    await self._session(ws)
                if self._closing:
                    return
                log.warning("backend closed the connection (code %s)", ws.close_code)
            except InvalidStatus as exc:
                status = exc.response.status_code
                log.error("backend refused connection: HTTP %d%s", status,
                          " (check HARU_TERMINAL_TOKEN)" if status == 401 else "")
            except ConnectionClosed as exc:
                rcvd = exc.rcvd
                if rcvd is not None and rcvd.code == CloseCode.UNSUPPORTED_VERSION:
                    log.error("backend does not speak protocol v%d: %s", PROTOCOL_VERSION, rcvd.reason)
                else:
                    log.warning("connection closed: %s", exc)
            except (OSError, TimeoutError, InvalidHandshake) as exc:
                log.warning("cannot reach backend at %s: %s", self.config.backend_url,
                            exc or type(exc).__name__)
            finally:
                self._ws = None
                self._online.clear()
                self._set_state(LinkState.OFFLINE)
            if self._closing:
                return

            delay = min(self.config.backoff_max_s, self.config.backoff_initial_s * 2 ** attempt)
            delay *= random.uniform(0.5, 1.0)  # jitter
            attempt += 1
            log.info("reconnecting in %.1fs (attempt %d)", delay, attempt)
            await asyncio.sleep(delay)

    async def _connect(self) -> ClientConnection:
        return await connect(
            self.config.backend_url,
            additional_headers={"Authorization": f"Bearer {self.config.token}"},
            open_timeout=self.config.connect_timeout_s,
            ping_interval=10,
            ping_timeout=10,
            close_timeout=2,
            max_size=2**20,
        )

    async def _session(self, ws: ClientConnection) -> None:
        await ws.send(encode(MessageType.DEVICE_STATUS, self._hello()))
        self._ws = ws
        self._set_state(LinkState.ONLINE)
        self._online.set()
        heartbeat = asyncio.create_task(self._heartbeat(ws))
        try:
            async for raw in ws:
                await self._dispatch(raw)
        finally:
            heartbeat.cancel()

    def _hello(self) -> dict:
        hello = {
            "device_id": self.config.device_id,
            "protocol_version": PROTOCOL_VERSION,
            "software": f"haru-terminal/{__version__}",
            "platform": f"{platform.system()} {platform.machine()}",
            "capabilities": self.capabilities,
        }
        for provider in self._hello_fields:
            hello.update(provider())
        return hello

    async def _heartbeat(self, ws: ClientConnection) -> None:
        while True:
            await asyncio.sleep(self.config.heartbeat_interval_s)
            data = {
                "uptime_s": round(time.monotonic() - self.started_at, 1),
                "rtt_ms": round(ws.latency * 1000, 1),
            }
            for provider in self._heartbeat_fields:
                try:
                    data.update(provider())
                except Exception:
                    log.exception("heartbeat field provider failed")
            try:
                await ws.send(encode(MessageType.DEVICE_HEARTBEAT, data))
            except ConnectionClosed:
                return

    async def _dispatch(self, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            for fh in self._frame_handlers:
                await _call(fh, raw)
            return
        try:
            msg = decode(raw)
        except ProtocolError as exc:
            log.warning("dropping malformed message from backend: %s", exc)
            return
        handlers = self._handlers.get(msg.type)
        if not handlers:
            log.debug("no handler for %s: %s", msg.type, msg.data)
            return
        for h in handlers:
            try:
                await _call(h, msg)
            except Exception:
                # A buggy handler must never take the link down.
                log.exception("handler for %s failed", msg.type)

    def _set_state(self, state: LinkState) -> None:
        if state == self.state:
            return
        log.info("link %s -> %s", self.state, state)
        self.state = state
        for listener in self._state_listeners:
            try:
                listener(state)
            except Exception:
                log.exception("state listener failed")


async def _call(fn, arg) -> None:
    result = fn(arg)
    if asyncio.iscoroutine(result):
        await result
