"""Haru terminal gateway: the only thing the Pi ever talks to.

Stage 1 scope: authenticate terminals, validate the hello, track sessions,
answer with backend.status, log everything. Later stages hang STT/LLM/TTS off
the dispatch table in `TerminalSession.handle` without the Pi changing.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from http import HTTPStatus

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from haru_protocol import (
    DISPLAY_PATH,
    PROTOCOL_VERSION,
    TERMINAL_PATH,
    TERMINAL_TO_BACKEND,
    CloseCode,
    MessageType,
    ProtocolError,
    decode,
    decode_frame,
    encode,
)

from urllib.parse import parse_qs, urlsplit

from pathlib import Path
from zoneinfo import ZoneInfo

from haru_gateway import audio, chatlog, conversation, display, llm, memory, relevan, stt, tts
from haru_gateway.tools import ToolRegistry
from haru_gateway.tools.alexandria import alexandria_tools
from haru_gateway.tools.calendar import calendar_tools
from haru_gateway.tools.capy import CapyClient, CapyConfig, CapyService
from haru_gateway.tools.mcp import MCPServer, github_server_config
from haru_gateway.tools.notifications import NotificationService
from haru_gateway.tools.timers import TimerService
from haru_gateway.tools.weather import weather_tools
from haru_gateway.tools.web import web_tools

log = logging.getLogger("haru.gateway")

HELLO_TIMEOUT_S = 10.0


@dataclass
class GatewayConfig:
    host: str
    port: int
    terminal_token: str
    llm: llm.LLMConfig | None = None
    stt: stt.STTConfig | None = None
    tts: tts.TTSConfig | None = None
    # debug.ask / debug.say from any authenticated connection. Dev only.
    debug_commands: bool = False
    # Shared by all terminals; sinks key their state by stream label.
    audio_sinks: list[audio.AudioSink] = field(default_factory=lambda: [audio.StatsSink()])

    @classmethod
    def from_env(cls) -> GatewayConfig:
        token = os.environ.get("HARU_TERMINAL_TOKEN", "")
        if not token:
            raise SystemExit("HARU_TERMINAL_TOKEN is not set; refusing to start unauthenticated")
        return cls(
            host=os.environ.get("HARU_GATEWAY_HOST", "0.0.0.0"),
            port=int(os.environ.get("HARU_GATEWAY_PORT", "8765")),
            terminal_token=token,
            llm=llm.LLMConfig.from_env(),
            stt=stt.STTConfig.from_env(),
            tts=tts.TTSConfig.from_env(),
            debug_commands=os.environ.get("HARU_DEBUG_COMMANDS", "") == "1",
            audio_sinks=audio.sinks_from_env(),
        )


@dataclass
class TerminalSession:
    device_id: str
    ws: ServerConnection
    info: dict
    audio: audio.AudioRouter
    gateway: Gateway
    connected_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)

    async def send(self, type_: MessageType, data: dict | None = None) -> None:
        await self.ws.send(encode(type_, data))

    async def handle(self, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            self.audio.handle_frame(decode_frame(raw))
            return
        msg = decode(raw)
        if msg.type == MessageType.DEVICE_HEARTBEAT:
            self.last_heartbeat = time.time()
            log.debug("heartbeat from %s: %s", self.device_id, msg.data)
        elif msg.type == MessageType.DEVICE_STATUS:
            if "playback" in msg.data:
                if self.gateway.tts:
                    self.gateway.tts.on_playback(self.device_id, msg.data["playback"])
                return
            self.info.update(msg.data)
            log.info("status from %s: %s", self.device_id, msg.data)
        elif msg.type == MessageType.AUDIO_INPUT:
            self.audio.handle_control(msg.data)
        elif msg.type in ("debug.ask", "debug.say", "debug.gesture", "debug.notify", "debug.notifications"):
            await self.gateway.handle_debug(self.device_id, msg.type, msg.data)
        elif msg.type in TERMINAL_TO_BACKEND:
            # Known but not wired to anything yet (buttons, camera).
            log.info("event from %s: %s %s", self.device_id, msg.type, msg.data)
        else:
            log.warning("ignoring unknown message type %r from %s", msg.type, self.device_id)


class Gateway:
    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self.sessions: dict[str, TerminalSession] = {}
        self.displays = display.DisplayHub()
        self.started_at = time.time()
        self.llm_status = "not_configured" if config.llm is None else "unknown"
        self.stt: stt.SpeechToText | None = None
        self.audio_sinks = list(config.audio_sinks)
        if config.stt is not None:
            self.stt = stt.SpeechToText(config.stt, on_status=lambda _: self.broadcast_status())
            self.audio_sinks.append(self.stt)
        self.tts: tts.TextToSpeech | None = None
        if config.tts is not None:
            self.tts = tts.TextToSpeech(config.tts, _TerminalAccess(self),
                                        on_status=lambda _: self.broadcast_status())
        self.conversations: conversation.ConversationManager | None = None
        self.relevan: relevan.Relevan | None = None
        self.memory: memory.MemoryStore | None = None
        if (rel_cfg := relevan.RelevanConfig.from_env()) is not None:
            self.relevan = relevan.Relevan(rel_cfg, timeout=5.0)
            self.memory = memory.MemoryStore(self.relevan)
        self.chatlog = chatlog.ChatLog.from_env()
        conv_config = conversation.ConversationConfig.from_env()
        self.timers = TimerService(Path(os.environ.get("HARU_TIMERS_FILE", "/data/state/timers.json")),
                                   conv_config.tz, self._announce)
        self.notifications = NotificationService(
            Path(os.environ.get("HARU_NOTIFICATIONS_FILE", "/data/state/notifications.json")), conv_config.tz)
        self.notifications.on_badge(self._push_badge)
        self.notifications.on_panel(self.displays.panel)
        self.displays.on_attach.append(self._push_badge)
        self.tools = ToolRegistry()
        self.tools.add(*web_tools(), *weather_tools(), *self.timers.tools(), *calendar_tools(conv_config.tz),
                       *alexandria_tools(), *self.notifications.tools())
        self.capy: CapyService | None = None
        if (capy_cfg := CapyConfig.from_env()) is not None:
            self.capy = CapyService(capy_cfg, CapyClient(capy_cfg), self.notifications, conv_config.tz)
            self.tools.add(*self.capy.tools())
        log.info("tools: %s", ", ".join(self.tools.tools) or "none")
        # MCP servers register their tools once connected (see tools/mcp.py).
        self.mcp_servers = [MCPServer(c, self.tools) for c in (github_server_config(),) if c is not None]
        if config.stt is not None and config.llm is not None:
            self.conversations = conversation.ConversationManager(
                config.llm, conv_config, on_llm_status=self._on_llm_status,
                memory=self.memory, chatlog=self.chatlog, tools=self.tools)
            self.stt.on_transcript(self.conversations.handle_transcript)
            self.conversations.on_emotion(lambda dev, emotion: self.displays.set_state(dev, emotion=emotion))
            self.conversations.on_gesture(self.displays.gesture)
            if self.tts is not None:
                c, t = self.conversations, self.tts
                c.on_sentence(lambda dev, sentence, started: t.say(dev, sentence, started))
                c.on_reply(lambda reply: t.end(reply.device_id))
                c.on_interrupt(t.cancel)
                c.set_echo_filter(t.is_echo)
        self._wire_avatar()

    def backend_status(self) -> dict:
        return {
            "gateway": "ok",
            "llm": self.llm_status,
            "stt": self.stt.status if self.stt else "not_configured",
            "tts": self.tts.status if self.tts else "not_configured",
            "uptime_s": round(time.time() - self.started_at, 1),
        }

    def _wire_avatar(self) -> None:
        """Drive each terminal's avatar from the pipeline:
        speech start -> listening, utterance end -> thinking, playback -> speaking -> idle."""
        d = self.displays
        if self.stt is not None:
            def on_vad(label: str, event: str) -> None:
                device_id = label.split("/", 1)[0]
                speaking = d.state[device_id]["state"] == "speaking"
                if event == "speech_start" and self.conversations is not None:
                    self.conversations.start_session(device_id)  # warm the memory cache before STT is done
                if event == "speech_start" and not speaking:
                    asyncio.create_task(d.set_state(device_id, "listening"))
                elif event == "speech_end" and not speaking:
                    asyncio.create_task(d.set_state(device_id, "thinking"))
            self.stt.on_vad(on_vad)
        if self.tts is not None:
            async def on_audio(device_id: str, stream_id: int, pcm: bytes, rate: int, channels: int) -> None:
                await d.mouth(device_id, stream_id, display.mouth_envelope(pcm, rate, channels))

            async def on_playback(device_id: str, event: str, stream_id: int) -> None:
                if event == "started":
                    await d.set_state(device_id, "speaking", stream_id=stream_id)
                else:
                    await d.set_state(device_id, "idle")
            self.tts.on_audio(on_audio)
            self.tts.on_playback_event(on_playback)

    async def _push_badge(self, device_id: str | None) -> None:
        """Send the unread count to that device's displays (device_id None: every connected display)."""
        targets = [device_id] if device_id else [d for d, ws in self.displays.displays.items() if ws]
        for dev in targets:
            await self.displays.badge(dev, self.notifications.unread_count(dev))

    async def _announce(self, device_id: str, text: str) -> None:
        """Something Haru says unprompted (a timer or reminder going off)."""
        if self.tts is None:
            log.warning("can't announce %r on %s: no TTS", text, device_id)
            return
        await self.displays.set_state(device_id, emotion="surprised")
        self.tts.say(device_id, text, time.monotonic())
        self.tts.end(device_id)
        if self.chatlog and self.conversations and (conv := self.conversations.conversations.get(device_id)):
            self.chatlog.append(device_id, conv.session, "assistant", text, kind="announcement")

    def _check_token(self, presented: str) -> bool:
        return hmac.compare_digest(presented.encode(), self.config.terminal_token.encode())

    def process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        url = urlsplit(request.path)
        if url.path == DISPLAY_PATH:
            q = parse_qs(url.query)
            if not q.get("device") or not self._check_token(q.get("token", [""])[0]):
                log.warning("rejected display from %s", connection.remote_address)
                return _respond(connection, HTTPStatus.UNAUTHORIZED, "unauthorized\n")
            return None
        if request.path == "/healthz":
            body = json.dumps({"ok": True, "terminals": len(self.sessions)}) + "\n"
            return _respond(connection, HTTPStatus.OK, body, "application/json")
        if request.path != TERMINAL_PATH:
            return _respond(connection, HTTPStatus.NOT_FOUND, "not found\n")
        auth = request.headers.get("Authorization", "")
        expected = f"Bearer {self.config.terminal_token}"
        if not hmac.compare_digest(auth.encode(), expected.encode()):
            log.warning("rejected unauthenticated terminal from %s", connection.remote_address)
            return _respond(connection, HTTPStatus.UNAUTHORIZED, "unauthorized\n")
        return None

    async def handle_connection(self, ws: ServerConnection) -> None:
        url = urlsplit(ws.request.path)
        if url.path == DISPLAY_PATH:
            await self.displays.attach(parse_qs(url.query)["device"][0], ws)
            return
        peer = ws.remote_address
        try:
            raw = await asyncio.wait_for(ws.recv(), HELLO_TIMEOUT_S)
            hello = decode(raw)
        except ProtocolError as exc:
            code = CloseCode.UNSUPPORTED_VERSION if "version" in str(exc) else CloseCode.BAD_HELLO
            log.warning("bad hello from %s: %s", peer, exc)
            await ws.close(code, str(exc)[:120])
            return
        except (TimeoutError, ConnectionClosed):
            log.warning("no hello from %s", peer)
            return
        device_id = hello.data.get("device_id")
        if hello.type != MessageType.DEVICE_STATUS or not isinstance(device_id, str) or not device_id:
            log.warning("bad hello from %s: %s", peer, hello)
            await ws.close(CloseCode.BAD_HELLO, "first message must be device.status with device_id")
            return

        if (old := self.sessions.get(device_id)) is not None:
            log.info("terminal %s reconnected; closing previous connection", device_id)
            await old.ws.close(CloseCode.REPLACED, "replaced by newer connection")

        session = TerminalSession(device_id=device_id, ws=ws, info=dict(hello.data),
                                  audio=audio.AudioRouter(device_id, self.audio_sinks), gateway=self)
        self.sessions[device_id] = session
        log.info("terminal %s connected from %s: %s", device_id, peer, hello.data)
        try:
            await session.send(MessageType.BACKEND_STATUS, self.backend_status())
            async for raw in ws:
                try:
                    await session.handle(raw)
                except ProtocolError as exc:
                    log.warning("dropping malformed message from %s: %s", device_id, exc)
        except ConnectionClosed:
            pass
        finally:
            session.audio.close("disconnected")
            if self.sessions.get(device_id) is session:
                del self.sessions[device_id]
            log.info("terminal %s disconnected (code=%s, %.0fs session)",
                     device_id, ws.close_code, time.time() - session.connected_at)

    async def _on_llm_status(self, status: str) -> None:
        # Reported by both the health probe and every conversation turn; only
        # a change is worth telling the terminals about.
        if status == self.llm_status:
            return
        self.llm_status = status
        await self.broadcast_status()

    async def handle_debug(self, from_device: str, type_: str, data: dict) -> None:
        if not self.config.debug_commands:
            log.warning("ignoring %s from %s: HARU_DEBUG_COMMANDS is off", type_, from_device)
            return
        target, text = data.get("device_id", from_device), str(data.get("text", "")).strip()
        if not text:
            return
        log.info("%s from %s -> %s: %r", type_, from_device, target, text)
        if type_ == "debug.say" and self.tts:
            await self.tts.cancel(target)
            self.tts.say(target, text, time.monotonic())
            self.tts.end(target)
        elif type_ == "debug.gesture":
            await self.displays.gesture(target, text)
        elif type_ == "debug.notify":
            # A test notification, as if a task had finished: {"device_id"?, "text": title, "body"?}
            await self.notifications.add(text, str(data.get("body", "")), source="debug",
                                         device_id=data.get("device_id"))
        elif type_ == "debug.notifications":
            # Drive the notification UI without Haru speaking: "open", "open all" or "clear".
            if text.startswith("open"):
                await self.notifications.open(target, include_read=text == "open all")
            elif text == "clear":
                await self.notifications.clear(target, [n.id for n in self.notifications.for_device(target)])
        elif type_ == "debug.ask" and self.conversations:
            # Exactly as if `target`'s mic had heard it.
            self.conversations.handle_transcript(stt.Transcript(f"{target}/debug", text, 0, 0.0, 0.0, 0.0))

    async def broadcast_status(self) -> None:
        status = self.backend_status()
        for session in list(self.sessions.values()):
            try:
                await session.send(MessageType.BACKEND_STATUS, status)
            except ConnectionClosed:
                pass

    async def serve(self, stop: asyncio.Future | None = None) -> None:
        background = []
        self.timers.start()
        background += [asyncio.create_task(s.run_forever()) for s in self.mcp_servers]
        if self.capy is not None:
            background.append(asyncio.create_task(self.capy.run_forever()))
        if self.config.llm is not None:
            background.append(asyncio.create_task(llm.watch(self.config.llm, self._on_llm_status)))
        if self.stt is not None:
            background.append(asyncio.create_task(self.stt.run()))
        if self.tts is not None:
            background.append(asyncio.create_task(self.tts.run()))
        try:
            await self._serve(stop)
        finally:
            for task in background:
                task.cancel()
            if self.stt is not None:
                await self.stt.close()
            if self.conversations is not None:
                await self.conversations.close()
            if self.tts is not None:
                await self.tts.close()
            self.timers.close()
            if self.capy is not None:
                await self.capy.client.close()
            if self.memory is not None:
                await self.memory.close()
            if self.relevan is not None:
                await self.relevan.close()

    async def _serve(self, stop: asyncio.Future | None) -> None:
        async with serve(
            self.handle_connection,
            self.config.host,
            self.config.port,
            process_request=self.process_request,
            ping_interval=10,
            ping_timeout=10,
            max_size=2**20,
        ) as server:
            self.server: Server = server
            log.info("gateway listening on ws://%s:%d%s (protocol v%d)",
                     self.config.host, self.config.port, TERMINAL_PATH, PROTOCOL_VERSION)
            if stop is None:
                await server.serve_forever()
            else:
                await stop


class _TerminalAccess(tts.Terminals):
    """How the TTS reaches terminals, without depending on the gateway's internals."""

    def __init__(self, gateway: Gateway) -> None:
        self.gateway = gateway

    def output_format(self, device_id: str) -> dict | None:
        session = self.gateway.sessions.get(device_id)
        if session is None or "audio.output" not in session.info.get("capabilities", []):
            return None
        return session.info.get("audio_output")

    async def send(self, device_id: str, type_: MessageType, data: dict) -> bool:
        session = self.gateway.sessions.get(device_id)
        if session is None:
            return False
        try:
            await session.send(type_, data)
            return True
        except ConnectionClosed:
            return False

    async def send_frame(self, device_id: str, frame: bytes) -> bool:
        session = self.gateway.sessions.get(device_id)
        if session is None:
            return False
        try:
            await session.ws.send(frame)
            return True
        except ConnectionClosed:
            return False


def _respond(connection: ServerConnection, status: HTTPStatus, body: str,
             content_type: str = "text/plain") -> Response:
    response = connection.respond(status, body)
    del response.headers["Content-Type"]
    response.headers["Content-Type"] = content_type
    return response


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("HARU_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # websockets logs every non-upgrade request (e.g. /healthz) as an error otherwise.
    logging.getLogger("websockets.server").setLevel(logging.WARNING)
    config = GatewayConfig.from_env()

    async def run() -> None:
        loop = asyncio.get_running_loop()
        stop = loop.create_future()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: stop.done() or stop.set_result(None))
        await Gateway(config).serve(stop)

    asyncio.run(run())


if __name__ == "__main__":
    main()
