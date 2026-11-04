"""Stage 1 tests: protocol encoding plus a real gateway <-> terminal link over
localhost sockets (no mocks — same code paths as the Pi and the backend)."""

from __future__ import annotations

import asyncio
import json
import socket

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError, InvalidStatus

from haru_gateway.server import Gateway, GatewayConfig
from haru_protocol import (
    CloseCode,
    FrameKind,
    MessageType,
    ProtocolError,
    decode,
    decode_frame,
    encode,
    encode_frame,
)
from haru_terminal.config import TerminalConfig
from haru_terminal.transport import BackendLink, LinkState

TOKEN = "test-token"


# --- protocol ----------------------------------------------------------------

def test_envelope_roundtrip():
    msg = decode(encode(MessageType.BUTTON_PRESS, {"button": "main"}))
    assert msg.type == "button.press"
    assert msg.data == {"button": "main"}
    assert msg.version == 1 and msg.timestamp > 0


@pytest.mark.parametrize("raw, err", [
    ("not json", "not JSON"),
    ("[]", "object"),
    ('{"version": 2, "type": "x"}', "version"),
    ('{"version": 1}', "type"),
    ('{"version": 1, "type": "x", "data": []}', "data"),
])
def test_envelope_rejects(raw, err):
    with pytest.raises(ProtocolError, match=err):
        decode(raw)


def test_unknown_type_passes_decode():
    assert decode('{"version": 1, "type": "future.thing"}').type == "future.thing"


def test_frame_roundtrip():
    f = decode_frame(encode_frame(FrameKind.AUDIO_INPUT, 7, 42, 1_790_000_000_123_456, b"\x01\x02"))
    assert (f.kind, f.stream_id, f.seq, f.captured_at_us, f.payload) == \
        (FrameKind.AUDIO_INPUT, 7, 42, 1_790_000_000_123_456, b"\x01\x02")
    with pytest.raises(ProtocolError):
        decode_frame(b"\x01")
    with pytest.raises(ProtocolError):
        decode_frame(bytes([1, 99]) + bytes(14))


# --- live link -----------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class RunningGateway:
    def __init__(self, port: int):
        self.gateway = Gateway(GatewayConfig("127.0.0.1", port, TOKEN))
        self.stop: asyncio.Future | None = None
        self.task: asyncio.Task | None = None

    async def start(self):
        self.stop = asyncio.get_running_loop().create_future()
        self.task = asyncio.create_task(self.gateway.serve(self.stop))
        await asyncio.sleep(0.1)
        return self

    async def shutdown(self):
        self.stop.set_result(None)
        await self.task


def terminal_config(port: int, **kw) -> TerminalConfig:
    return TerminalConfig(backend_url=f"ws://127.0.0.1:{port}/terminal", token=TOKEN,
                          device_id="test-pi", heartbeat_interval_s=0.1,
                          backoff_initial_s=0.05, backoff_max_s=0.2, **kw)


async def until(pred, timeout=3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not pred():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)


def test_connect_hello_status_heartbeat_and_events():
    async def body():
        port = free_port()
        gw = await RunningGateway(port).start()
        link = BackendLink(terminal_config(port), capabilities=["audio.input"])
        statuses = []
        link.on(MessageType.BACKEND_STATUS, lambda m: statuses.append(m.data))
        runner = asyncio.create_task(link.run_forever())

        await link.wait_online(2)
        await until(lambda: statuses)
        assert statuses[0]["gateway"] == "ok"
        session = gw.gateway.sessions["test-pi"]
        assert session.info["capabilities"] == ["audio.input"]

        first_hb = session.last_heartbeat
        await until(lambda: session.last_heartbeat > first_hb)

        assert await link.send(MessageType.BUTTON_PRESS, {"button": "main"})

        runner.cancel()
        await until(lambda: "test-pi" not in gw.gateway.sessions)
        await gw.shutdown()

    asyncio.run(body())


def test_reconnects_after_backend_restart():
    async def body():
        port = free_port()
        gw = await RunningGateway(port).start()
        link = BackendLink(terminal_config(port))
        states = []
        link.on_state(states.append)
        runner = asyncio.create_task(link.run_forever())
        await link.wait_online(2)

        await gw.shutdown()
        await until(lambda: link.state != LinkState.ONLINE)
        assert not await link.send(MessageType.BUTTON_PRESS, {})  # offline: dropped, not raised
        await asyncio.sleep(0.5)  # a few failed attempts while it's down

        gw = await RunningGateway(port).start()
        await until(lambda: link.state == LinkState.ONLINE)
        assert "test-pi" in gw.gateway.sessions
        assert states.count(LinkState.ONLINE) == 2

        runner.cancel()
        await gw.shutdown()

    asyncio.run(body())


def test_rejects_bad_token_and_bad_path():
    async def body():
        port = free_port()
        gw = await RunningGateway(port).start()
        with pytest.raises(InvalidStatus) as exc:
            await connect(f"ws://127.0.0.1:{port}/terminal",
                          additional_headers={"Authorization": "Bearer wrong"})
        assert exc.value.response.status_code == 401
        with pytest.raises(InvalidStatus) as exc:
            await connect(f"ws://127.0.0.1:{port}/nope")
        assert exc.value.response.status_code == 404
        await gw.shutdown()

    asyncio.run(body())


def test_rejects_bad_hello_and_wrong_version():
    async def body():
        port = free_port()
        gw = await RunningGateway(port).start()
        url = f"ws://127.0.0.1:{port}/terminal"
        headers = {"Authorization": f"Bearer {TOKEN}"}

        async with connect(url, additional_headers=headers) as ws:
            await ws.send(encode(MessageType.BUTTON_PRESS, {}))
            with pytest.raises(ConnectionClosedError):
                await ws.recv()
            assert ws.close_code == CloseCode.BAD_HELLO

        async with connect(url, additional_headers=headers) as ws:
            await ws.send(json.dumps({"version": 99, "type": "device.status", "data": {}}))
            with pytest.raises(ConnectionClosedError):
                await ws.recv()
            assert ws.close_code == CloseCode.UNSUPPORTED_VERSION
        await gw.shutdown()

    asyncio.run(body())


def test_same_device_id_replaces_old_connection():
    async def body():
        port = free_port()
        gw = await RunningGateway(port).start()
        url = f"ws://127.0.0.1:{port}/terminal"
        headers = {"Authorization": f"Bearer {TOKEN}"}
        hello = encode(MessageType.DEVICE_STATUS, {"device_id": "dup"})

        old = await connect(url, additional_headers=headers)
        await old.send(hello)
        await old.recv()  # backend.status
        new = await connect(url, additional_headers=headers)
        await new.send(hello)
        await new.recv()

        with pytest.raises(ConnectionClosedError):
            await old.recv()
        assert old.close_code == CloseCode.REPLACED
        await until(lambda: gw.gateway.sessions.get("dup") and gw.gateway.sessions["dup"].ws is not None)
        await new.close()
        await until(lambda: "dup" not in gw.gateway.sessions)
        await gw.shutdown()

    asyncio.run(body())


def test_clean_close_says_going_away():
    async def body():
        port = free_port()
        gw = await RunningGateway(port).start()
        link = BackendLink(terminal_config(port))
        runner = asyncio.create_task(link.run_forever())
        await until(lambda: "test-pi" in gw.gateway.sessions)
        ws = gw.gateway.sessions["test-pi"].ws
        await link.close()
        await asyncio.wait_for(runner, 2)  # run_forever returns instead of reconnecting
        await until(lambda: "test-pi" not in gw.gateway.sessions)
        assert ws.close_code == 1001
        await gw.shutdown()

    asyncio.run(body())


def test_llm_probe_and_status_broadcast():
    """Probe against a fake OpenAI-compatible /models endpoint; status changes
    are pushed to connected terminals as backend.status."""
    import http.server
    import threading

    from haru_gateway import llm

    class Models(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer k":
                self.send_response(401); self.end_headers(); return
            body = json.dumps({"data": [{"id": "gpt-oss-120b"}]}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Models)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    cfg = lambda **kw: llm.LLMConfig(**{"base_url": base, "model": "gpt-oss-120b", "api_key": "k",
                                         "reasoning_effort": "", "health_interval_s": 0.05, **kw})
    assert llm.probe(cfg()) == "ok"
    assert llm.probe(cfg(api_key="bad")) == "unauthorized"
    assert llm.probe(cfg(api_key="")) == "no_api_key"
    assert llm.probe(cfg(model="other")) == "model_missing"
    assert llm.probe(cfg(base_url="http://127.0.0.1:1")) == "unavailable"

    async def body():
        port = free_port()
        gw = RunningGateway(port)
        gw.gateway.config.llm = cfg()
        await gw.start()
        link = BackendLink(terminal_config(port))
        seen = []
        link.on(MessageType.BACKEND_STATUS, lambda m: seen.append(m.data["llm"]))
        runner = asyncio.create_task(link.run_forever())
        await until(lambda: "ok" in seen)
        srv.shutdown(); srv.server_close()  # LLM server disappears...
        await until(lambda: seen[-1] == "unavailable")
        assert link.state == LinkState.ONLINE  # ...and the Pi link doesn't care
        await link.close(); runner.cancel()
        await gw.shutdown()

    asyncio.run(body())
