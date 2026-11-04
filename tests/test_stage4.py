"""Stage 4 tests: the conversation loop against a fake streaming
OpenAI-compatible server (SSE, gpt-oss style, with reasoning deltas)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from haru_gateway.conversation import ConversationConfig, ConversationManager
from haru_gateway.llm import LLMConfig
from haru_gateway.stt import Transcript


class FakeLLM(BaseHTTPRequestHandler):
    requests: list[dict] = []
    fail_next = False
    delay_s = 0.0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLLM.requests.append(body)
        if FakeLLM.fail_next:
            FakeLLM.fail_next = False
            self.send_response(502); self.end_headers(); self.wfile.write(b"bad gateway"); return
        assert self.headers["User-Agent"].startswith("haru-gateway/")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        user = body["messages"][-1]["content"]

        def event(delta):
            chunk = {"choices": [{"index": 0, "delta": delta}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        event({"role": "assistant", "content": ""})
        event({"reasoning": "The user said something.", "reasoning_content": "The user said something."})
        for piece in ["You said ", f"{user}. ", "That's ", "interesting!", " Anything", " else?"]:
            time.sleep(FakeLLM.delay_s)
            event({"content": piece})
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


@pytest.fixture
def llm_server():
    FakeLLM.requests, FakeLLM.fail_next, FakeLLM.delay_s = [], False, 0.0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def manager(base, **conv_kw):
    llm = LLMConfig(base_url=base, model="gpt-oss-120b", api_key="k", reasoning_effort="low",
                    health_interval_s=30)
    return ConversationManager(llm, ConversationConfig(persona="You are a test.", **conv_kw))


def heard(text, device="pi"):
    return Transcript(f"{device}/1", text, 1000, 300.0, 300.0, 300.0)


async def wait_for(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


def test_streams_sentences_and_keeps_history(llm_server):
    async def body():
        m = manager(llm_server)
        sentences, replies = [], []
        m.on_sentence(lambda dev, s, started: sentences.append((dev, s, len(replies))))
        m.on_reply(replies.append)

        m.handle_transcript(heard("hello"))
        await wait_for(lambda: len(replies) == 1)
        # Sentences arrive individually, and before the reply completes.
        assert [s for _, s, _ in sentences] == ["You said hello.", "That's interesting!", "Anything else?"]
        assert all(n == 0 for _, _, n in sentences)
        r = replies[0]
        assert r.text == "You said hello. That's interesting! Anything else?"
        assert "reasoning" not in r.text.lower() and "user said" not in r.text
        assert 0 < r.first_token_ms <= r.first_sentence_ms <= r.total_ms

        m.handle_transcript(heard("second"))
        await wait_for(lambda: len(replies) == 2)
        msgs = FakeLLM.requests[-1]["messages"]
        assert [x["role"] for x in msgs] == ["system", "user", "assistant", "user"]
        assert msgs[1]["content"] == "hello" and msgs[-1]["content"] == "second"
        assert FakeLLM.requests[-1]["reasoning_effort"] == "low" and FakeLLM.requests[-1]["stream"]

        # Separate terminals get separate threads.
        m.handle_transcript(heard("other device", device="kitchen"))
        await wait_for(lambda: len(replies) == 3)
        assert [x["role"] for x in FakeLLM.requests[-1]["messages"]] == ["system", "user"]
        await m.close()

    asyncio.run(body())


def test_speech_during_reply_merges_into_one_turn(llm_server):
    FakeLLM.delay_s = 0.1  # slow stream, so the second utterance lands mid-reply

    async def body():
        m = manager(llm_server)
        replies = []
        m.on_reply(replies.append)
        m.handle_transcript(heard("what's the weather"))
        await asyncio.sleep(0.15)
        m.handle_transcript(heard("in London"))
        await wait_for(lambda: replies)
        await asyncio.sleep(0.3)
        assert len(replies) == 1
        assert replies[0].user_text == "what's the weather in London"
        assert replies[0].merged_utterances == 2
        conv = m.conversations["pi"]
        assert [x["content"] for x in conv.history if x["role"] == "user"] == ["what's the weather in London"]
        await m.close()

    asyncio.run(body())


def test_llm_failure_drops_turn_and_recovers(llm_server):
    async def body():
        statuses = []

        async def on_status(s):
            statuses.append(s)

        m = manager(llm_server)
        m._on_llm_status = on_status
        replies = []
        m.on_reply(replies.append)
        FakeLLM.fail_next = True
        m.handle_transcript(heard("this one fails"))
        await wait_for(lambda: statuses == ["unavailable"])
        assert m.conversations["pi"].history == [] and m.conversations["pi"].pending == []

        m.handle_transcript(heard("this one works"))
        await wait_for(lambda: replies)
        assert statuses[-1] == "ok"
        assert [x["content"] for x in m.conversations["pi"].history if x["role"] == "user"] == ["this one works"]
        await m.close()

    asyncio.run(body())


def test_idle_reset_and_window(llm_server):
    async def body():
        m = manager(llm_server, max_messages=4, idle_reset_s=3600)
        replies = []
        m.on_reply(replies.append)
        for i in range(3):
            m.handle_transcript(heard(f"turn {i}"))
            await wait_for(lambda: len(replies) == i + 1)
        conv = m.conversations["pi"]
        assert len(conv.history) == 4 and conv.history[0]["content"] == "turn 1"

        conv.last_activity -= 7200  # two hours idle
        m.handle_transcript(heard("fresh start"))
        await wait_for(lambda: len(replies) == 4)
        assert [x["role"] for x in FakeLLM.requests[-1]["messages"]] == ["system", "user"]
        await m.close()

    asyncio.run(body())


def test_unreachable_llm_is_reported_not_raised():
    async def body():
        statuses = []

        async def on_status(s):
            statuses.append(s)

        m = manager("http://127.0.0.1:1")
        m._on_llm_status = on_status
        m.handle_transcript(heard("anyone there?"))
        await wait_for(lambda: statuses == ["unavailable"])
        await m.close()

    asyncio.run(body())


def test_gateway_broadcasts_llm_status_only_on_change():
    from haru_gateway.server import Gateway, GatewayConfig

    async def body():
        gw = Gateway(GatewayConfig("127.0.0.1", 0, "t"))
        sent = []

        async def fake_broadcast():
            sent.append(gw.llm_status)

        gw.broadcast_status = fake_broadcast
        for s in ["ok", "ok", "unavailable", "unavailable", "ok"]:
            await gw._on_llm_status(s)
        assert sent == ["ok", "unavailable", "ok"]

    asyncio.run(body())
