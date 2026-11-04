"""Long-term memory tests: session recall (async, cached, budgeted), the chat log,
and the memory writer, all against a fake Relevan API and a fake LLM."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from haru_gateway.chatlog import ChatLog
from haru_gateway.conversation import ConversationConfig, ConversationManager
from haru_gateway.llm import LLMConfig
from haru_gateway.memory import MemoryStore, format_for_prompt
from haru_gateway.memory_writer import MemoryWriter, WriterConfig
from haru_gateway.relevan import Relevan, RelevanConfig
from haru_gateway.stt import Transcript

BIRTHDAY = {"id": "mem-bday", "text": "The user's wife's birthday is June 3.", "subject": "wife birthday",
            "kind": "event", "tags": ["wife", "birthday"], "importance": 5, "status": "active", "use_count": 0,
            "person": "household", "created_at": "2026-09-01T00:00:00Z"}
COFFEE = {"id": "mem-coffee", "text": "The user drinks oat milk lattes.", "subject": "coffee",
          "kind": "preference", "tags": ["coffee"], "importance": 3, "status": "active", "use_count": 0}
OLD = {"id": "mem-old", "text": "The user lives in Austin.", "subject": "home", "kind": "fact",
       "importance": 4, "status": "active", "use_count": 0}


class FakeRelevan(BaseHTTPRequestHandler):
    docs: dict[str, dict] = {}
    calls: list[tuple] = []
    search_delay = 0.0

    def _reply(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n)) if n else {}

    def do_GET(self):
        assert self.headers["x-api-key"] == "k"
        url = urlsplit(self.path)
        if url.path.endswith("/documents"):
            page = int(parse_qs(url.query).get("page", ["1"])[0])
            docs = list(FakeRelevan.docs.values())[(page - 1) * 100: page * 100]
            FakeRelevan.calls.append(("list", page))
            return self._reply({"documents": [{"id": d["id"], "content": {k: v for k, v in d.items() if k != "id"}}
                                              for d in docs]})
        self._reply({}, 404)

    def do_POST(self):
        body = self._body()
        if self.path.endswith("/search"):
            time.sleep(FakeRelevan.search_delay)
            q = set(body["query"].lower().replace("'", " ").split())
            hits = [d for d in FakeRelevan.docs.values()
                    if d.get("status") == "active" and q & set((d["text"] + " " + d["subject"]).lower().replace("'", " ").replace(".", "").split())]
            FakeRelevan.calls.append(("search", body["query"], body.get("sessionId"), body.get("customerId")))
            return self._reply({"results": [{"id": d["id"], "score": 1.0, "document": {k: v for k, v in d.items() if k != "id"},
                                             "explanation": "", "matchedFields": [], "matchedTerms": []} for d in hits],
                                "searchContext": {"queryId": f"q-{len(FakeRelevan.calls)}", "sessionId": "rs-1"}})
        if self.path.endswith("/_bulk"):
            for d in body["documents"]:
                FakeRelevan.docs[d["id"]] = {"id": d["id"], **d["content"]}
            FakeRelevan.calls.append(("bulk", [d["id"] for d in body["documents"]]))
            return self._reply({"accepted": len(body["documents"]), "documentIds": [d["id"] for d in body["documents"]]})
        if self.path.endswith("/feedback"):
            FakeRelevan.calls.append(("feedback", body))
            return self._reply({})
        self._reply({}, 404)

    def do_PUT(self):
        doc_id = self.path.rsplit("/", 1)[-1]
        FakeRelevan.docs[doc_id] = {"id": doc_id, **self._body()["content"]}
        FakeRelevan.calls.append(("put", doc_id))
        self._reply({})

    def log_message(self, *a):
        pass


@pytest.fixture
def relevan_server():
    FakeRelevan.docs = {d["id"]: dict(d) for d in (BIRTHDAY, COFFEE, OLD)}
    FakeRelevan.calls, FakeRelevan.search_delay = [], 0.0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeRelevan)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield RelevanConfig(api_key="k", org="org", index="haru-memories", base_url=f"http://127.0.0.1:{srv.server_port}")
    srv.shutdown()


async def wait_for(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


# --- read side ------------------------------------------------------------------------

def test_recall_uses_relevan_within_budget_and_caches(relevan_server):
    async def body():
        store = MemoryStore(Relevan(relevan_server), budget_s=0.5)
        store.start_session("pi")
        await wait_for(lambda: len(store.sessions["pi"].cache) == 3)  # preloaded in the background
        r = await store.recall("pi", "what should I get my wife for her birthday")
        assert r.from_search and r.ids[0] == "mem-bday" and r.query_id
        assert r.relevan_session == "rs-1"
        await store.recall("pi", "coffee order")
        assert FakeRelevan.calls[-1][2] == "rs-1"          # sessionId carried across turns
        assert FakeRelevan.calls[-1][3] == "household"     # stable customerId
        await store.close()

    asyncio.run(body())


def test_slow_relevan_never_delays_the_reply(relevan_server):
    FakeRelevan.search_delay = 1.0

    async def body():
        store = MemoryStore(Relevan(relevan_server), budget_s=0.2)
        store.start_session("pi")
        await wait_for(lambda: store.sessions["pi"].cache)
        t0 = time.monotonic()
        r = await store.recall("pi", "my wife's birthday")
        assert time.monotonic() - t0 < 0.4
        assert not r.from_search and "mem-bday" in r.ids     # local match over the session cache
        await store.close()

    asyncio.run(body())


def test_format_for_prompt():
    text = format_for_prompt([BIRTHDAY])
    assert "wife's birthday is June 3" in text and "noted 2026-09-01" in text
    assert format_for_prompt([]) == ""


class PromptCapture(BaseHTTPRequestHandler):
    bodies: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        PromptCapture.bodies.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ['<|ACT {"emotion":"happy"}|>', "Maybe some flowers. ", "She likes tulips!"]:
            self.wfile.write(f'data: {json.dumps({"choices": [{"delta": {"content": piece}}]})}\n\n'.encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


def test_conversation_injects_memories_and_logs_turns(relevan_server, tmp_path):
    PromptCapture.bodies = []
    llm_srv = ThreadingHTTPServer(("127.0.0.1", 0), PromptCapture)
    threading.Thread(target=llm_srv.serve_forever, daemon=True).start()

    async def body():
        store = MemoryStore(Relevan(relevan_server), budget_s=0.5)
        m = ConversationManager(
            LLMConfig(base_url=f"http://127.0.0.1:{llm_srv.server_port}", model="m", api_key="k",
                      reasoning_effort="", health_interval_s=30),
            ConversationConfig(persona="You are Haru."), memory=store, chatlog=ChatLog(tmp_path))
        replies = []
        m.on_reply(replies.append)
        m.start_session("pi")
        m.handle_transcript(Transcript("pi/1", "what should I get my wife for her birthday", 0, 0, 0, 0))
        await wait_for(lambda: replies)
        system = PromptCapture.bodies[0]["messages"][0]["content"]
        assert "You are Haru." in system and "June 3" in system and "It is currently" in system
        lines = [json.loads(line) for line in next(tmp_path.glob("pi/*.jsonl")).read_text().splitlines()]
        assert [e["role"] for e in lines] == ["user", "assistant"]
        assert lines[0]["memories"][0] == "mem-bday" and lines[0]["relevan"]["queryId"]
        assert lines[1]["text"] == "Maybe some flowers. She likes tulips!" and lines[1]["emotion"] == "happy"
        assert lines[0]["session"] == lines[1]["session"]
        await m.close()
        await store.close()

    asyncio.run(body())
    llm_srv.shutdown()


# --- write side -------------------------------------------------------------------------

class FakeExtractor(BaseHTTPRequestHandler):
    requests: list[dict] = []
    reply = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeExtractor.requests.append(body)
        out = json.dumps({"choices": [{"message": {"content": "```json\n" + json.dumps(FakeExtractor.reply) + "\n```"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def _write_log(tmp_path, entries):
    path = tmp_path / "pi" / f"{dt.date.today()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))


def test_writer_extracts_updates_retires_and_gives_feedback(relevan_server, tmp_path):
    FakeExtractor.requests = []
    FakeExtractor.reply = {
        "add": [{"text": "The user has a dog named Biscuit.", "subject": "dog", "kind": "person",
                 "tags": ["Dog", "pet"], "importance": 4}, {"text": ""}],   # the empty one is dropped
        "update": [{"id": "mem-coffee", "text": "The user drinks oat milk flat whites now."}],
        "retire": [{"id": "mem-old", "reason": "moved to Denver"}, {"id": "mem-made-up"}],
        "relevant": ["mem-bday", "mem-nonexistent"],
    }
    llm_srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeExtractor)
    threading.Thread(target=llm_srv.serve_forever, daemon=True).start()
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    _write_log(tmp_path, [
        {"ts": old, "session": "s1", "role": "user", "text": "My wife's birthday is coming up, and we moved to Denver.",
         "memories": ["mem-bday", "mem-coffee"], "relevan": {"queryId": "q-9", "sessionId": "rs-1"}},
        {"ts": old, "session": "s1", "role": "assistant", "text": "June 3rd, right?"},
        {"ts": old, "session": "s1", "role": "user", "text": "Yes. Also our dog Biscuit says hi."},
    ])

    async def body():
        w = MemoryWriter(WriterConfig(chatlog_dir=tmp_path, state_path=tmp_path / "state.json", idle_s=60),
                         Relevan(relevan_server),
                         LLMConfig(base_url=f"http://127.0.0.1:{llm_srv.server_port}", model="m", api_key="k",
                                   reasoning_effort="", health_interval_s=30))
        assert await w.run_once() == 1
        prompt = FakeExtractor.requests[0]["messages"][1]["content"]
        assert "mem-bday" in prompt and "Biscuit" in prompt and FakeExtractor.requests[0]["reasoning_effort"] == "medium"
        new = [d for d in FakeRelevan.docs.values() if "Biscuit" in d["text"]]
        assert len(new) == 1 and new[0]["tags"] == ["dog", "pet"] and new[0]["status"] == "active"
        assert new[0]["source"] == "s1" and new[0]["person"] == "household" and new[0]["subject"] == "dog"
        assert FakeRelevan.docs["mem-coffee"]["text"] == "The user drinks oat milk flat whites now."
        assert FakeRelevan.docs["mem-old"]["status"] == "retired"
        fb = [c[1] for c in FakeRelevan.calls if c[0] == "feedback"]
        assert fb == [{"actionName": "llm_judgment", "customerId": "household", "objectId": "mem-bday",
                       "queryId": "q-9", "sessionId": "rs-1", "position": 0}]
        assert FakeRelevan.docs["mem-bday"]["use_count"] == 1 and FakeRelevan.docs["mem-bday"]["last_used_at"]
        # Checkpointed: nothing new, nothing reprocessed (even by a fresh writer).
        assert await w.run_once() == 0
        w2 = MemoryWriter(w.config, Relevan(relevan_server), w.llm)
        assert await w2.run_once() == 0 and len(FakeExtractor.requests) == 1
        await w.close()
        await w2.close()

    asyncio.run(body())
    llm_srv.shutdown()


def test_useless_subject_falls_back_to_tags():
    from haru_gateway.memory_writer import _clean_memory
    m = _clean_memory({"text": "The user's favorite snack is salted pretzels.", "subject": "user",
                       "tags": ["food", "snack"]})
    assert m["subject"] == "food snack"
    assert _clean_memory({"text": "x", "subject": "favorite snack"})["subject"] == "favorite snack"


def test_writer_waits_for_quiet_session(relevan_server, tmp_path):
    now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    _write_log(tmp_path, [{"ts": now, "session": "s2", "role": "user", "text": "I love hiking."}])

    async def body():
        w = MemoryWriter(WriterConfig(chatlog_dir=tmp_path, state_path=tmp_path / "state.json", idle_s=120),
                         Relevan(relevan_server),
                         LLMConfig(base_url="http://127.0.0.1:1", model="m", api_key="k",
                                   reasoning_effort="", health_interval_s=30))
        assert await w.run_once() == 0   # still talking: not due yet
        await w.close()

    asyncio.run(body())
