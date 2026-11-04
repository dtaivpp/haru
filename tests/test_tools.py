"""Tool-calling tests: the registry's reliability layer, the streaming tool loop
in the conversation, and timers/reminders."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo

import pytest

from haru_gateway.conversation import FILLERS, MAX_TOOL_ROUNDS, ConversationConfig, ConversationManager
from haru_gateway.llm import LLMConfig
from haru_gateway.stt import Transcript
from haru_gateway.tools import Tool, ToolCall, ToolContext, ToolError, ToolRegistry, validate
from haru_gateway.tools.timers import TimerService

CTX = ToolContext(device_id="pi")


# --- registry ------------------------------------------------------------------------

def test_validate():
    schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1},
                                               "mode": {"type": "string", "enum": ["a", "b"]},
                                               "flag": {"type": "boolean"}}, "required": ["n"]}
    assert validate({"n": 3, "mode": "a"}, schema) == []
    assert "missing required argument 'n'" in validate({}, schema)[0]
    assert "must be integer" in validate({"n": "3"}, schema)[0]
    assert "must be integer" in validate({"n": True}, schema)[0]      # bools aren't ints
    assert "must be one of" in validate({"n": 1, "mode": "c"}, schema)[0]
    assert ">= 1" in validate({"n": 0}, schema)[0]
    assert validate([], schema) == ["arguments must be a JSON object"]


def test_registry_turns_every_failure_into_a_result():
    async def ok(args, ctx):
        return {"echo": args["x"], "device": ctx.device_id}

    async def boom(args, ctx):
        raise RuntimeError("kaboom")

    async def expected(args, ctx):
        raise ToolError("no such place")

    async def slow(args, ctx):
        await asyncio.sleep(1)

    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    reg = ToolRegistry()
    reg.add(Tool("ok", "", schema, ok), Tool("boom", "", {"type": "object"}, boom),
            Tool("expected", "", {"type": "object"}, expected), Tool("slow", "", {"type": "object"}, slow, timeout_s=0.05))

    async def body():
        r = await reg.run(ToolCall("1", "ok", '{"x": "hi"}'), CTX)
        assert r.ok and json.loads(r.content) == {"echo": "hi", "device": "pi"}
        for call, fragment in [(ToolCall("2", "nope", "{}"), "unknown tool"),
                               (ToolCall("3", "ok", "{not json"), "not valid JSON"),
                               (ToolCall("4", "ok", '{"x": 5}'), "must be string"),
                               (ToolCall("5", "boom", "{}"), "kaboom"),
                               (ToolCall("6", "expected", "{}"), "no such place"),
                               (ToolCall("7", "slow", "{}"), "timed out")]:
            r = await reg.run(call, CTX)
            assert not r.ok and fragment in json.loads(r.content)["error"], (call, r.content)

    asyncio.run(body())
    assert [s["function"]["name"] for s in reg.specs()] == ["ok", "boom", "expected", "slow"]


# --- the streaming tool loop -----------------------------------------------------------

class ToolLLM(BaseHTTPRequestHandler):
    requests: list[dict] = []
    always_call = False

    def _sse(self, chunks):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for c in chunks:
            self.wfile.write(f"data: {json.dumps({'choices': [c]})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ToolLLM.requests.append(body)
        tool_results = [m for m in body["messages"] if m["role"] == "tool"]
        if body.get("tools") and (ToolLLM.always_call or not tool_results):
            # A tool call, streamed in fragments the way vLLM does it.
            self._sse([
                {"index": 0, "delta": {"role": "assistant", "tool_calls": [
                    {"index": 0, "id": f"call_{len(ToolLLM.requests)}", "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"loca'}}]}},
                {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'tion": "Denver"}'}}]}},
                {"index": 0, "delta": {}, "finish_reason": "tool_calls"},
            ])
        else:
            temp = json.loads(tool_results[-1]["content"])["temp"] if tool_results else "?"
            self._sse([{"index": 0, "delta": {"content": '<|ACT {"emotion":"happy"}|>'}},
                       {"index": 0, "delta": {"content": f"It's {temp} degrees in Denver. "}},
                       {"index": 0, "delta": {"content": "Enjoy!"}, "finish_reason": "stop"}])

    def log_message(self, *a):
        pass


@pytest.fixture
def tool_llm():
    ToolLLM.requests, ToolLLM.always_call = [], False
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ToolLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def manager_with_weather(base, calls):
    async def weather(args, ctx):
        calls.append((args, ctx.device_id))
        return {"temp": 71}

    reg = ToolRegistry()
    reg.add(Tool("get_weather", "weather", {"type": "object", "properties": {"location": {"type": "string"}}}, weather))
    llm = LLMConfig(base_url=base, model="m", api_key="k", reasoning_effort="", health_interval_s=30)
    return ConversationManager(llm, ConversationConfig(persona="You are Haru."), tools=reg)


async def wait_for(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


def test_tool_loop_calls_tool_speaks_filler_and_answers(tool_llm):
    async def body():
        calls, said, replies = [], [], []
        m = manager_with_weather(tool_llm, calls)
        m.on_sentence(lambda dev, s, started: said.append(s))
        m.on_reply(replies.append)
        m.handle_transcript(Transcript("pi/1", "what's the weather in Denver", 0, 0, 0, 0))
        await wait_for(lambda: replies)
        assert calls == [({"location": "Denver"}, "pi")]          # fragments reassembled, context passed
        assert said[0] in FILLERS                                  # no dead air while the tool ran
        assert said[1:] == ["It's 71 degrees in Denver.", "Enjoy!"]
        assert replies[0].text == "It's 71 degrees in Denver. Enjoy!"  # filler isn't part of the answer
        second = ToolLLM.requests[1]["messages"]
        assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert second[-1] == {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 71}'}
        roles = [x["role"] for x in m.conversations["pi"].history]
        assert roles == ["user", "assistant", "tool", "assistant"]  # follow-ups can see the tool round
        await m.close()

    asyncio.run(body())


def test_tool_rounds_are_bounded(tool_llm):
    ToolLLM.always_call = True   # a model that would call tools forever

    async def body():
        calls, replies = [], []
        m = manager_with_weather(tool_llm, calls)
        m.on_reply(replies.append)
        m.handle_transcript(Transcript("pi/1", "weather?", 0, 0, 0, 0))
        await wait_for(lambda: replies)
        assert len(calls) == MAX_TOOL_ROUNDS
        assert "tools" not in ToolLLM.requests[-1]                  # the last round must answer
        assert replies[0].text.startswith("It's 71 degrees")
        await m.close()

    asyncio.run(body())


# --- timers ----------------------------------------------------------------------------------

TZ = ZoneInfo("America/New_York")


def test_timer_fires_and_announces(tmp_path):
    async def body():
        said = []

        async def announce(dev, text):
            said.append((dev, text))

        svc = TimerService(tmp_path / "timers.json", TZ, announce)
        svc.start()
        tools = {t.name: t for t in svc.tools()}
        out = await tools["set_timer"].handler({"duration_seconds": 1, "label": "pasta"}, CTX)
        assert out["duration"] == "1 second" and out["set"]["label"] == "pasta"
        listed = await tools["list_timers"].handler({}, CTX)
        assert len(listed["timers_and_reminders"]) == 1
        await wait_for(lambda: said, timeout=3)
        assert said == [("pi", "Time's up! That's your pasta timer.")]
        assert svc.alarms == {} and json.loads((tmp_path / "timers.json").read_text()) == []
        svc.close()

    asyncio.run(body())


def test_reminders_persist_cancel_and_reject_past_times(tmp_path):
    async def body():
        said = []

        async def announce(dev, text):
            said.append(text)

        svc = TimerService(tmp_path / "timers.json", TZ, announce)
        svc.start()
        tools = {t.name: t for t in svc.tools()}
        await tools["set_reminder"].handler({"message": "call the dentist", "in_minutes": 30}, CTX)
        await tools["set_timer"].handler({"duration_seconds": 600, "label": "laundry"}, CTX)
        with pytest.raises(ToolError, match="already passed"):
            await tools["set_reminder"].handler({"message": "x", "at": "2000-01-01T09:00"}, CTX)
        with pytest.raises(ToolError, match="either in_minutes or at"):
            await tools["set_reminder"].handler({"message": "x"}, CTX)
        out = await tools["cancel_timer"].handler({"label": "laund"}, CTX)
        assert out["cancelled"]["label"] == "laundry"
        svc.close()

        # A restarted gateway gets the reminder back from disk.
        svc2 = TimerService(tmp_path / "timers.json", TZ, announce)
        assert [a.label for a in svc2.alarms.values()] == ["call the dentist"]
        # And one that fell due while it was down is announced late, not dropped.
        a = next(iter(svc2.alarms.values()))
        a.due = (svc2.now() - dt.timedelta(minutes=5)).isoformat()
        svc2.start()
        await wait_for(lambda: said)
        assert said == ["Sorry, this one's late. Reminder: call the dentist."]
        svc2.close()

    asyncio.run(body())


# --- calendar ----------------------------------------------------------------------------------

class FakeCalendar(BaseHTTPRequestHandler):
    requests: list[tuple] = []

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import parse_qs, urlsplit
        url = urlsplit(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        FakeCalendar.requests.append(("GET", url.path, q, self.headers["Authorization"]))
        if "nobody" in url.path:
            return self._json({"error": "not found"}, 404)
        items = [
            {"summary": "Standup", "start": {"dateTime": "2026-10-02T15:30:00-04:00"},
             "end": {"dateTime": "2026-10-02T16:00:00-04:00"}, "location": "Zoom"},
            {"summary": "Dentist", "start": {"dateTime": "2026-10-02T09:00:00-04:00"},
             "end": {"dateTime": "2026-10-02T10:00:00-04:00"}},
            {"summary": "Holiday", "start": {"date": "2026-10-02"}, "end": {"date": "2026-10-03"}},
            {"summary": "Gone", "status": "cancelled", "start": {"dateTime": "2026-10-02T11:00:00-04:00"},
             "end": {"dateTime": "2026-10-02T12:00:00-04:00"}},
        ]
        if q.get("q"):
            items = [i for i in items if q["q"].lower() in i["summary"].lower()]
        self._json({"items": items})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeCalendar.requests.append(("POST", self.path, body, self.headers["Authorization"]))
        cal = body["items"][0]["id"]
        self._json({"calendars": {cal: {"busy": [
            {"start": "2026-10-02T13:00:00-04:00", "end": "2026-10-02T14:00:00-04:00"},
            {"start": "2026-10-02T13:30:00-04:00", "end": "2026-10-02T15:00:00-04:00"}]}}})

    def log_message(self, *a):
        pass


@pytest.fixture
def calendar_tools_fake(monkeypatch, tmp_path):
    FakeCalendar.requests = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeCalendar)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    from haru_gateway.tools import calendar as cal_mod

    class FakeToken:
        def __init__(self, sa_file): pass
        async def get(self): return "tok"

    monkeypatch.setattr(cal_mod, "_Token", FakeToken)
    monkeypatch.setattr(cal_mod, "API", f"http://127.0.0.1:{srv.server_port}")
    sa = tmp_path / "sa.json"
    sa.write_text("{}")
    monkeypatch.setenv("HARU_GOOGLE_SA_FILE", str(sa))
    monkeypatch.setenv("HARU_CALENDARS", "mine=me@example.com, work = w@example.com")
    yield {t.name: t for t in cal_mod.calendar_tools(TZ)}
    srv.shutdown()


def test_calendar_events_window_search_and_format(calendar_tools_fake):
    async def body():
        tools = calendar_tools_fake
        out = await tools["calendar_events"].handler({"start": "2026-10-02", "calendar": "mine"}, CTX)
        method, path, q, auth = FakeCalendar.requests[0]
        assert path == "/calendars/me@example.com/events" and auth == "Bearer tok"
        assert q["timeMin"] == "2026-10-02T00:00:00-04:00" and q["timeMax"] == "2026-10-03T00:00:00-04:00"
        assert q["singleEvents"] == "true" and q["timeZone"] == "America/New_York"
        assert [e["title"] for e in out["events"]] == ["Holiday", "Dentist", "Standup"]   # sorted, cancelled dropped
        standup = out["events"][2]
        assert standup["starts"] == "Friday October 2, 3:30 PM" and standup["location"] == "Zoom"
        assert out["events"][0]["all_day"] is True
        found = await tools["calendar_events"].handler({"start": "2026-10-02", "search": "dentist"}, CTX)
        assert [e["title"] for e in found["events"]] == ["Dentist", "Dentist"]           # both calendars searched
        with pytest.raises(ToolError, match="no calendar called"):
            await tools["calendar_events"].handler({"calendar": "golf"}, CTX)
        with pytest.raises(ToolError, match="end must be after start"):
            await tools["calendar_events"].handler({"start": "2026-10-02T10:00", "end": "2026-10-02T09:00"}, CTX)

    asyncio.run(body())


def test_calendar_availability_merges_busy_blocks(calendar_tools_fake):
    async def body():
        out = await calendar_tools_fake["calendar_availability"].handler(
            {"start": "2026-10-02T12:00", "end": "2026-10-02T17:00", "calendar": "mine"}, CTX)
        assert FakeCalendar.requests[-1][2]["items"] == [{"id": "me@example.com"}]
        assert out["free"] == ["Friday 12:00 PM to 1:00 PM", "Friday 3:00 PM to 5:00 PM"]   # overlap merged
        assert not out["completely_free"]

    asyncio.run(body())


def test_calendar_disabled_without_config(monkeypatch):
    from haru_gateway.tools import calendar as cal_mod
    monkeypatch.delenv("HARU_CALENDARS", raising=False)
    assert cal_mod.calendar_tools(TZ) == []


# --- Alexandria ----------------------------------------------------------------------------------

class FakeFirecrawl(BaseHTTPRequestHandler):
    calls: list[dict] = []
    contract_cost = 5
    terms_required = False

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeFirecrawl.calls.append((self.path, body))
        if self.path == "/search":
            return self._json({"success": True, "creditsUsed": 0, "data": {"tools": [
                {"provider": "bing-com", "capability": "businesses/search", "description": "Local businesses near a place"}]}})
        a = body["alexandria"]
        if a["capability"] == "find-tools":
            return self._json({"data": {"alexandria": [{"data": {"items": [{
                "provider": "bing-com", "capability": "businesses/search", "creditsCost": FakeFirecrawl.contract_cost,
                "perRecord": False, "options": [
                    {"name": "query", "type": "string", "required": True, "about": "What to find"},
                    {"name": "location", "type": "string", "about": "Where"},
                    {"name": "count", "type": "number", "default": 18, "min": 1, "max": 30}]}]}}]}})
        if FakeFirecrawl.terms_required:
            return self._json({"data": {"alexandria": [{"error": {"code": "THIRD_PARTY_DATA_TERMS_REQUIRED",
                               "requiresAction": {"url": "https://firecrawl.dev/terms/bing"}}}]}})
        return self._json({"data": {"alexandria": [{"creditsCost": 5, "data": {"results": [
            {"name": "Taxus Street Coffee", "rating": 5, "image_url": "https://x/img", "maps_url": "https://x/map"}]}}]}})

    def log_message(self, *a):
        pass


@pytest.fixture
def alexandria(monkeypatch, tmp_path):
    FakeFirecrawl.calls, FakeFirecrawl.contract_cost, FakeFirecrawl.terms_required = [], 5, False
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeFirecrawl)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    from haru_gateway.tools import alexandria as alex_mod
    monkeypatch.setattr(alex_mod, "FIRECRAWL", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc")
    monkeypatch.setenv("HARU_ALEXANDRIA_BUDGET_FILE", str(tmp_path / "budget.json"))
    monkeypatch.setenv("HARU_ALEXANDRIA_DAILY_CREDITS", "12")
    yield {t.name: t for t in alex_mod.alexandria_tools()}, tmp_path
    srv.shutdown()


def test_alexandria_find_run_budget_and_terms(alexandria):
    tools, tmp = alexandria

    async def body():
        found = await tools["find_data_tools"].handler({"need": "coffee shops near an address"}, CTX)
        t = found["tools"][0]
        assert (t["provider"], t["capability"], t["credits"]) == ("bing-com", "businesses/search", 5)
        assert t["options"][0] == {"name": "query", "type": "string", "required": True, "about": "What to find"}
        assert FakeFirecrawl.calls[0][1]["sources"] == ["alexandria"]          # discovery is the free source

        with pytest.raises(ToolError, match="missing required options"):
            await tools["run_data_tool"].handler({"provider": "bing-com", "capability": "businesses/search"}, CTX)
        out = await tools["run_data_tool"].handler({"provider": "bing-com", "capability": "businesses/search",
                                                    "options": {"query": "coffee", "location": "Denver, CO"}}, CTX)
        assert out["credits_used"] == 5 and out["data"]["results"] == [{"name": "Taxus Street Coffee", "rating": 5}]
        assert json.loads((tmp / "budget.json").read_text())["spent"] == 5
        await tools["run_data_tool"].handler({"provider": "bing-com", "capability": "businesses/search",
                                              "options": {"query": "tea"}}, CTX)
        with pytest.raises(ToolError, match="budget"):                           # 10 spent, cap 12
            await tools["run_data_tool"].handler({"provider": "bing-com", "capability": "businesses/search",
                                                  "options": {"query": "donuts"}}, CTX)

    asyncio.run(body())


def test_alexandria_price_cap_and_terms(alexandria, monkeypatch):
    tools, tmp = alexandria

    async def body():
        FakeFirecrawl.terms_required = True
        with pytest.raises(ToolError, match="terms accepted by a Firecrawl org admin at https://firecrawl.dev/terms/bing"):
            await tools["run_data_tool"].handler({"provider": "bing-com", "capability": "businesses/search",
                                                  "options": {"query": "x"}}, CTX)

    asyncio.run(body())
    # A fresh registry (fresh contract cache) with an expensive tool is refused before any spend.
    FakeFirecrawl.contract_cost = 80
    from haru_gateway.tools import alexandria as alex_mod
    fresh = {t.name: t for t in alex_mod.alexandria_tools()}

    async def body2():
        with pytest.raises(ToolError, match="costs 80 credits per call, over Haru's limit of 25"):
            await fresh["run_data_tool"].handler({"provider": "bing-com", "capability": "businesses/search",
                                                  "options": {"query": "x"}}, CTX)

    asyncio.run(body2())


# --- MCP ------------------------------------------------------------------------------------

def test_mcp_server_tools_read_only_and_errors():
    import sys
    from pathlib import Path
    from haru_gateway.tools.mcp import MCPServer, MCPServerConfig
    fake = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"

    async def body():
        reg = ToolRegistry()
        srv = MCPServer(MCPServerConfig(name="gh", command=sys.executable, args=[str(fake)]), reg)
        runner = asyncio.create_task(srv.run_forever())
        await asyncio.wait_for(srv.ready.wait(), 20)
        assert "gh_lookup" in reg.tools and "gh_broken" in reg.tools
        assert "gh_delete_repo" not in reg.tools            # not annotated read-only: refused
        r = await reg.run(ToolCall("1", "gh_lookup", '{"repo": "haru"}'), CTX)
        assert r.ok and "haru has 42 stars" in r.content
        r = await reg.run(ToolCall("2", "gh_lookup", '{}'), CTX)        # schema from the server enforced
        assert not r.ok and "missing required argument 'repo'" in r.content
        r = await reg.run(ToolCall("3", "gh_broken", '{}'), CTX)
        assert not r.ok and "upstream exploded" in r.content
        runner.cancel()

    asyncio.run(body())


def test_gated_tool_groups_only_offered_on_topic(tool_llm):
    async def noop(args, ctx):
        return {"ok": True}

    async def body():
        calls = []
        m = manager_with_weather(tool_llm, calls)
        m.tools.add(Tool("github_list_notifications", "gh", {"type": "object"}, noop, group="github"))
        replies = []
        m.on_reply(replies.append)
        offered = lambda i: [t["function"]["name"] for t in ToolLLM.requests[i].get("tools", [])]
        m.handle_transcript(Transcript("pi/1", "weather in Denver?", 0, 0, 0, 0))
        await wait_for(lambda: len(replies) == 1)
        assert "github_list_notifications" not in offered(0)
        n = len(ToolLLM.requests)
        m.handle_transcript(Transcript("pi/1", "any new pull requests for me?", 0, 0, 0, 0))
        await wait_for(lambda: len(replies) == 2)
        assert "github_list_notifications" in offered(n)
        n = len(ToolLLM.requests)
        m.handle_transcript(Transcript("pi/1", "what about the other one?", 0, 0, 0, 0))   # sticky follow-up
        await wait_for(lambda: len(replies) == 3)
        assert "github_list_notifications" in offered(n)
        await m.close()

    asyncio.run(body())


def test_github_search_is_pinned_to_the_org():
    from haru_gateway.tools.mcp import scope_github_search
    rw = scope_github_search("acme")
    assert rw("search_pull_requests", {"query": "is:pr is:open review:required"})["query"] == \
        "is:pr is:open review:required org:acme"
    assert rw("search_issues", {"query": "bug repo:acme/monorepo"})["query"] == "bug repo:acme/monorepo"
    assert rw("search_repositories", {"query": "user:dtaivpp"})["query"] == "user:dtaivpp"
    assert rw("list_issues", {"owner": "x", "repo": "y"}) == {"owner": "x", "repo": "y"}   # non-search untouched


def test_act_marker_motion_fires_gesture():
    async def body():
        m = ConversationManager(LLMConfig(base_url="http://127.0.0.1:1", model="m", api_key="k", reasoning_effort="",
                                          health_interval_s=30), ConversationConfig(persona="p"))
        gestures, emotions = [], []
        m.on_gesture(lambda dev, g: gestures.append((dev, g)))
        m.on_emotion(lambda dev, e: emotions.append(e))
        out = await m._take_markers("pi", '<|ACT {"emotion":"happy","motion":"wave"}|>Hi there!')
        assert out == "Hi there!" and gestures == [("pi", "wave")] and emotions == ["happy"]
        out = await m._take_markers("pi", '<|ACT {"emo')            # incomplete marker: held, nothing fired
        assert out == '<|ACT {"emo' and len(gestures) == 1
        await m.close()

    asyncio.run(body())
