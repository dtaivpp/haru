"""Capy integration: thread watching -> silent notifications, and the tools (against a fake Capy API)."""

from __future__ import annotations

import asyncio
import json
from zoneinfo import ZoneInfo

import httpx
import pytest

from haru_gateway.tools import ToolCall, ToolContext, ToolRegistry
from haru_gateway.tools.capy import NEWEST, CapyClient, CapyConfig, CapyService, plain
from haru_gateway.tools.notifications import NotificationService

TZ = ZoneInfo("UTC")
P = "project_1"


class FakeCapy:
    def __init__(self):
        self.threads = {
            "jam_a": {"id": "jam_a", "title": "Fix the flaky CI timeout", "status": "working", "needsYou": False},
            "jam_b": {"id": "jam_b", "title": "Migrate auth to OAuth 2.1", "status": "working", "needsYou": False},
        }
        self.messages = {
            # The newest page has only tool output: last_reply must page further back.
            "jam_a": [[{"id": "01A", "source": "user", "text": "fix it"},
                       {"id": "01B", "source": "assistant", "text": "**Done.** Opened `PR #128`; all checks passed."}],
                      [{"id": "01C", "source": "tool", "text": "ok"}]],
            "jam_b": [[{"id": "02A", "source": "assistant", "text": "Working on it."}]],
        }
        self.created, self.sent, self.requests = [], [], []

    def handler(self, req: httpx.Request) -> httpx.Response:
        assert req.headers["authorization"] == "Bearer capy_test"
        path, q = req.url.path.removeprefix("/api/v1"), req.url.params
        self.requests.append((req.method, path))
        if path == "/projects":
            return httpx.Response(200, json={"items": [{"id": P, "name": "acme +2", "code": "ACME",
                                                        "repos": [{"repoFullName": "acme/monorepo"}]}]})
        if path == "/threads" and req.method == "GET":
            assert q["projectId"] == P
            return httpx.Response(200, json={"items": list(self.threads.values()), "cursor": None})
        if path == "/threads" and req.method == "POST":
            body = json.loads(req.content)
            self.created.append(body)
            t = {"id": "jam_new", "title": body.get("title"), "status": "working", "createdAt": "2026-11-04T00:00:00Z"}
            return httpx.Response(200, json=t)
        tid = path.split("/")[2]
        if path.endswith("/messages"):
            pages = self.messages[tid]
            idx = 0 if q["before"] == NEWEST else int(q["before"])
            items = pages[-1 - idx] if idx < len(pages) else []
            more = idx + 1 < len(pages)
            return httpx.Response(200, json={"items": items, "cursor": None, "beforeCursor": str(idx + 1) if more else None})
        if path.endswith("/message"):
            self.sent.append((tid, json.loads(req.content)))
            return httpx.Response(200, json={"id": "x" * 26, "deduped": False})
        return httpx.Response(200, json=self.threads[tid])


@pytest.fixture
def capy(tmp_path):
    fake = FakeCapy()
    cfg = CapyConfig(api_key="capy_test", state_path=tmp_path / "capy.json", daily_tasks=1)
    client = CapyClient(cfg)
    client.http = httpx.AsyncClient(base_url=cfg.base_url, transport=httpx.MockTransport(fake.handler),
                                    headers=client.http.headers)
    notes = NotificationService(tmp_path / "notifications.json", TZ)
    return fake, CapyService(cfg, client, notes, TZ), notes


def test_plain_strips_markdown_and_trims():
    assert plain("## Done\n- **Opened** `PR #12`, see [link](http://x)") == "Done Opened PR #12, see link"
    long = "First sentence here. " * 60
    out = plain(long, 100)
    assert len(out) <= 100 and out.endswith(".")


def test_first_poll_is_silent_then_transitions_notify(capy):
    async def body():
        fake, svc, notes = capy
        assert await svc.poll() == 0                       # learns what exists; announces nothing
        assert notes.items == {}
        fake.threads["jam_a"]["status"] = "idle"           # finished
        fake.threads["jam_b"].update(status="waiting", needsYou=True,
                                     ask={"text": "Should refresh tokens rotate on every use?"})
        assert await svc.poll() == 2
        got = {n.title: n for n in notes.items.values()}
        done = got["Capy finished: Fix the flaky CI timeout"]
        assert done.body == "Done. Opened PR #128; all checks passed." and done.source == "capy"
        assert done.ref == {"capy_thread_id": "jam_a"} and done.device_id is None
        assert got["Capy needs you: Migrate auth to OAuth 2.1"].body == "Should refresh tokens rotate on every use?"
        assert await svc.poll() == 0                       # no change, no repeat
        # A restart keeps state, so it doesn't re-announce or forget.
        again = CapyService(svc.config, svc.client, notes, TZ)
        assert again.state["seeded"] and again.state["threads"]["jam_a"]["status"] == "idle"
    asyncio.run(body())


def test_tools_read_and_start_with_confirmation(capy):
    async def body():
        fake, svc, _ = capy
        await svc.poll()
        reg = ToolRegistry()
        reg.add(*svc.tools())
        ctx = ToolContext("desk")

        async def call(name, **args):
            r = await reg.run(ToolCall("1", name, json.dumps(args)), ctx)
            return r.ok, json.loads(r.content)

        ok, r = await call("capy_tasks")
        assert ok and {t["title"] for t in r["tasks"]} == {"Fix the flaky CI timeout", "Migrate auth to OAuth 2.1"}
        ok, r = await call("capy_task", task="flaky CI")
        assert ok and r["id"] == "jam_a" and r["latest_reply"].startswith("Done.")
        ok, r = await call("capy_task", task="nonexistent thing")
        assert not ok and "no Capy task matches" in r["error"]

        # Unconfirmed: nothing is sent, Haru is told to read it back.
        ok, r = await call("capy_start_task", request="Add a retry with backoff to the Relevan client.")
        assert ok and r["not_started"] and fake.created == []
        ok, r = await call("capy_start_task", request="Add a retry with backoff to the Relevan client.",
                           title="Relevan retry", confirmed=True)
        assert ok and r["started"] and fake.created[0]["projectId"] == P
        assert fake.created[0]["requestId"].startswith("haru-") and svc.state["threads"]["jam_new"]
        # Daily cap (1 in this fixture).
        ok, r = await call("capy_start_task", request="Another task that is long enough.", confirmed=True)
        assert not ok and "daily limit" in r["error"]

        ok, r = await call("capy_reply", task="OAuth", text="Yes, rotate them.")
        assert ok and r["not_sent"] and fake.sent == []
        ok, r = await call("capy_reply", task="OAuth", text="Yes, rotate them.", confirmed=True)
        assert ok and fake.sent == [("jam_b", {"text": "Yes, rotate them.", "delivery": "queue",
                                               "clientKey": fake.sent[0][1]["clientKey"]})]
    asyncio.run(body())


def test_disabled_without_key(monkeypatch):
    monkeypatch.delenv("CAPY_SERVICE_API_KEY", raising=False)
    assert CapyConfig.from_env() is None


def test_repeat_updates_on_one_task_keep_only_the_latest(capy):
    """A task babysitting CI goes working -> idle again and again; the inbox keeps one entry for it."""
    async def body():
        fake, svc, notes = capy
        await svc.poll()
        for reply in ("CI running.", "CI still running.", "CI completed, all green."):
            fake.threads["jam_a"]["status"] = "working"
            await svc.poll()
            fake.messages["jam_a"] = [[{"id": "01Z", "source": "assistant", "text": reply}]]
            fake.threads["jam_a"]["status"] = "idle"
            assert await svc.poll() == 1
            if reply == "CI running.":
                await notes.open("desk")          # read the first one; a newer update still comes back unread
        assert [(n.body, n.read) for n in notes.items.values()] == [("CI completed, all green.", False)]
        assert notes.unread_count("desk") == 1
    asyncio.run(body())
