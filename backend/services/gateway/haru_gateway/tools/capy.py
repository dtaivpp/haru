"""Capy (capy.ai) coding agents: what they're working on, and handing them work.

Haru watches every thread in your Capy projects (one list call per project per
poll; Capy's API has no webhooks) and posts a *silent* notification when a
thread finishes (working -> idle), fails, or starts needing you. Nothing is
spoken: the bell on the face page lights up, and you ask Haru when it suits you.

Tools:
  capy_tasks        recent threads and their status ("what's Capy working on?")
  capy_task         one thread's status and Capy's latest reply ("what did it say?")
  capy_start_task   hand Capy new work; read back to the user first (confirmed=true)
  capy_reply        answer a thread that needs you; also read back first

Capy opens its own pull requests through its GitHub App; Haru never writes to
GitHub. Auth: CAPY_SERVICE_API_KEY (a Capy service-user key, Settings -> API).
API: https://docs.capy.ai/api-reference/overview
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from haru_gateway.tools import Tool, ToolContext, ToolError
from haru_gateway.tools.notifications import NotificationService, _ago

log = logging.getLogger("haru.gateway.capy")

USER_AGENT = "haru-gateway/0.1"
# Event ids are ULIDs. Paging *before* the largest possible one returns the newest
# page of a transcript (with no cursor, Capy returns the oldest page).
NEWEST = "7ZZZZZZZZZZZZZZZZZZZZZZZZZ"
REPLY_CHARS = 600                    # how much of Capy's latest reply goes in a notification / tool result
PROJECTS_REFRESH_S = 3600


@dataclass
class CapyConfig:
    api_key: str
    base_url: str = "https://api.capy.ai/api/v1"
    projects: list[str] = field(default_factory=list)   # ids or codes to watch; empty: all
    poll_s: float = 60.0
    daily_tasks: int = 10                              # cap on tasks Haru starts per day
    state_path: Path = Path("/data/state/capy.json")

    @classmethod
    def from_env(cls) -> CapyConfig | None:
        e = os.environ.get
        key = e("CAPY_SERVICE_API_KEY", "")
        if not key:
            return None
        return cls(api_key=key,
                   base_url=e("HARU_CAPY_BASE_URL", "https://api.capy.ai/api/v1").rstrip("/"),
                   projects=[p.strip() for p in e("HARU_CAPY_PROJECTS", "").split(",") if p.strip()],
                   poll_s=float(e("HARU_CAPY_POLL_S", "60")),
                   daily_tasks=int(e("HARU_CAPY_DAILY_TASKS", "10")),
                   state_path=Path(e("HARU_CAPY_STATE", "/data/state/capy.json")))


class CapyError(Exception):
    pass


class CapyClient:
    def __init__(self, config: CapyConfig, timeout: float = 15.0) -> None:
        self.http = httpx.AsyncClient(base_url=config.base_url, timeout=timeout, headers={
            "Authorization": f"Bearer {config.api_key}", "User-Agent": USER_AGENT})

    async def close(self) -> None:
        await self.http.aclose()

    async def _call(self, method: str, path: str, **kw) -> dict:
        try:
            r = await self.http.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise CapyError(f"{type(exc).__name__}: {exc}") from exc
        if r.status_code >= 400:
            try:
                tag = r.json().get("_tag", "")
            except ValueError:
                tag = ""
            raise CapyError(f"HTTP {r.status_code} {tag} on {method} {path}".strip())
        return r.json() if r.content else {}

    async def projects(self) -> list[dict]:
        return (await self._call("GET", "/projects")).get("items") or []

    async def threads(self, project_id: str, limit: int = 50) -> list[dict]:
        data = await self._call("GET", "/threads", params={"projectId": project_id, "limit": limit})
        return data.get("items") or []

    async def thread(self, thread_id: str) -> dict:
        return await self._call("GET", f"/threads/{thread_id}")

    async def last_reply(self, thread_id: str, pages: int = 3) -> str | None:
        """Capy's most recent assistant message, newest pages first."""
        before = NEWEST
        for _ in range(pages):
            data = await self._call("GET", f"/threads/{thread_id}/messages", params={"before": before, "limit": 50})
            items = data.get("items") or []
            for m in reversed(items):
                if m.get("source") == "assistant" and (m.get("text") or "").strip():
                    return m["text"]
            before = data.get("beforeCursor")
            if not before or not items:
                return None
        return None

    async def create_thread(self, project_id: str, message: str, title: str | None = None) -> dict:
        body = {"requestId": f"haru-{uuid.uuid4().hex}", "projectId": project_id, "message": message}
        if title:
            body["title"] = title
        return await self._call("POST", "/threads", json=body)

    async def send(self, thread_id: str, text: str) -> dict:
        # "queue": wait for the current step instead of interrupting work in progress.
        return await self._call("POST", f"/threads/{thread_id}/message",
                                json={"text": text, "delivery": "queue", "clientKey": uuid.uuid4().hex})


def plain(text: str, limit: int = REPLY_CHARS) -> str:
    """Markdown-ish agent output -> a short plain paragraph for a screen card or for speech."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s*(#+|[-*]|\d+\.)\s+", "", text, flags=re.M)
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return (cut[:end + 1] if end > limit // 2 else cut.rstrip() + "…")


class CapyService:
    def __init__(self, config: CapyConfig, client: CapyClient, notifications: NotificationService,
                 tz: ZoneInfo) -> None:
        self.config = config
        self.client = client
        self.notifications = notifications
        self.tz = tz
        self.projects: dict[str, dict] = {}      # id -> project
        self._projects_at = 0.0
        # threads: id -> {"status", "needs_you", "title", "project", "updated"}; started: {date: count}
        self.state: dict = self._load()

    # --- persistence --------------------------------------------------------------------

    def _load(self) -> dict:
        try:
            s = json.loads(self.config.state_path.read_text())
            return {"threads": s.get("threads") or {}, "started": s.get("started") or {},
                    "seeded": bool(s.get("seeded"))}
        except (OSError, ValueError):
            return {"threads": {}, "started": {}, "seeded": False}

    def _save(self) -> None:
        try:
            self.config.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.config.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=1))
            tmp.replace(self.config.state_path)
        except OSError as exc:
            log.warning("could not save capy state: %s", exc)

    # --- watching --------------------------------------------------------------------------

    async def refresh_projects(self, force: bool = False) -> None:
        if not force and self.projects and time.monotonic() - self._projects_at < PROJECTS_REFRESH_S:
            return
        wanted = set(self.config.projects)
        projects = await self.client.projects()
        self.projects = {p["id"]: p for p in projects
                         if not wanted or p["id"] in wanted or p.get("code") in wanted or p.get("name") in wanted}
        self._projects_at = time.monotonic()
        log.info("capy: watching %s", ", ".join(f"{p['name']} ({p.get('code')})" for p in self.projects.values())
                 or "no projects")

    async def poll(self) -> int:
        """One pass over every watched project. Returns how many notifications were posted."""
        await self.refresh_projects()
        known: dict = self.state["threads"]
        first = not self.state["seeded"]   # first run ever: learn what exists, announce nothing
        posted = 0
        for pid in self.projects:
            for t in await self.client.threads(pid):
                tid, status = t["id"], t.get("status", "")
                needs_you = bool(t.get("needsYou"))
                prev = known.get(tid)
                known[tid] = {"status": status, "needs_you": needs_you, "title": t.get("title") or "untitled task",
                              "project": pid, "updated": t.get("lastActivityAt") or t.get("updatedAt") or ""}
                if first or prev is None or status == "archived":
                    continue
                event = self._event(prev, status, needs_you)
                if event:
                    await self._notify(event, t)
                    posted += 1
        for tid in [i for i, t in known.items() if t["status"] == "archived"]:
            del known[tid]
        if len(known) > 300:   # keep the most recently active
            keep = sorted(known.items(), key=lambda kv: kv[1].get("updated", ""), reverse=True)[:300]
            self.state["threads"] = dict(keep)
        self.state["seeded"] = True
        self._save()
        return posted

    @staticmethod
    def _event(prev: dict, status: str, needs_you: bool) -> str | None:
        if needs_you and not prev.get("needs_you"):
            return "needs_you"
        if status == "failed" and prev.get("status") != "failed":
            return "failed"
        if status == "idle" and prev.get("status") in ("working", "waiting"):
            return "finished"
        return None

    async def _notify(self, event: str, t: dict) -> None:
        title = t.get("title") or "untitled task"
        heading = {"finished": "Capy finished", "failed": "Capy task failed", "needs_you": "Capy needs you"}[event]
        body = ""
        if event == "needs_you" and (ask := (t.get("ask") or {}).get("text")):
            body = plain(ask)
        else:
            try:
                body = plain(await self.client.last_reply(t["id"]) or "")
            except CapyError as exc:
                log.warning("capy: couldn't read the reply on %s: %s", t["id"], exc)
        await self.notifications.add(f"{heading}: {title}", body, source="capy",
                                     ref={"capy_thread_id": t["id"]}, key=f"capy:{t['id']}")

    async def run_forever(self) -> None:
        while True:
            try:
                n = await self.poll()
                if n:
                    log.info("capy: %d new notification%s", n, "s" * (n != 1))
            except CapyError as exc:
                log.warning("capy poll failed: %s", exc)
            except Exception:
                log.exception("capy poll crashed")
            await asyncio.sleep(self.config.poll_s)

    # --- tools -----------------------------------------------------------------------------

    def _find(self, task: str) -> tuple[str, dict]:
        """A thread by id, or by (part of) its title among the threads we've seen."""
        known = self.state["threads"]
        if task in known:
            return task, known[task]
        if task.startswith("jam_"):
            return task, {}
        q = task.lower()
        hits = [(i, t) for i, t in known.items() if q in t["title"].lower()]
        if not hits:   # every word present, in any order
            words = q.split()
            hits = [(i, t) for i, t in known.items() if all(w in t["title"].lower() for w in words)]
        if not hits:
            raise ToolError(f"no Capy task matches {task!r}; use capy_tasks to see what exists")
        if len(hits) > 1:
            hits.sort(key=lambda kv: kv[1].get("updated", ""), reverse=True)
            if q != hits[0][1]["title"].lower():
                raise ToolError("several tasks match: " + "; ".join(f"{t['title']} ({i})" for i, t in hits[:5])
                                + ". Ask which one, then pass its id")
        return hits[0]

    def _project(self, name: str | None) -> str:
        if not self.projects:
            raise ToolError("no Capy projects are available")
        if not name:
            if len(self.projects) == 1:
                return next(iter(self.projects))
            raise ToolError("which project? " + ", ".join(p["name"] for p in self.projects.values()))
        q = name.lower()
        for pid, p in self.projects.items():
            repos = " ".join(r.get("repoFullName", "") for r in p.get("repos") or [])
            if q in (pid.lower(), (p.get("code") or "").lower()) or q in p["name"].lower() or q in repos.lower():
                return pid
        raise ToolError(f"no Capy project matches {name!r}; projects: "
                        + ", ".join(p["name"] for p in self.projects.values()))

    def _started_today(self) -> int:
        return int(self.state["started"].get(str(dt.date.today()), 0))

    def tools(self) -> list[Tool]:
        def when(iso: str) -> str:
            try:
                return _ago(dt.datetime.fromisoformat(iso.replace("Z", "+00:00")), dt.datetime.now(dt.timezone.utc))
            except ValueError:
                return ""

        async def capy_tasks(args: dict, ctx: ToolContext) -> dict:
            await self.refresh_projects()
            wanted = args.get("status")
            items = []
            for pid in self.projects:
                for t in await self.client.threads(pid, limit=20):
                    if t.get("status") == "archived" or (wanted and t.get("status") != wanted):
                        continue
                    items.append({"id": t["id"], "title": t.get("title") or "untitled", "status": t.get("status"),
                                  "needs_you": bool(t.get("needsYou")),
                                  "last_activity": when(t.get("lastActivityAt") or t.get("updatedAt") or ""),
                                  "_sort": t.get("lastActivityAt") or t.get("updatedAt") or ""})
            items.sort(key=lambda i: i.pop("_sort"), reverse=True)
            return {"tasks": items[:10], "total": len(items)}

        async def capy_task(args: dict, ctx: ToolContext) -> dict:
            tid, _ = self._find(str(args["task"]).strip())
            try:
                t = await self.client.thread(tid)
                reply = await self.client.last_reply(tid)
            except CapyError as exc:
                raise ToolError(f"couldn't read that task: {exc}") from exc
            out = {"id": tid, "title": t.get("title"), "status": t.get("status"), "needs_you": bool(t.get("needsYou")),
                   "last_activity": when(t.get("lastActivityAt") or t.get("updatedAt") or ""),
                   "latest_reply": plain(reply or "", 1500) or None}
            if (ask := (t.get("ask") or {}).get("text")):
                out["question_for_you"] = plain(ask, 600)
            return out

        async def capy_start_task(args: dict, ctx: ToolContext) -> dict:
            request = str(args["request"]).strip()
            if len(request) < 10:
                raise ToolError("the request is too short; ask the user what exactly Capy should do")
            await self.refresh_projects()
            pid = self._project(args.get("project"))
            if not args.get("confirmed"):
                return {"not_started": True, "project": self.projects[pid]["name"], "request": request,
                        "next": "Read this request back to the user in one sentence and ask whether to send it. "
                                "Only if they clearly say yes, call capy_start_task again with confirmed=true."}
            if self._started_today() >= self.config.daily_tasks:
                raise ToolError(f"the daily limit of {self.config.daily_tasks} new Capy tasks is reached")
            try:
                t = await self.client.create_thread(pid, request, args.get("title"))
            except CapyError as exc:
                raise ToolError(f"Capy didn't accept the task: {exc}") from exc
            today = str(dt.date.today())
            self.state["started"] = {today: self._started_today() + 1}
            self.state["threads"][t["id"]] = {"status": t.get("status", "working"), "needs_you": False,
                                              "title": t.get("title") or request[:60], "project": pid,
                                              "updated": t.get("createdAt", "")}
            self._save()
            log.info("capy: started %s in %s: %r", t["id"], self.projects[pid]["name"], request[:120])
            return {"started": True, "id": t["id"], "title": t.get("title") or request[:60],
                    "note": "You'll get a notification on screen when it finishes or needs the user."}

        async def capy_reply(args: dict, ctx: ToolContext) -> dict:
            tid, known = self._find(str(args["task"]).strip())
            text = str(args["text"]).strip()
            if not text:
                raise ToolError("nothing to send")
            if not args.get("confirmed"):
                return {"not_sent": True, "task": known.get("title", tid), "text": text,
                        "next": "Read the message back to the user and ask whether to send it. Only if they "
                                "clearly say yes, call capy_reply again with confirmed=true."}
            try:
                await self.client.send(tid, text)
            except CapyError as exc:
                raise ToolError(f"Capy didn't accept the message: {exc}") from exc
            log.info("capy: replied on %s: %r", tid, text[:120])
            return {"sent": True, "task": known.get("title", tid)}

        task_arg = {"type": "string", "description": "Task id (jam_...) or a few words of its title"}
        return [
            Tool("capy_tasks", "List the user's recent Capy coding-agent tasks (Capy is the AI coding agent that "
                 "works on their repositories and opens pull requests) with status: working, waiting, idle "
                 "(done), failed.",
                 {"type": "object", "properties": {
                     "status": {"type": "string", "enum": ["working", "waiting", "idle", "failed"],
                                "description": "Only tasks in this state"},
                 }}, capy_tasks),
            Tool("capy_task", "Get one Capy task's status and Capy's latest reply (what it did, the PR it opened, "
                 "or what it's asking). Use for follow-ups on a Capy notification; its ref has the task id.",
                 {"type": "object", "properties": {"task": task_arg}, "required": ["task"]}, capy_task),
            Tool("capy_start_task", "Hand a new coding task to Capy. Write the request the way you'd brief an "
                 "engineer: what to change and why, with any details the user gave. First call without "
                 "confirmed; then read it back and only start it once the user says yes.",
                 {"type": "object", "properties": {
                     "request": {"type": "string", "description": "The task for Capy, in full sentences"},
                     "title": {"type": "string", "description": "Short title, a few words"},
                     "project": {"type": "string", "description": "Project or repo name, if there's more than one"},
                     "confirmed": {"type": "boolean", "description": "true only after the user agreed to send it"},
                 }, "required": ["request"]}, capy_start_task, timeout_s=30),
            Tool("capy_reply", "Send a message to a Capy task, e.g. to answer its question when it needs the "
                 "user. Read it back and get a yes before sending (confirmed=true).",
                 {"type": "object", "properties": {
                     "task": task_arg,
                     "text": {"type": "string", "description": "The message for Capy"},
                     "confirmed": {"type": "boolean", "description": "true only after the user agreed to send it"},
                 }, "required": ["task", "text"]}, capy_reply, timeout_s=30),
        ]
