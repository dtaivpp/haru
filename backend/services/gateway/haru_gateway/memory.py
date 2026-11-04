"""Haru's long-term memory, read side: what she remembers during a conversation.

Memories live in a Relevan index (written by memory_writer.py, which mines the
chat log). During a session nothing here ever blocks a reply:

- When a session starts (the moment the VAD hears speech, before STT has even
  finished), all active memories are pulled in the background into a
  per-session cache.
- Each turn, `recall()` fires a Relevan search on what the user said and waits
  at most `budget_s` for it. If Relevan is slower, the reply uses a local match
  over the cache, and the search result still lands in the cache for later turns.
- The cache lives for the session and is dropped when the conversation resets.

What was surfaced each turn (ids, and Relevan's queryId/sessionId) is returned
so the chat log can record it; the writer later sends Relevan feedback on the
memories that actually mattered.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

from haru_gateway.relevan import Relevan, RelevanError

log = logging.getLogger("haru.gateway.memory")

_WORD = re.compile(r"[a-z0-9']+")
_STOP = frozenset("a an the and or but if of to in on at for with is are was were be been am i me my you your "
                  "it its this that what who how when where do does did can could would should will just so "
                  "hey hi haru please tell about".split())


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


@dataclass
class Recall:
    memories: list[dict]
    query_id: str | None = None
    relevan_session: str | None = None
    from_search: bool = False     # False: Relevan was too slow (or down); local match only

    @property
    def ids(self) -> list[str]:
        return [m["id"] for m in self.memories]


@dataclass
class _Session:
    cache: dict[str, dict] = field(default_factory=dict)
    loading: asyncio.Task | None = None
    relevan_session: str | None = None
    started: float = field(default_factory=time.monotonic)


def format_for_prompt(memories: list[dict]) -> str:
    if not memories:
        return ""
    lines = []
    for m in memories:
        when = (m.get("updated_at") or m.get("created_at") or "")[:10]
        person = m.get("person")
        who = f" ({person})" if person and person != "household" else ""
        lines.append(f"- {m.get('text', '').strip()}{who}" + (f" [noted {when}]" if when else ""))
    return ("What you remember from earlier conversations (may be out of date; use it only when "
            "it's relevant, and don't recite it unprompted):\n" + "\n".join(lines))


class MemoryStore:
    def __init__(self, relevan: Relevan, *, budget_s: float = 0.25, max_in_prompt: int = 10) -> None:
        self.relevan = relevan
        self.budget_s = budget_s
        self.max_in_prompt = max_in_prompt
        self.sessions: dict[str, _Session] = {}
        self._pending: set[asyncio.Task] = set()

    # --- session lifecycle ------------------------------------------------------

    def start_session(self, device_id: str) -> None:
        """Idempotent; call as early as possible (VAD speech start)."""
        s = self.sessions.get(device_id)
        if s is None:
            s = self.sessions[device_id] = _Session()
        if s.loading is None:
            s.loading = self._spawn(self._load(device_id, s))

    def end_session(self, device_id: str) -> None:
        s = self.sessions.pop(device_id, None)
        if s and s.loading and not s.loading.done():
            s.loading.cancel()

    async def _load(self, device_id: str, s: _Session) -> None:
        t0 = time.monotonic()
        try:
            docs = await self.relevan.list_documents()
        except RelevanError as exc:
            log.warning("could not preload memories for %s: %s", device_id, exc)
            return
        for d in docs:
            if d.get("status", "active") == "active":
                s.cache.setdefault(d["id"], d)
        log.info("preloaded %d memories for %s in %.0f ms", len(s.cache), device_id, (time.monotonic() - t0) * 1000)

    # --- per-turn recall -----------------------------------------------------------

    async def recall(self, device_id: str, text: str) -> Recall:
        self.start_session(device_id)
        s = self.sessions[device_id]
        search = self._spawn(self._search(s, text))
        # Wait (within one shared budget) for the search, and for the session's
        # preload if it's still running, so the local fallback isn't matching
        # against an empty cache on a session's first turn.
        waiting = {search} | ({s.loading} if s.loading and not s.loading.done() else set())
        await asyncio.wait(waiting, timeout=self.budget_s)
        hits, query_id, from_search = [], None, False    # a late search still completes into the cache
        if search.done() and not search.cancelled():
            if search.exception() is None:
                hits, query_id = search.result()
                from_search = True
            else:
                log.debug("memory search failed: %s", search.exception())
        chosen: dict[str, dict] = {h["id"]: h for h in hits}
        for m in self._local_match(s, text):
            if len(chosen) >= self.max_in_prompt:
                break
            chosen.setdefault(m["id"], m)
        memories = list(chosen.values())[: self.max_in_prompt]
        return Recall(memories=memories, query_id=query_id, relevan_session=s.relevan_session,
                      from_search=from_search)

    async def _search(self, s: _Session, text: str) -> tuple[list[dict], str | None]:
        # Content words only: a full spoken sentence ("I am feeling peckish, any snack
        # ideas for me?") scored 0 in Relevan (most terms matched nothing), while
        # "snack" alone scored 3.5.
        query = " ".join(sorted(_terms(text), key=text.lower().find))
        if not query:
            return [], None
        data = await self.relevan.search(query, size=self.max_in_prompt, session_id=s.relevan_session,
                                         filters=[{"field": "status", "value": "active"}])
        ctx = data.get("searchContext") or {}
        s.relevan_session = ctx.get("sessionId") or s.relevan_session
        hits = []
        for r in data.get("results") or []:
            doc = {"id": r["id"], **(r.get("document") or {})}
            s.cache[doc["id"]] = doc
            hits.append(doc)
        return hits, ctx.get("queryId")

    def _local_match(self, s: _Session, text: str) -> list[dict]:
        """Cheap overlap ranking over the session cache, plus the most important memories."""
        q = _terms(text)

        def score(m: dict) -> float:
            hay = _terms(" ".join([m.get("text", ""), m.get("subject", ""), " ".join(m.get("tags") or [])]))
            return len(q & hay) * 2 + float(m.get("importance") or 0) * 0.3

        ranked = sorted(s.cache.values(), key=score, reverse=True)
        return [m for m in ranked if score(m) >= 1.2]  # some overlap, or importance >= 4

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        # A search that fails after recall() stopped waiting must not warn about
        # an unretrieved exception; it's already been handled as "no results".
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        return task

    async def close(self) -> None:
        for t in list(self._pending):
            t.cancel()
