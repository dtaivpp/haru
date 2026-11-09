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

Each turn's prompt gets three tiers of memories:

- core: who the household is (the user's name, family). Always included, with
  no search, so Haru never forgets who she is talking to.
- recent: projects, plans, events and findings from the last few days, newest
  first, so "what was Capy working on?" has context without the right keywords.
- relevant: search hits for what was just said, minus weak hits far below the
  best one.

What was surfaced each turn (ids, and Relevan's queryId/sessionId) is returned
so the chat log can record it; the writer later sends Relevan feedback on the
memories that actually mattered.
"""

from __future__ import annotations

import asyncio
import datetime as dt
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


MAX_CORE = 8
CORE_KINDS = frozenset({"person"})          # family and other people in the user's life
RECENT_DAYS = 7
RECENT_KINDS = frozenset({"project", "plan", "event", "finding"})
MAX_RECENT = 4
MAX_RELEVANT = 5
RELATIVE_CUTOFF = 0.3    # drop search hits that score under 30% of the best hit


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


@dataclass
class Recall:
    core: list[dict] = field(default_factory=list)       # always in the prompt
    recent: list[dict] = field(default_factory=list)     # the last few days
    relevant: list[dict] = field(default_factory=list)   # search hits first, then local matches
    query_id: str | None = None
    relevan_session: str | None = None
    from_search: bool = False     # False: Relevan was too slow (or down); local match only
    search_ids: list[str] = field(default_factory=list)  # the search's hits in its order, for feedback

    @property
    def memories(self) -> list[dict]:
        return self.core + self.recent + self.relevant

    @property
    def ids(self) -> list[str]:
        return [m["id"] for m in self.memories]


@dataclass
class _Session:
    cache: dict[str, dict] = field(default_factory=dict)
    loading: asyncio.Task | None = None
    relevan_session: str | None = None
    started: float = field(default_factory=time.monotonic)


def is_core(m: dict) -> bool:
    return float(m.get("importance") or 0) >= 5 or m.get("kind") in CORE_KINDS


def _when(m: dict) -> str:
    return (m.get("updated_at") or m.get("created_at") or "")[:10]


def _line(m: dict, dated: bool = True) -> str:
    person = m.get("person")
    who = f" ({person})" if person and person != "household" else ""
    when = _when(m) if dated else ""
    return f"- {m.get('text', '').strip()}{who}" + (f" [noted {when}]" if when else "")


def format_for_prompt(recall: Recall) -> str:
    parts = []
    if recall.core:
        parts.append(
            "Who you live with (from your memory):\n" + "\n".join(_line(m, dated=False) for m in recall.core)
            + "\nThe person talking to you is almost always the user described here. Other people in the "
              "household, such as the user's children, sometimes chime in. Treat the speaker as the user "
              "unless what they say shows it's someone else (they give another name, talk about the user as "
              "someone else, or sound like a child talking). Then reply to that person, and don't call them by "
              "the user's name. If you can't tell and it matters, ask.")
    if recall.recent:
        parts.append("Recent context from the last week, newest first (use it when the user picks up where "
                     "they left off):\n" + "\n".join(_line(m) for m in recall.recent))
    if recall.relevant:
        parts.append("Other things you remember that may relate to what was just said (may be out of date; use "
                     "them only when they're relevant, and don't recite them unprompted):\n"
                     + "\n".join(_line(m) for m in recall.relevant))
    return "\n\n".join(parts)


def _when_full(m: dict) -> str:
    return (m.get("updated_at") or m.get("created_at") or "")[:19]


class MemoryStore:
    def __init__(self, relevan: Relevan, *, budget_s: float = 0.25, max_relevant: int = MAX_RELEVANT) -> None:
        self.relevan = relevan
        self.budget_s = budget_s
        self.max_relevant = max_relevant
        self.sessions: dict[str, _Session] = {}
        # The last full list of memories. A new session starts from it, so the core
        # memories are there on the first turn even before that session's preload ends.
        self.known: dict[str, dict] = {}
        self._pending: set[asyncio.Task] = set()

    # --- session lifecycle ------------------------------------------------------

    def start_session(self, device_id: str) -> None:
        """Idempotent; call as early as possible (VAD speech start)."""
        s = self.sessions.get(device_id)
        if s is None:
            s = self.sessions[device_id] = _Session(cache=dict(self.known))
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
        active = {d["id"]: d for d in docs if d.get("status", "active") == "active"}
        self.known = dict(active)
        for doc_id in list(s.cache):
            if doc_id not in active:
                del s.cache[doc_id]       # retired or deleted since the last session
        s.cache.update(active)
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
        core = sorted((m for m in s.cache.values() if is_core(m)),     # people first: who's who matters most
                      key=lambda m: (m.get("kind") in CORE_KINDS, float(m.get("importance") or 0)),
                      reverse=True)[:MAX_CORE]
        taken = {m["id"] for m in core}
        recent = [m for m in self._recent(s) if m["id"] not in taken]
        taken |= {m["id"] for m in recent}
        relevant = [h for h in hits if h["id"] not in taken][: self.max_relevant]
        taken |= {m["id"] for m in relevant}
        for m in self._local_match(s, text):
            if len(relevant) >= self.max_relevant:
                break
            if m["id"] not in taken:
                relevant.append(m)
                taken.add(m["id"])
        return Recall(core=core, recent=recent, relevant=relevant, query_id=query_id,
                      relevan_session=s.relevan_session, from_search=from_search, search_ids=[h["id"] for h in hits])

    def _recent(self, s: _Session) -> list[dict]:
        cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=RECENT_DAYS)).isoformat()[:19]
        fresh = [m for m in s.cache.values() if m.get("kind") in RECENT_KINDS and _when_full(m) >= cutoff]
        return sorted(fresh, key=_when_full, reverse=True)[:MAX_RECENT]

    async def _search(self, s: _Session, text: str) -> tuple[list[dict], str | None]:
        # Content words only: a full spoken sentence ("I am feeling peckish, any snack
        # ideas for me?") scored 0 in Relevan (most terms matched nothing), while
        # "snack" alone scored 3.5.
        query = " ".join(sorted(_terms(text), key=text.lower().find))
        if not query:
            return [], None
        data = await self.relevan.search(query, size=self.max_relevant + MAX_CORE, session_id=s.relevan_session,
                                         filters=[{"field": "status", "value": "active"}])
        ctx = data.get("searchContext") or {}
        s.relevan_session = ctx.get("sessionId") or s.relevan_session
        results = data.get("results") or []
        # Weak hits share one stray word with the turn ("open" in "open notifications"
        # matching a coffee shop that is "open now"). Keep the hits near the best one.
        top = max((float(r.get("score") or 0) for r in results), default=0.0)
        hits = []
        for r in results:
            doc = {"id": r["id"], **(r.get("document") or {})}
            s.cache[doc["id"]] = doc
            if float(r.get("score") or 0) >= top * RELATIVE_CUTOFF:
                hits.append(doc)
        return hits, ctx.get("queryId")

    def _local_match(self, s: _Session, text: str) -> list[dict]:
        """Cheap overlap ranking over the session cache (the core memories are already in the prompt)."""
        q = _terms(text)

        def overlap(m: dict) -> int:
            return len(q & _terms(" ".join([m.get("text", ""), m.get("subject", ""), " ".join(m.get("tags") or [])])))

        def score(m: dict) -> float:
            return overlap(m) * 2 + float(m.get("importance") or 0) * 0.1

        return sorted((m for m in s.cache.values() if overlap(m) > 0), key=score, reverse=True)

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
