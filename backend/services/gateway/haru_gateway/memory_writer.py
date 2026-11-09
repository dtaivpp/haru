"""Haru's long-term memory, write side: mines the chat log into Relevan.

Runs as its own process (the `memory` container), separate from the gateway,
so a slow or failing extraction can never touch a live conversation:

    python -m haru_gateway.memory_writer

Every POLL_S it reads recent chat-log files, groups turns by session, and
processes a session once it has gone quiet (IDLE_S) or has piled up MAX_TURNS
unprocessed turns. For each batch, gpt-oss reads the new turns (with a little
earlier context) plus the existing memories most related to them, and returns
JSON operations:

    {"add":     [{"text", "subject", "kind", "tags", "importance", "person"}],
     "update":  [{"id", "text", ...changed fields}],
     "retire":  [{"id", "reason"}],
     "relevant": ["mem-..."]}      # surfaced memories that actually mattered

Those become Relevan document writes, plus `llm_judgment` feedback (and
use_count / last_used_at bumps) for the memories that mattered. Progress is
checkpointed per session, so restarts never redo or skip turns.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import re
import signal
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx

from haru_gateway.chatlog import utcnow_iso
from haru_gateway.llm import USER_AGENT, LLMConfig
from haru_gateway.memory import _terms
from haru_gateway.relevan import Relevan, RelevanConfig, RelevanError

log = logging.getLogger("haru.memory_writer")

KINDS = ("fact", "preference", "person", "event", "plan", "project", "interest", "finding")
CONTEXT_TURNS = 6          # earlier turns shown for context, not mined again
MAX_EXISTING = 40          # existing memories shown to the model for dedup/updates

SYSTEM_PROMPT = """\
You maintain the long-term memory of Haru, a voice personal assistant on the user's desk.
Read the conversation excerpt and decide what is worth remembering about the user and their
household for future conversations.

Haru is a personal assistant, so remember what a good human assistant would jot down:
- facts about the user and their life: their name, family, relationships, pets, home, work
  (answers to questions Haru asked, like their name, matter most)
- preferences and dislikes, routines, and habits
- important dates, and upcoming events or plans
- projects: what they're working on or building, and their goals and commitments
- interests: what they're researching, evaluating, comparing or considering buying/joining.
  A lookup request usually reveals this: "look up what the Google Developer Program includes" means
  "The user is evaluating the Google Developer Program." Record the topic and why it seems to matter.
- key findings Haru reported that the user may come back to (the specific answer, briefly:
  "Haru found the Google Developer Program premium tier includes about $500 a year in Cloud credits.")
- anything they explicitly ask Haru to remember

Do NOT remember:
- small talk, greetings, and one-off trivia with no bearing on the user's life
  ("what's the capital of France", "who won the race" asked in passing)
- anything Haru said about herself, or Haru's own suggestions the user didn't adopt
- passing states ("I'm hungry", "it's raining")
- garbled speech-recognition fragments, or speech that sounds like TV, radio or other people
  not talking to Haru (the microphone is always on)
- notifications Haru read out or the user cleared, and status checks (CI runs, open PRs, task
  updates). They change by the hour. Remember the project behind them only if it's new.
- things that are already in memory and unchanged

Who is speaking: the microphone hears the whole household. Almost every turn is the user, but
their children or guests sometimes chime in. Never record what a child or guest says as a fact
about the user. If someone else clearly says something worth keeping about themselves, write it
about them by name ("The user's daughter Emma is learning to swim.").

Projects, interests and findings are time-sensitive: put the date in the sentence ("As of
September 30, 2026, the user is evaluating..."). The current date is given below.

Write each memory as one short, standalone sentence in the third person about "the user",
for example "The user's wife's birthday is June 3." Keep proper names exactly as heard.
"subject" is the topic or thing the memory is about, in a few searchable words: "wife birthday",
"favorite snack", "dog Biscuit", "work schedule". It is never "user" or "the user"; every memory is
about the user, so that word says nothing.
If a new fact changes or adds to an existing memory, UPDATE that memory by id with the complete,
corrected sentence ("The user's children are Emma, Micah and Matthew."), never a second memory
beside it. If one is no longer true, or is a finding the user has since dealt with, RETIRE it.
If the user asks Haru to forget or remove something, RETIRE the matching memory.
Importance: 5 = core identity: the user's name, and the names of their partner, children and pets
(these are in every conversation, so keep them to who the household is),
4 = important dates, plans and active projects,
3 = preferences and current interests, 2 = findings and minor details, 1 = trivia. Extract everything
useful; a typical session yields zero to four memories. Returning nothing is correct only when nothing
in the new turns says anything about the user.

"relevant" lists the ids of the EXISTING memories that actually helped Haru in these turns.

Reply with JSON only, no prose:
{"add": [{"text": "...", "subject": "...", "kind": "fact|preference|person|event|plan|project|interest|finding",
          "tags": ["..."], "importance": 1-5}],
 "update": [{"id": "mem-...", "text": "...", "importance": 1-5}],
 "retire": [{"id": "mem-...", "reason": "..."}],
 "relevant": ["mem-..."]}
"""


@dataclass
class WriterConfig:
    chatlog_dir: Path
    state_path: Path
    idle_s: float = 120.0
    max_turns: int = 20
    poll_s: float = 20.0
    reasoning_effort: str = "medium"
    days: int = 3                  # how many days of log files to scan

    @classmethod
    def from_env(cls) -> WriterConfig:
        e = os.environ.get
        return cls(
            chatlog_dir=Path(e("HARU_CHATLOG_DIR", "/data/conversations")),
            state_path=Path(e("HARU_MEMORY_STATE", "/data/memory/writer-state.json")),
            idle_s=float(e("HARU_MEMORY_IDLE_S", "120")),
            max_turns=int(e("HARU_MEMORY_MAX_TURNS", "20")),
            poll_s=float(e("HARU_MEMORY_POLL_S", "20")),
            reasoning_effort=e("HARU_MEMORY_REASONING_EFFORT", "medium"),
        )


def _parse_ts(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _extract_json(text: str) -> dict:
    """The model's reply as JSON, tolerating code fences or stray prose around it."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def _clean_memory(m: dict) -> dict | None:
    text = str(m.get("text", "")).strip()
    if not text or len(text) > 400:
        return None
    kind = m.get("kind") if m.get("kind") in KINDS else "fact"
    try:
        importance = min(5, max(1, int(m.get("importance", 3))))
    except (TypeError, ValueError):
        importance = 3
    tags = [str(t).strip().lower()[:40] for t in (m.get("tags") or []) if str(t).strip()][:8]
    subject = str(m.get("subject") or "").strip()[:120]
    if subject.lower() in ("user", "the user", "household", ""):
        subject = " ".join(tags[:3])   # the model ignored the instruction; tags are a better topic
    return {"text": text, "subject": subject, "kind": kind,
            "tags": tags, "importance": importance, "person": str(m.get("person") or "household")[:60]}


class MemoryWriter:
    def __init__(self, config: WriterConfig, relevan: Relevan, llm: LLMConfig) -> None:
        self.config = config
        self.relevan = relevan
        self.llm = llm
        self.state: dict[str, int] = self._load_state()     # session -> entries processed
        self.http = httpx.AsyncClient(timeout=120, headers={"Authorization": f"Bearer {llm.api_key}",
                                                            "User-Agent": USER_AGENT})

    # --- state -------------------------------------------------------------------------

    def _load_state(self) -> dict[str, int]:
        try:
            return json.loads(self.config.state_path.read_text())
        except (OSError, ValueError):
            return {}

    def _save_state(self) -> None:
        self.config.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.config.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.config.state_path)

    # --- chat log ------------------------------------------------------------------------

    def _sessions(self) -> dict[str, list[dict]]:
        today = dt.date.today()
        wanted = {str(today - dt.timedelta(days=d)) for d in range(self.config.days + 1)}
        sessions: dict[str, list[dict]] = {}
        for path in sorted(self.config.chatlog_dir.glob("*/*.jsonl")):
            if path.stem not in wanted:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                sessions.setdefault(e.get("session", "?"), []).append(e)
        return sessions

    def _due(self, entries: list[dict], done: int) -> bool:
        new = len(entries) - done
        if new <= 0:
            return False
        idle = (dt.datetime.now(dt.timezone.utc) - _parse_ts(entries[-1]["ts"])).total_seconds()
        return idle >= self.config.idle_s or new >= self.config.max_turns

    # --- one pass ----------------------------------------------------------------------------

    async def run_once(self) -> int:
        """Process every session that's due. Returns how many were processed."""
        processed = 0
        for session, entries in self._sessions().items():
            done = self.state.get(session, 0)
            if not self._due(entries, done):
                continue
            try:
                await self._process(session, entries, done)
            except (RelevanError, httpx.HTTPError, ValueError) as exc:
                log.warning("session %s: extraction failed, will retry: %s", session[:8], exc)
                continue
            self.state[session] = len(entries)
            self._save_state()
            processed += 1
        return processed

    async def _process(self, session: str, entries: list[dict], done: int) -> None:
        new = entries[done:]
        context = entries[max(0, done - CONTEXT_TURNS):done]
        if not any(e["role"] == "user" for e in new):
            return
        existing = await self._existing_for(new)
        reply = await self._ask_llm(context, new, existing)
        ops = _extract_json(reply)
        existing_ids = {m["id"] for m in existing}
        now = utcnow_iso()

        adds = []
        for m in ops.get("add") or []:
            if (clean := _clean_memory(m)) is not None:
                adds.append({"id": f"mem-{uuid.uuid4().hex[:16]}", **clean, "status": "active", "use_count": 0,
                             "created_at": now, "updated_at": now, "source": session})
        if adds:
            await self.relevan.ingest(adds)
        by_id = {m["id"]: m for m in existing}
        updates = 0
        for u in ops.get("update") or []:
            if u.get("id") in existing_ids and (clean := _clean_memory({**by_id[u["id"]], **u})) is not None:
                await self.relevan.update(u["id"], {**_content(by_id[u["id"]]), **clean, "updated_at": now})
                updates += 1
        retires = 0
        for r in ops.get("retire") or []:
            if r.get("id") in existing_ids:
                await self.relevan.update(r["id"], {**_content(by_id[r["id"]]), "status": "retired",
                                                    "updated_at": now})
                retires += 1
        relevant = [i for i in (ops.get("relevant") or []) if i in existing_ids]
        await self._feedback(new, relevant, by_id, now)
        log.info("session %s: %d new turns -> +%d memories, %d updated, %d retired, %d relevant%s",
                 session[:8], len(new), len(adds), updates, retires, len(relevant),
                 "".join(f"\n  + {a['text']} [{a['kind']}, {a['importance']}]" for a in adds))

    async def _existing_for(self, new: list[dict]) -> list[dict]:
        """Active memories most related to these turns (plus everything they surfaced)."""
        docs = [d for d in await self.relevan.list_documents() if d.get("status", "active") == "active"]
        surfaced = {i for e in new for i in (e.get("memories") or [])}
        q = _terms(" ".join(e["text"] for e in new))

        def score(d: dict) -> float:
            hay = _terms(" ".join([d.get("text", ""), d.get("subject", ""), " ".join(d.get("tags") or [])]))
            return (100 if d["id"] in surfaced else 0) + len(q & hay) * 2 + float(d.get("importance") or 0) * 0.1

        return sorted(docs, key=score, reverse=True)[:MAX_EXISTING]

    async def _ask_llm(self, context: list[dict], new: list[dict], existing: list[dict]) -> str:
        def fmt(es):
            return "\n".join(f"{'User' if e['role'] == 'user' else 'Haru'}: {e['text']}" for e in es)
        mem = "\n".join(f"{m['id']}: {m.get('text')} (importance {m.get('importance')})" for m in existing) or "(none)"
        user = (f"Current date: {dt.date.today():%B %-d, %Y}\n\nExisting memories:\n{mem}\n\n"
                + (f"Earlier context (already processed, don't extract from it):\n{fmt(context)}\n\n" if context else "")
                + f"New conversation turns to process:\n{fmt(new)}")
        body = {"model": self.llm.model, "stream": False, "max_tokens": 2000,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]}
        if self.config.reasoning_effort:
            body["reasoning_effort"] = self.config.reasoning_effort
        r = await self.http.post(f"{self.llm.base_url}/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content") or "{}"

    async def _feedback(self, new: list[dict], relevant: list[str], by_id: dict[str, dict], now: str) -> None:
        for mem_id in relevant:
            # Credit every search in these turns that surfaced this memory.
            for e in new:
                ids = e.get("memories") or []
                ctx = e.get("relevan") or {}
                # The position is the memory's place in the search results. A memory that was
                # in the prompt but not a hit (a core or recent memory) gets no feedback.
                hits = ctx.get("hits", ids)      # older log lines have no "hits": surfaced ids were hits first
                if mem_id in ids and mem_id in hits and ctx.get("queryId"):
                    await self.relevan.feedback(action="llm_judgment", object_id=mem_id, query_id=ctx["queryId"],
                                                session_id=ctx.get("sessionId"), position=hits.index(mem_id))
            doc = by_id[mem_id]
            await self.relevan.update(mem_id, {**_content(doc), "use_count": int(doc.get("use_count") or 0) + 1,
                                               "last_used_at": now})

    async def run_forever(self) -> None:
        log.info("memory writer watching %s (idle %.0fs, batch %d turns) -> relevan %s/%s",
                 self.config.chatlog_dir, self.config.idle_s, self.config.max_turns,
                 self.relevan.config.org, self.relevan.config.index)
        while True:
            try:
                await self.run_once()
            except Exception:
                log.exception("memory writer pass failed")
            await asyncio.sleep(self.config.poll_s)

    async def close(self) -> None:
        await self.http.aclose()
        await self.relevan.close()


def _content(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k != "id"}


def main() -> None:
    logging.basicConfig(level=os.environ.get("HARU_LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rel_cfg, llm_cfg = RelevanConfig.from_env(), LLMConfig.from_env()
    if rel_cfg is None or llm_cfg is None:
        raise SystemExit("memory writer needs RELEVAN_API_KEY, RELEVAN_ORG and the HARU_LLM_* settings")
    writer = MemoryWriter(WriterConfig.from_env(), Relevan(rel_cfg), llm_cfg)

    async def run() -> None:
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(writer.run_forever())
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, task.cancel)
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            await writer.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
