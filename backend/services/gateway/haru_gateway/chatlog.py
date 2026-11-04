"""Persistent chat log: one JSONL file per terminal per day.

    /data/conversations/<device_id>/<YYYY-MM-DD>.jsonl

Each line is one turn:
    {"ts": "...Z", "device": "living-room", "session": "<uuid>", "role": "user" | "assistant",
     "text": "...", "emotion": "happy",                        (assistant)
     "memories": ["mem-..."], "relevan": {"queryId", "sessionId"}}  (user: what recall surfaced)

This is the source of truth the memory writer mines, and a record of what Haru
said. Writes are small appends on the event loop (well under a millisecond).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path

log = logging.getLogger("haru.gateway.chatlog")


def utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class ChatLog:
    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def from_env(cls) -> ChatLog | None:
        root = os.environ.get("HARU_CHATLOG_DIR", "")
        return cls(Path(root)) if root else None

    def append(self, device_id: str, session: str, role: str, text: str, **extra) -> None:
        entry = {"ts": utcnow_iso(), "device": device_id, "session": session, "role": role, "text": text,
                 **{k: v for k, v in extra.items() if v not in (None, [], {})}}
        path = self.root / device_id / f"{entry['ts'][:10]}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("chat log write failed (%s): %s", path, exc)
