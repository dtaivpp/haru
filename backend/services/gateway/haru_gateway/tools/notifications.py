"""Notifications: things that happened while you weren't asking (a Capy task finishing...).

Unlike timers, a notification is never spoken unprompted. It shows up on the
face page as a bell with an unread count, and stays there until the user asks
Haru to open their notifications. Then the list opens on screen, Haru sums it
up, and the opened items are marked read. Haru stays quiet in a meeting, and nothing gets lost.

A notification belongs to one terminal (device_id) or, with device_id None, to
every terminal. Read state is shared: opening them anywhere marks them read.
They're saved to a JSON file on every change, so a restart keeps them.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

from haru_gateway.tools import Tool, ToolContext, ToolError

log = logging.getLogger("haru.gateway.notifications")

KEEP = 50                          # newest notifications kept, read or not
READ_RETENTION = dt.timedelta(days=7)

OnBadge = Callable[[str | None], Awaitable[None]]           # device_id (None: every device) whose count changed
OnPanel = Callable[[str, list[dict], bool], Awaitable[None]]  # (device, items, open): show or close the list


@dataclass
class Notification:
    id: str
    title: str                     # short, for the panel and for Haru to say: "Capy finished: fix CI timeout"
    body: str = ""                 # detail Haru can summarise when asked
    source: str = "haru"           # "capy", "debug", ...
    device_id: str | None = None   # None: every terminal
    created: str = ""              # ISO 8601 with offset
    read: bool = False
    ref: dict = field(default_factory=dict)   # source-specific handle, e.g. {"thread_id": ..., "url": ...}


def _ago(then: dt.datetime, now: dt.datetime) -> str:
    s = max(0, int((now - then).total_seconds()))
    if s < 60:
        return "just now"
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if s >= size:
            n = s // size
            return f"{n} {unit}{'s' * (n != 1)} ago"
    return "just now"


class NotificationService:
    def __init__(self, path: Path, tz: ZoneInfo) -> None:
        self.path = path
        self.tz = tz
        self.items: dict[str, Notification] = {}
        self._on_badge: list[OnBadge] = []
        self._on_panel: list[OnPanel] = []
        self._load()

    def on_badge(self, cb: OnBadge) -> None:
        self._on_badge.append(cb)

    def on_panel(self, cb: OnPanel) -> None:
        self._on_panel.append(cb)

    def now(self) -> dt.datetime:
        return dt.datetime.now(self.tz)

    # --- persistence -------------------------------------------------------------------

    def _load(self) -> None:
        try:
            self.items = {n["id"]: Notification(**n) for n in json.loads(self.path.read_text())}
        except (OSError, ValueError, TypeError):
            self.items = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps([asdict(n) for n in self.items.values()], indent=1))
            tmp.replace(self.path)
        except OSError as exc:
            log.warning("could not save notifications: %s", exc)

    def _prune(self) -> None:
        now = self.now()
        keep = [n for n in self.items.values()
                if not (n.read and now - dt.datetime.fromisoformat(n.created) > READ_RETENTION)]
        keep.sort(key=lambda n: n.created)
        self.items = {n.id: n for n in keep[-KEEP:]}

    # --- queries -------------------------------------------------------------------------

    def for_device(self, device_id: str, *, unread_only: bool = False) -> list[Notification]:
        """Newest first."""
        mine = [n for n in self.items.values() if n.device_id in (None, device_id) and not (unread_only and n.read)]
        return sorted(mine, key=lambda n: n.created, reverse=True)

    def unread_count(self, device_id: str) -> int:
        return len(self.for_device(device_id, unread_only=True))

    def describe(self, n: Notification) -> dict:
        d = {"id": n.id, "title": n.title, "source": n.source,
             "when": _ago(dt.datetime.fromisoformat(n.created), self.now()), "read": n.read}
        if n.body:
            d["body"] = n.body
        if n.ref:
            d["ref"] = n.ref     # for follow-ups, e.g. {"capy_thread_id": ...} -> capy_task
        return d

    # --- changes ---------------------------------------------------------------------------

    async def add(self, title: str, body: str = "", *, source: str = "haru", device_id: str | None = None,
                  ref: dict | None = None) -> Notification:
        n = Notification(id=uuid.uuid4().hex[:8], title=title.strip(), body=body.strip(), source=source,
                         device_id=device_id, created=self.now().isoformat(), ref=ref or {})
        self.items[n.id] = n
        self._prune()
        self._save()
        log.info("notification %s for %s from %s: %r", n.id, device_id or "all devices", source, n.title)
        await self._badge(device_id)
        return n

    async def open(self, device_id: str, *, include_read: bool = False, limit: int = 10) -> list[dict]:
        """Show the device's notifications on its screen and mark the shown ones read."""
        items = self.for_device(device_id, unread_only=not include_read)[:limit]
        shown = [self.describe(n) for n in items]   # before marking, so "read" says what it was
        for n in items:
            n.read = True
        if items:
            self._save()
        for cb in self._on_panel:
            await cb(device_id, shown, True)
        # Read state is shared, so every screen's count may have changed, not just this one's.
        await self._badge(None)
        return shown

    async def clear(self, device_id: str, ids: list[str] | None = None) -> int:
        """Delete the given notifications, or all of this device's read ones."""
        mine = {n.id for n in self.for_device(device_id)}
        doomed = [i for i in (ids or []) if i in mine] if ids else \
            [n.id for n in self.for_device(device_id) if n.read]
        for i in doomed:
            del self.items[i]
        if doomed:
            self._save()
            await self._badge(None)
        for cb in self._on_panel:   # what the panel showed may be gone: close it
            await cb(device_id, [], False)
        return len(doomed)

    async def _badge(self, device_id: str | None) -> None:
        for cb in self._on_badge:
            try:
                await cb(device_id)
            except Exception:
                log.exception("notification badge update failed")

    # --- tools -------------------------------------------------------------------------------

    def tools(self) -> list[Tool]:
        async def open_notifications(args: dict, ctx: ToolContext) -> dict:
            shown = await self.open(ctx.device_id, include_read=bool(args.get("include_read")))
            if not shown:
                return {"notifications": [], "note": "nothing new"}
            return {"notifications": shown, "note": "now shown on screen and marked read"}

        async def clear_notifications(args: dict, ctx: ToolContext) -> dict:
            ids = args.get("ids")
            if ids is not None and not all(isinstance(i, str) for i in ids):
                raise ToolError("ids must be a list of notification ids")
            n = await self.clear(ctx.device_id, ids)
            return {"cleared": n, "unread_left": self.unread_count(ctx.device_id)}

        return [
            Tool("open_notifications",
                 "Open the user's notifications (e.g. coding tasks that finished) on the screen and get them, "
                 "newest first. Use when the user asks about notifications, alerts, or what's new or done. "
                 "Then sum them up in a sentence or two; don't read every detail out.",
                 {"type": "object", "properties": {
                     "include_read": {"type": "boolean",
                                      "description": "Also show ones already opened (default: unread only)"},
                 }}, open_notifications),
            Tool("clear_notifications",
                 "Delete notifications: the given ids, or with no ids, every notification already opened.",
                 {"type": "object", "properties": {
                     "ids": {"type": "array", "description": "Notification ids to delete"},
                 }}, clear_notifications),
        ]
