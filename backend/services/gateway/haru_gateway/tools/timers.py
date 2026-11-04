"""Timers and reminders. Haru announces them out loud on the terminal when due.

They're saved to a JSON file on every change, so a gateway restart doesn't
lose them. One that fell due while the gateway was down is announced as soon
as it's back ("late"), not silently dropped.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

from haru_gateway.tools import Tool, ToolContext, ToolError

log = logging.getLogger("haru.gateway.timers")

Announce = Callable[[str, str], Awaitable[None]]   # (device_id, what to say)


@dataclass
class Alarm:
    id: str
    device_id: str
    kind: str          # "timer" | "reminder"
    due: str           # ISO 8601 with offset
    label: str
    created: str


def _human_delta(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = [f"{h} hour{'s' * (h != 1)}" if h else "", f"{m} minute{'s' * (m != 1)}" if m else "",
             f"{s} second{'s' * (s != 1)}" if s and not h else ""]
    return " ".join(p for p in parts if p) or "0 seconds"


class TimerService:
    def __init__(self, path: Path, tz: ZoneInfo, announce: Announce) -> None:
        self.path = path
        self.tz = tz
        self.announce = announce
        self.alarms: dict[str, Alarm] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._load()

    def now(self) -> dt.datetime:
        return dt.datetime.now(self.tz)

    # --- persistence ---------------------------------------------------------------

    def _load(self) -> None:
        try:
            self.alarms = {a["id"]: Alarm(**a) for a in json.loads(self.path.read_text())}
        except (OSError, ValueError, TypeError):
            self.alarms = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps([asdict(a) for a in self.alarms.values()], indent=1))
            tmp.replace(self.path)
        except OSError as exc:
            log.warning("could not save timers: %s", exc)

    def start(self) -> None:
        """Schedule everything loaded from disk (call once the event loop is running)."""
        for a in list(self.alarms.values()):
            self._schedule(a)

    # --- scheduling -----------------------------------------------------------------------

    def add(self, device_id: str, kind: str, due: dt.datetime, label: str) -> Alarm:
        a = Alarm(id=uuid.uuid4().hex[:8], device_id=device_id, kind=kind, due=due.isoformat(timespec="seconds"),
                  label=label.strip(), created=self.now().isoformat(timespec="seconds"))
        self.alarms[a.id] = a
        self._save()
        self._schedule(a)
        log.info("%s %s for %s at %s: %r", kind, a.id, device_id, a.due, a.label)
        return a

    def cancel(self, alarm_id: str) -> Alarm | None:
        a = self.alarms.pop(alarm_id, None)
        if a:
            if (t := self._tasks.pop(alarm_id, None)) and t is not asyncio.current_task():
                t.cancel()
            self._save()
        return a

    def for_device(self, device_id: str) -> list[Alarm]:
        return sorted((a for a in self.alarms.values() if a.device_id == device_id), key=lambda a: a.due)

    def _schedule(self, a: Alarm) -> None:
        self._tasks[a.id] = asyncio.create_task(self._fire_at(a))

    async def _fire_at(self, a: Alarm) -> None:
        delay = (dt.datetime.fromisoformat(a.due) - self.now()).total_seconds()
        late = delay < -60
        await asyncio.sleep(max(0.0, delay))
        if a.kind == "timer":
            what = f"your {a.label} timer" if a.label else "your timer"
            text = f"Time's up! That's {what}."
        else:
            text = f"Reminder: {a.label}." if a.label else "This is your reminder."
        if late:
            text = f"Sorry, this one's late. {text}"
        self.cancel(a.id)
        log.info("%s %s due: announcing on %s", a.kind, a.id, a.device_id)
        try:
            await self.announce(a.device_id, text)
        except Exception:
            log.exception("announcing %s failed", a.id)

    def close(self) -> None:
        for t in self._tasks.values():
            t.cancel()

    # --- tools ---------------------------------------------------------------------------

    def tools(self) -> list[Tool]:
        def describe(a: Alarm) -> dict:
            due = dt.datetime.fromisoformat(a.due)
            return {"id": a.id, "kind": a.kind, "label": a.label,
                    "due": due.strftime("%A %-I:%M %p"), "in": _human_delta((due - self.now()).total_seconds())}

        async def set_timer(args: dict, ctx: ToolContext) -> dict:
            secs = int(args["duration_seconds"])
            a = self.add(ctx.device_id, "timer", self.now() + dt.timedelta(seconds=secs), args.get("label", ""))
            return {"set": describe(a), "duration": _human_delta(secs)}

        async def set_reminder(args: dict, ctx: ToolContext) -> dict:
            if "in_minutes" in args:
                due = self.now() + dt.timedelta(minutes=float(args["in_minutes"]))
            elif "at" in args:
                try:
                    due = dt.datetime.fromisoformat(args["at"])
                except ValueError as exc:
                    raise ToolError(f"'at' must be an ISO time like 2026-09-30T17:30, got {args['at']!r}") from exc
                if due.tzinfo is None:
                    due = due.replace(tzinfo=self.tz)
                if due <= self.now():
                    raise ToolError(f"{due:%A %-I:%M %p} has already passed; pick a future time")
            else:
                raise ToolError("give either in_minutes or at")
            a = self.add(ctx.device_id, "reminder", due, args["message"])
            return {"set": describe(a)}

        async def list_timers(args: dict, ctx: ToolContext) -> dict:
            items = [describe(a) for a in self.for_device(ctx.device_id)]
            return {"timers_and_reminders": items} if items else {"timers_and_reminders": [], "note": "none set"}

        async def cancel_timer(args: dict, ctx: ToolContext) -> dict:
            mine = self.for_device(ctx.device_id)
            target = args.get("id")
            if not target and args.get("label"):
                matches = [a for a in mine if args["label"].lower() in a.label.lower()]
                if len(matches) > 1:
                    raise ToolError(f"several match {args['label']!r}: {[describe(a) for a in matches]}; pass an id")
                target = matches[0].id if matches else None
            if not target and len(mine) == 1:
                target = mine[0].id
            a = self.cancel(target) if target else None
            if a is None:
                raise ToolError(f"no such timer or reminder; current ones: {[describe(x) for x in mine]}")
            return {"cancelled": {"kind": a.kind, "label": a.label}}

        return [
            Tool("set_timer", "Start a countdown timer that you announce out loud when it ends.",
                 {"type": "object", "properties": {
                     "duration_seconds": {"type": "integer", "minimum": 1, "maximum": 86400,
                                          "description": "Timer length in seconds (10 minutes = 600)"},
                     "label": {"type": "string", "description": "What it's for, e.g. \"pasta\""},
                 }, "required": ["duration_seconds"]}, set_timer),
            Tool("set_reminder", "Remind the user of something at a later time; you'll say it out loud then. "
                 "Use in_minutes for relative times, or at for a clock time.",
                 {"type": "object", "properties": {
                     "message": {"type": "string", "description": "What to remind them, e.g. \"call the dentist\""},
                     "in_minutes": {"type": "number", "minimum": 0.1, "maximum": 10080},
                     "at": {"type": "string", "description": "Local ISO time, e.g. 2026-09-30T17:30"},
                 }, "required": ["message"]}, set_reminder),
            Tool("list_timers", "List the timers and reminders currently set.",
                 {"type": "object", "properties": {}}, list_timers),
            Tool("cancel_timer", "Cancel a timer or reminder, by id or by (part of) its label.",
                 {"type": "object", "properties": {"id": {"type": "string"}, "label": {"type": "string"}}},
                 cancel_timer),
        ]
