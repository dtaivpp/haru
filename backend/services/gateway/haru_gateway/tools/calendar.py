"""Google Calendar (read-only) through a service account.

Haru's service account sees only the calendars that were explicitly shared
with it (Google Calendar -> Settings -> Share with specific people), at the
permission chosen there. Access is revoked by removing it from that sharing
list. Configured calendars are friendly names -> calendar ids:

    HARU_CALENDARS=mine=you@example.com,family=abc123@group.calendar.google.com
    HARU_GOOGLE_SA_FILE=/config/google-service-account.json
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
from zoneinfo import ZoneInfo

import httpx

from haru_gateway.tools import Tool, ToolContext, ToolError

log = logging.getLogger("haru.gateway.calendar")

API = os.environ.get("HARU_GOOGLE_CALENDAR_API", "https://www.googleapis.com/calendar/v3")
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
MAX_EVENTS = 25


def _parse_calendars(raw: str) -> dict[str, str]:
    out = {}
    for part in (p.strip() for p in raw.split(",") if p.strip()):
        name, _, cal_id = part.partition("=")
        if cal_id:
            out[name.strip().lower()] = cal_id.strip()
    return out


class _Token:
    """Service-account access token, refreshed (in a thread) shortly before expiry."""

    def __init__(self, sa_file: str) -> None:
        from google.oauth2 import service_account
        self.creds = service_account.Credentials.from_service_account_file(sa_file, scopes=SCOPES)
        self.lock = asyncio.Lock()

    async def get(self) -> str:
        async with self.lock:
            if not self.creds.valid or (self.creds.expiry and
                                        self.creds.expiry - dt.datetime.utcnow() < dt.timedelta(minutes=2)):
                from google.auth.transport.requests import Request
                await asyncio.to_thread(self.creds.refresh, Request())
            return self.creds.token


def calendar_tools(tz: ZoneInfo) -> list[Tool]:
    sa_file = os.environ.get("HARU_GOOGLE_SA_FILE", "")
    calendars = _parse_calendars(os.environ.get("HARU_CALENDARS", ""))
    if not sa_file or not calendars:
        return []
    if not os.path.isfile(sa_file):
        log.warning("calendar tools disabled: %s not found", sa_file)
        return []
    token = _Token(sa_file)
    client = httpx.AsyncClient(timeout=15, headers={"User-Agent": "haru-gateway/0.1"})

    async def api(method: str, path: str, **kw) -> dict:
        try:
            r = await client.request(method, f"{API}{path}",
                                     headers={"Authorization": f"Bearer {await token.get()}"}, **kw)
        except httpx.HTTPError as exc:
            raise ToolError(f"calendar request failed: {exc}") from exc
        if r.status_code == 404:
            raise ToolError("that calendar isn't shared with Haru (or doesn't exist)")
        if r.status_code >= 400:
            raise ToolError(f"Google Calendar HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def when(value: str | None, default: dt.datetime) -> dt.datetime:
        """'2026-10-02' or '2026-10-02T15:00' (local unless it has an offset)."""
        if not value:
            return default
        try:
            d = dt.datetime.fromisoformat(value)
        except ValueError as exc:
            raise ToolError(f"times must be ISO format like 2026-10-02 or 2026-10-02T15:00, got {value!r}") from exc
        return d.replace(tzinfo=tz) if d.tzinfo is None else d

    def pick(name: str | None) -> dict[str, str]:
        if not name:
            return calendars
        cal = calendars.get(name.lower())
        if cal is None:
            raise ToolError(f"no calendar called {name!r}; available: {sorted(calendars)}")
        return {name.lower(): cal}

    def fmt_time(t: dict) -> tuple[str, bool]:
        if "date" in t:
            return dt.date.fromisoformat(t["date"]).strftime("%A %B %-d"), True
        d = dt.datetime.fromisoformat(t["dateTime"]).astimezone(tz)
        return d.strftime("%A %B %-d, %-I:%M %p"), False

    async def calendar_events(args: dict, ctx: ToolContext) -> dict:
        now = dt.datetime.now(tz)
        start = when(args.get("start"), now)
        # A bare date means that whole day; the default window is the rest of today.
        default_end = (start + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = when(args.get("end"), default_end)
        if end <= start:
            raise ToolError("end must be after start")
        events = []
        for name, cal_id in pick(args.get("calendar")).items():
            params = {"timeMin": start.isoformat(), "timeMax": end.isoformat(), "singleEvents": "true",
                      "orderBy": "startTime", "maxResults": MAX_EVENTS, "timeZone": str(tz)}
            if args.get("search"):
                params["q"] = args["search"]
            data = await api("GET", f"/calendars/{cal_id}/events", params=params)
            for e in data.get("items") or []:
                if e.get("status") == "cancelled":
                    continue
                starts, all_day = fmt_time(e.get("start", {}))
                ends, _ = fmt_time(e.get("end", {}))
                item = {"title": e.get("summary") or "(busy)", "starts": starts, "ends": ends,
                        "all_day": all_day, "calendar": name, "_sort": (e.get("start") or {}).get("dateTime")
                        or (e.get("start") or {}).get("date")}
                if e.get("location"):
                    item["location"] = e["location"]
                if e.get("description"):
                    item["notes"] = e["description"][:200]
                events.append(item)
        events.sort(key=lambda e: e.pop("_sort") or "")
        window = f"{start:%A %B %-d %-I:%M %p} to {end:%A %B %-d %-I:%M %p}"
        return {"window": window, "events": events[:MAX_EVENTS]} if events else {"window": window, "events": [],
                                                                                  "note": "nothing scheduled"}

    async def calendar_availability(args: dict, ctx: ToolContext) -> dict:
        start, end = when(args["start"], dt.datetime.now(tz)), when(args["end"], dt.datetime.now(tz))
        if end <= start:
            raise ToolError("end must be after start")
        cals = pick(args.get("calendar"))
        data = await api("POST", "/freeBusy", json={"timeMin": start.isoformat(), "timeMax": end.isoformat(),
                                                    "timeZone": str(tz), "items": [{"id": c} for c in cals.values()]})
        busy = []
        for name, cal_id in cals.items():
            info = (data.get("calendars") or {}).get(cal_id, {})
            if info.get("errors"):
                raise ToolError(f"can't read {name}'s availability: {info['errors'][0].get('reason')}")
            for b in info.get("busy") or []:
                b0 = dt.datetime.fromisoformat(b["start"]).astimezone(tz)
                b1 = dt.datetime.fromisoformat(b["end"]).astimezone(tz)
                busy.append((b0, b1))
        busy.sort()
        free, cursor = [], start
        for b0, b1 in busy:
            if b0 > cursor:
                free.append((cursor, b0))
            cursor = max(cursor, b1)
        if cursor < end:
            free.append((cursor, end))
        fmt = "%A %-I:%M %p"
        return {"window": f"{start:%A %B %-d %-I:%M %p} to {end:%-I:%M %p}",
                "busy": [f"{a:{fmt}} to {b:%-I:%M %p}" for a, b in busy],
                "free": [f"{a:{fmt}} to {b:%-I:%M %p}" for a, b in free],
                "completely_free": not busy}

    names = ", ".join(sorted(calendars))
    return [
        Tool("calendar_events",
             f"List events on the user's Google calendars ({names}) in a time range, optionally searching by text "
             "(\"dentist\"). Defaults to the rest of today. Use it for \"what's on my calendar\", \"when's my next "
             "meeting\" (start=now, end a week out), or \"when is X\". Read-only.",
             {"type": "object", "properties": {
                 "start": {"type": "string", "description": "ISO local date or time, e.g. 2026-10-02 or 2026-10-02T09:00"},
                 "end": {"type": "string", "description": "ISO local date or time (exclusive)"},
                 "search": {"type": "string", "description": "Only events matching this text"},
                 "calendar": {"type": "string", "description": f"One of: {names} (default: all)"},
             }}, calendar_events),
        Tool("calendar_availability",
             "Check when the user is busy or free between two times, e.g. \"am I free Thursday afternoon\".",
             {"type": "object", "properties": {
                 "start": {"type": "string", "description": "ISO local time, e.g. 2026-10-02T12:00"},
                 "end": {"type": "string", "description": "ISO local time, e.g. 2026-10-02T17:00"},
                 "calendar": {"type": "string", "description": f"One of: {names} (default: all)"},
             }, "required": ["start", "end"]}, calendar_availability),
    ]
