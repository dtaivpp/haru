"""Notifications: the silent inbox (a bell with a count on the face page, opened on request)."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from zoneinfo import ZoneInfo

from websockets.asyncio.client import connect

from haru_gateway.server import Gateway, GatewayConfig
from haru_gateway.tools import ToolCall, ToolContext, ToolRegistry
from haru_gateway.tools.notifications import KEEP, NotificationService
from haru_protocol import MessageType, decode, encode
from test_stage1 import TOKEN, free_port, until

TZ = ZoneInfo("UTC")


def service(tmp_path):
    svc = NotificationService(tmp_path / "notifications.json", TZ)
    badges, panels = [], []

    async def on_badge(dev):
        badges.append(dev)

    async def on_panel(dev, items, open_):
        panels.append((dev, items, open_))
    svc.on_badge(on_badge)
    svc.on_panel(on_panel)
    return svc, badges, panels


def test_add_scopes_counts_and_persists(tmp_path):
    async def body():
        svc, badges, _ = service(tmp_path)
        await svc.add("Capy finished: fix CI timeout", "PR #12 is open", source="capy", device_id="desk")
        await svc.add("Everyone sees this")
        assert badges == ["desk", None]
        assert svc.unread_count("desk") == 2 and svc.unread_count("kitchen") == 1
        # Saved on every change: a new service (a gateway restart) sees the same state.
        again = NotificationService(tmp_path / "notifications.json", TZ)
        assert again.unread_count("desk") == 2
        assert [n.title for n in again.for_device("desk")] == ["Everyone sees this", "Capy finished: fix CI timeout"]
    asyncio.run(body())


def test_open_shows_panel_marks_read_and_clear_removes(tmp_path):
    async def body():
        svc, badges, panels = service(tmp_path)
        a = await svc.add("first", device_id="desk")
        await svc.add("second", device_id="desk")
        shown = await svc.open("desk")
        assert [i["title"] for i in shown] == ["second", "first"]          # newest first
        assert all(not i["read"] for i in shown)          # shown as new; marked read afterwards
        assert panels[-1] == ("desk", shown, True)
        assert svc.unread_count("desk") == 0 and badges[-1] is None      # every screen refreshes
        assert await svc.open("desk") == []               # nothing unread left
        assert len(await svc.open("desk", include_read=True)) == 2
        assert await svc.clear("desk", [a.id]) == 1
        assert await svc.clear("desk") == 1               # every read one
        assert panels[-1] == ("desk", [], False)          # clearing closes the panel
        assert svc.for_device("desk") == []
    asyncio.run(body())


def test_old_read_ones_are_pruned_and_count_is_capped(tmp_path):
    async def body():
        svc, _, _ = service(tmp_path)
        old = await svc.add("old and read")
        old.read = True
        old.created = (svc.now() - dt.timedelta(days=8)).isoformat(timespec="seconds")
        for i in range(KEEP + 5):
            await svc.add(f"n{i}")
        assert old.id not in svc.items and len(svc.items) == KEEP
    asyncio.run(body())


def test_tools(tmp_path):
    async def body():
        svc, _, panels = service(tmp_path)
        reg = ToolRegistry()
        reg.add(*svc.tools())
        ctx = ToolContext(device_id="desk")
        r = await reg.run(ToolCall("1", "open_notifications", "{}"), ctx)
        assert r.ok and json.loads(r.content)["note"] == "nothing new"
        assert panels[-1] == ("desk", [], True)           # the panel still opens, saying "Nothing new"
        await svc.add("Capy finished: bump deps", "All checks passed.", source="capy")
        r = json.loads((await reg.run(ToolCall("2", "open_notifications", "{}"), ctx)).content)
        assert r["notifications"][0]["title"] == "Capy finished: bump deps"
        assert r["notifications"][0]["body"] == "All checks passed." and r["notifications"][0]["when"] == "just now"
        r = json.loads((await reg.run(ToolCall("3", "clear_notifications", "{}"), ctx)).content)
        assert r == {"cleared": 1, "unread_left": 0}
    asyncio.run(body())


def test_display_gets_badge_on_attach_and_on_debug_notify(tmp_path, monkeypatch):
    monkeypatch.setenv("HARU_NOTIFICATIONS_FILE", str(tmp_path / "n.json"))
    monkeypatch.setenv("HARU_TIMERS_FILE", str(tmp_path / "t.json"))

    async def body():
        port = free_port()
        gw = Gateway(GatewayConfig("127.0.0.1", port, TOKEN, debug_commands=True))
        await gw.notifications.add("waiting already", device_id="desk")
        stop = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(gw.serve(stop))
        await asyncio.sleep(0.1)
        got = []

        async def watch():
            async with connect(f"ws://127.0.0.1:{port}/display?device=desk&token={TOKEN}") as ws:
                async for raw in ws:
                    got.append(decode(raw))
        watcher = asyncio.create_task(watch())
        await until(lambda: any(m.type == MessageType.NOTIFY_BADGE for m in got))
        assert [m.data for m in got if m.type == MessageType.NOTIFY_BADGE] == [{"unread": 1}]

        async with connect(f"ws://127.0.0.1:{port}/terminal",
                           additional_headers={"Authorization": f"Bearer {TOKEN}"}) as term:
            await term.send(encode(MessageType.DEVICE_STATUS, {"device_id": "cli", "software": "test",
                                                               "platform": "test", "capabilities": []}))
            await term.recv()
            await term.send(encode("debug.notify", {"text": "Capy finished: fix CI timeout"}))
            await until(lambda: [m.data for m in got if m.type == MessageType.NOTIFY_BADGE][-1] == {"unread": 2})
        watcher.cancel()
        stop.set_result(None)
        await task

    asyncio.run(body())


def test_reading_on_one_screen_updates_every_screen(tmp_path, monkeypatch):
    """Read state is shared: a broadcast notification opened on one terminal clears every badge."""
    monkeypatch.setenv("HARU_NOTIFICATIONS_FILE", str(tmp_path / "n.json"))
    monkeypatch.setenv("HARU_TIMERS_FILE", str(tmp_path / "t.json"))

    async def body():
        port = free_port()
        gw = Gateway(GatewayConfig("127.0.0.1", port, TOKEN))
        await gw.notifications.add("for everyone")
        stop = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(gw.serve(stop))
        await asyncio.sleep(0.1)
        badges = {"desk": [], "kitchen": []}

        async def watch(dev):
            async with connect(f"ws://127.0.0.1:{port}/display?device={dev}&token={TOKEN}") as ws:
                async for raw in ws:
                    m = decode(raw)
                    if m.type == MessageType.NOTIFY_BADGE:
                        badges[dev].append(m.data["unread"])
        watchers = [asyncio.create_task(watch(d)) for d in badges]
        await until(lambda: badges["desk"] and badges["kitchen"])
        await gw.notifications.open("desk")
        await until(lambda: badges["kitchen"][-1] == 0 and badges["desk"][-1] == 0)
        for w in watchers:
            w.cancel()
        stop.set_result(None)
        await task

    asyncio.run(body())
