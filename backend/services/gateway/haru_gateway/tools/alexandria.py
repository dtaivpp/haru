"""Firecrawl Alexandria: a catalogue of structured-data tools Haru can discover and run.

https://docs.firecrawl.dev/features/alexandria. Discovery is free; running a tool
costs the credits listed on it (e.g. Bing local-business search: 5). Two tools
mirror Firecrawl's own flow:

  find_data_tools: search the catalogue (free), then fetch the matching tools'
                   contracts (inputs, price) in the same call, so the model can
                   go straight to running one.
  run_data_tool:   execute one tool, inside guard rails: a per-call price cap, a
                   daily credit budget, inputs checked against the contract.
                   Providers needing licence terms return where an admin must
                   accept them; Haru never accepts terms herself.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path

import httpx

from haru_gateway.tools import Tool, ToolContext, ToolError

log = logging.getLogger("haru.gateway.alexandria")

FIRECRAWL = "https://api.firecrawl.dev/v2"
# Data tools return lists (Bing: up to 30 businesses). Cut mid-JSON, the model misreads
# what's left, so allow more room than for web pages and drop fields that never help.
RESULT_CHARS = 12000
NOISE_KEYS = {"image_url", "maps_url", "category_path", "observed_at_ms", "thumbnail", "thumbnail_url", "icon",
              "latitude", "longitude", "style_category", "resolved_query", "image", "images", "logo_url"}


def _compact_options(options: list[dict]) -> list[dict]:
    out = []
    for o in options or []:
        item = {"name": o.get("name"), "type": o.get("type")}
        if o.get("required"):
            item["required"] = True
        for k in ("default", "min", "max", "enum"):
            if k in o:
                item[k] = o[k]
        if o.get("about"):
            item["about"] = o["about"][:160]
        out.append(item)
    return out


def _strip_noise(value):
    if isinstance(value, dict):
        return {k: _strip_noise(v) for k, v in value.items() if k not in NOISE_KEYS and v is not None}
    if isinstance(value, list):
        return [_strip_noise(v) for v in value]
    return value


class _Budget:
    """Credits spent today, persisted so a restart doesn't reset the daily cap."""

    def __init__(self, path: Path, daily: int) -> None:
        self.path, self.daily = path, daily

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def spent_today(self) -> int:
        d = self._read()
        return int(d.get("spent", 0)) if d.get("date") == str(dt.date.today()) else 0

    def add(self, credits: int) -> int:
        spent = self.spent_today() + credits
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"date": str(dt.date.today()), "spent": spent}))
        except OSError as exc:
            log.warning("could not save Alexandria budget: %s", exc)
        return spent


def alexandria_tools() -> list[Tool]:
    key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not key or os.environ.get("HARU_ALEXANDRIA", "1") != "1":
        return []
    max_call = int(os.environ.get("HARU_ALEXANDRIA_MAX_CREDITS_PER_CALL", "25"))
    budget = _Budget(Path(os.environ.get("HARU_ALEXANDRIA_BUDGET_FILE", "/data/state/alexandria-budget.json")),
                     int(os.environ.get("HARU_ALEXANDRIA_DAILY_CREDITS", "500")))
    client = httpx.AsyncClient(timeout=90, headers={"Authorization": f"Bearer {key}", "User-Agent": "haru-gateway/0.1"})
    contracts: dict[str, dict] = {}   # "provider/capability" -> contract (cached)

    async def post(path: str, body: dict) -> dict:
        try:
            r = await client.post(f"{FIRECRAWL}{path}", json=body)
        except httpx.HTTPError as exc:
            raise ToolError(f"Alexandria request failed: {exc}") from exc
        data = r.json() if r.content else {}
        if r.status_code == 403:
            raise ToolError("Alexandria isn't enabled for this Firecrawl team")
        if r.status_code >= 400:
            raise ToolError(f"Firecrawl HTTP {r.status_code}: {json.dumps(data)[:200]}")
        return data

    async def alexandria_call(provider: str, capability: str, options: dict) -> dict:
        data = await post("/scrape", {"alexandria": {"provider": provider, "capability": capability, "options": options}})
        results = (data.get("data") or {}).get("alexandria") or []
        if not results:
            raise ToolError("Alexandria returned no result")
        res = results[0]
        if err := res.get("error"):
            code = err.get("code", "") if isinstance(err, dict) else str(err)
            if code == "THIRD_PARTY_DATA_TERMS_REQUIRED":
                url = ((err.get("requiresAction") or {}).get("url") or (res.get("requiresAction") or {}).get("url")
                       or "https://www.firecrawl.dev/app/settings?tab=data-sources")
                raise ToolError(f"{provider} needs its data terms accepted by a Firecrawl org admin at {url} "
                                "before Haru can use it. Tell the user; don't retry.")
            raise ToolError(f"{provider}/{capability} failed: {json.dumps(err)[:240]}")
        return res

    async def load_contracts(providers: list[str]) -> None:
        missing = [p for p in providers if not any(k.startswith(p + "/") for k in contracts)]
        if not missing:
            return
        res = await alexandria_call("firecrawl", "find-tools", {
            "providers": missing, "level": "tools", "expand": ["options"], "limit": 50})
        for item in (res.get("data") or {}).get("items") or []:
            if item.get("provider") and item.get("capability"):
                contracts[f"{item['provider']}/{item['capability']}"] = item

    async def find_data_tools(args: dict, ctx: ToolContext) -> dict:
        data = await post("/search", {"query": args["need"], "sources": ["alexandria"],
                                      "limit": int(args.get("max_tools", 4)), "domainTools": False})
        found = (data.get("data") or {}).get("tools") or []
        if not found:
            return {"tools": [], "note": "no structured-data tool covers this; use web_search instead"}
        await load_contracts(sorted({t["provider"] for t in found if t.get("provider")}))
        tools = []
        for t in found:
            c = contracts.get(f"{t.get('provider')}/{t.get('capability')}", {})
            tools.append({"provider": t.get("provider"), "capability": t.get("capability"),
                          "description": (t.get("description") or c.get("description") or "")[:300],
                          "credits": c.get("creditsCost"), "per_record": c.get("perRecord"),
                          "options": _compact_options(c.get("options") or [])})
        return {"tools": tools, "budget_left_today": budget.daily - budget.spent_today()}

    async def run_data_tool(args: dict, ctx: ToolContext) -> dict:
        provider, capability = args["provider"], args["capability"]
        options = args.get("options") or {}
        await load_contracts([provider])
        contract = contracts.get(f"{provider}/{capability}")
        if contract is None:
            raise ToolError(f"no Alexandria tool {provider}/{capability}; use find_data_tools first")
        missing = [o["name"] for o in contract.get("options") or [] if o.get("required") and o["name"] not in options]
        if missing:
            raise ToolError(f"missing required options: {missing}")
        cost = contract.get("creditsCost") or 0
        if cost > max_call:
            raise ToolError(f"{provider}/{capability} costs {cost} credits per call, over Haru's limit of {max_call}")
        if contract.get("perRecord"):
            # Per-record pricing: cap the records so the worst case stays within the per-call limit.
            limit_opt = next((o["name"] for o in contract.get("options") or [] if o["name"] in ("count", "limit")), None)
            if limit_opt:
                options[limit_opt] = min(int(options.get(limit_opt, 5)), max(1, max_call // max(cost, 1)))
        spent = budget.spent_today()
        if spent + max(cost, 1) > budget.daily:
            raise ToolError(f"today's Alexandria budget ({budget.daily} credits) is used up ({spent} spent)")
        res = await alexandria_call(provider, capability, options)
        charged = int(res.get("creditsCost") or cost or 0)
        total = budget.add(charged)
        log.info("alexandria %s/%s cost %d credits (%d today)", provider, capability, charged, total)
        text = json.dumps(_strip_noise(res.get("data")), ensure_ascii=False, default=str)
        return {"tool": f"{provider}/{capability}", "credits_used": charged,
                "data": json.loads(text) if len(text) <= RESULT_CHARS else text[:RESULT_CHARS] + " …[truncated]"}

    return [
        Tool("find_data_tools",
             "Find structured-data tools (Firecrawl Alexandria) for a need, with their inputs and price. Covers local "
             "businesses and their hours/ratings/phones, restaurant menus, product prices and availability, stock "
             "and market data, company info, news, sports results, jobs, travel and hotels, podcasts, property, "
             "research papers and more. Free. Prefer it over web_search for those kinds of questions.",
             {"type": "object", "properties": {
                 "need": {"type": "string", "description": "What data you need, e.g. \"coffee shops near an address with hours\""},
                 "max_tools": {"type": "integer", "minimum": 1, "maximum": 6, "description": "Default 4"},
             }, "required": ["need"]}, find_data_tools, timeout_s=30),
        Tool("run_data_tool",
             "Run one tool returned by find_data_tools, with options from its contract. Costs its listed credits.",
             {"type": "object", "properties": {
                 "provider": {"type": "string"}, "capability": {"type": "string"},
                 "options": {"type": "object", "description": "Inputs from the tool's options list"},
             }, "required": ["provider", "capability"]}, run_data_tool, timeout_s=60),
    ]
