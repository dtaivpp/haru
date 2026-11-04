"""Web tools via Firecrawl (https://docs.firecrawl.dev): search, and read one page."""

from __future__ import annotations

import os

import httpx

from haru_gateway.tools import Tool, ToolContext, ToolError

FIRECRAWL = "https://api.firecrawl.dev/v2"
USER_AGENT = "haru-gateway/0.1"
PAGE_CHARS = 5000          # per page read; results go into the model's context
RESULT_CHARS = 1200        # per search result, when page content is included


def web_tools() -> list[Tool]:
    key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not key:
        return []
    client = httpx.AsyncClient(timeout=45, headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT})

    async def post(path: str, body: dict) -> dict:
        try:
            r = await client.post(f"{FIRECRAWL}{path}", json=body)
        except httpx.HTTPError as exc:
            raise ToolError(f"web request failed: {exc}") from exc
        if r.status_code >= 400:
            raise ToolError(f"Firecrawl HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        if data.get("success") is False:
            raise ToolError(f"Firecrawl error: {data.get('error') or data}")
        return data

    async def web_search(args: dict, ctx: ToolContext) -> dict:
        body = {"query": args["query"], "limit": int(args.get("max_results", 5))}
        if args.get("read_pages"):
            body["scrapeOptions"] = {"formats": ["markdown"], "onlyMainContent": True}
        if args.get("recent"):
            body["tbs"] = {"day": "qdr:d", "week": "qdr:w", "month": "qdr:m"}[args["recent"]]
        data = await post("/search", body)
        results = []
        for r in (data.get("data") or {}).get("web") or []:
            item = {"title": r.get("title"), "url": r.get("url"), "snippet": r.get("description")}
            if r.get("markdown"):
                item["content"] = r["markdown"][:RESULT_CHARS]
            results.append(item)
        return {"query": args["query"], "results": results} if results else {"query": args["query"], "results": [],
                                                                             "note": "no results"}

    async def read_webpage(args: dict, ctx: ToolContext) -> dict:
        data = await post("/scrape", {"url": args["url"], "formats": ["markdown"], "onlyMainContent": True})
        page = data.get("data") or {}
        md = page.get("markdown") or ""
        meta = page.get("metadata") or {}
        return {"url": args["url"], "title": meta.get("title"), "content": md[:PAGE_CHARS],
                "truncated": len(md) > PAGE_CHARS}

    return [
        Tool("web_search",
             "Search the web for current information and facts you're unsure of. Returns titles, URLs and "
             "snippets; set read_pages for page text when snippets won't be enough. NOT the first choice for local "
             "businesses (hours, open now, ratings, phone), restaurant menus, product prices, stock prices or sports "
             "results: use find_data_tools for those, it returns exact structured data.",
             {"type": "object", "properties": {
                 "query": {"type": "string", "description": "Search query, as you'd type it into a search engine"},
                 "max_results": {"type": "integer", "minimum": 1, "maximum": 8, "description": "Default 5"},
                 "read_pages": {"type": "boolean", "description": "Include each result's page text (slower)"},
                 "recent": {"type": "string", "enum": ["day", "week", "month"],
                            "description": "Only results from the last day/week/month"},
             }, "required": ["query"]},
             web_search, timeout_s=40),
        Tool("read_webpage",
             "Read the main text of a specific web page (for example a URL from web_search).",
             {"type": "object", "properties": {"url": {"type": "string", "description": "Full http(s) URL"}},
              "required": ["url"]},
             read_webpage, timeout_s=45),
    ]
