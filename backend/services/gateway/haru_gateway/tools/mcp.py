"""MCP servers as Haru tools (stdio transport).

Each configured server is a subprocess the gateway keeps running. Its tools are
registered as ordinary Haru tools (prefixed, e.g. `github_list_notifications`), so
the registry's validation, timeouts and error feedback apply to them too.

Two lessons baked in:
- One long-lived task owns each MCP session. The SDK's streams are tied to the
  task that entered them (anyio cancel scopes); calling a session from other
  tasks crashed an earlier MCP client. Tool calls are queued to that task.
- Read-only by construction: unless a server is explicitly allowed write tools,
  only tools the server itself annotates `readOnlyHint` are registered. That's a
  second guard behind the server's own read-only mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from haru_gateway.tools import Tool, ToolContext, ToolError, ToolRegistry

log = logging.getLogger("haru.gateway.mcp")

DESCRIPTION_CHARS = 600

# The voice-useful subset of github-mcp-server's read-only tools (it offers ~30,
# ~9k tokens of schema). Override with HARU_GITHUB_TOOLS. No notification tools:
# the token is a fine-grained one (Relevan org), and GitHub's notifications API
# only accepts classic tokens.
DEFAULT_GITHUB_TOOLS = {
    "get_me",
    "search_pull_requests", "list_pull_requests", "pull_request_read",
    "search_issues", "list_issues", "issue_read",
    "actions_list", "actions_get", "get_job_logs",
    "get_file_contents", "search_repositories", "list_commits", "get_latest_release",
}


@dataclass
class MCPServerConfig:
    name: str                              # "github": tool prefix and log name
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    allow: set[str] = field(default_factory=set)  # tool names to expose (empty: all read-only tools)
    allow_writes: bool = False
    call_timeout_s: float = 30.0
    rewrite_args: Callable[[str, dict], dict] | None = None   # (tool name, args) -> args, before each call


@dataclass
class _Call:
    name: str
    args: dict
    future: asyncio.Future


class MCPServer:
    def __init__(self, config: MCPServerConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry
        self.ready = asyncio.Event()
        self._queue: asyncio.Queue[_Call] = asyncio.Queue()
        self._registered: list[str] = []

    async def run_forever(self) -> None:
        attempt = 0
        while True:
            try:
                await self._session()
                attempt = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("MCP server %s stopped: %s", self.config.name, exc)
            self.ready.clear()
            self._unregister()
            attempt += 1
            await asyncio.sleep(min(60.0, 2 ** attempt) * random.uniform(0.5, 1.0))

    async def _session(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.types import PaginatedRequestParams

        params = StdioServerParameters(command=self.config.command, args=self.config.args,
                                       env={"PATH": os.environ.get("PATH", ""), **self.config.env})
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = []
            cursor = None
            while True:
                page = await session.list_tools(params=PaginatedRequestParams(cursor=cursor) if cursor else None)
                tools += page.tools
                cursor = page.next_cursor
                if not cursor:
                    break
            self._register(tools)
            self.ready.set()
            while True:
                call = await self._queue.get()
                if call.future.done():
                    continue
                try:
                    result = await session.call_tool(call.name, call.args,
                                                     read_timeout_seconds=self.config.call_timeout_s)
                except Exception as exc:
                    if not call.future.done():
                        call.future.set_exception(ToolError(f"{call.name} failed: {exc}"))
                    raise  # the session is suspect now: reconnect
                if not call.future.done():
                    call.future.set_result(result)

    def _register(self, tools) -> None:
        self._unregister()
        exposed, skipped_writes = [], []
        for t in tools:
            ann = t.annotations
            read_only = bool(ann and getattr(ann, "read_only_hint", False))
            if self.config.allow and t.name not in self.config.allow:
                continue
            if not read_only and not self.config.allow_writes:
                skipped_writes.append(t.name)
                continue
            name = f"{self.config.name}_{t.name}"
            self.registry.add(Tool(name, (t.description or t.title or t.name)[:DESCRIPTION_CHARS],
                                   t.input_schema or {"type": "object", "properties": {}},
                                   self._handler(t.name), timeout_s=self.config.call_timeout_s + 5,
                                   group=self.config.name))
            exposed.append(t.name)
        self._registered = [f"{self.config.name}_{n}" for n in exposed]
        log.info("MCP %s: %d tools exposed (%s)%s", self.config.name, len(exposed), ", ".join(exposed),
                 f"; refused {len(skipped_writes)} non-read-only: {', '.join(skipped_writes)}" if skipped_writes else "")

    def _unregister(self) -> None:
        for name in self._registered:
            self.registry.tools.pop(name, None)
        self._registered = []

    def _handler(self, tool_name: str):
        async def call(args: dict, ctx: ToolContext) -> str:
            if not self.ready.is_set():
                raise ToolError(f"{self.config.name} is reconnecting; try again in a moment")
            if self.config.rewrite_args:
                args = self.config.rewrite_args(tool_name, dict(args))
            fut = asyncio.get_running_loop().create_future()
            await self._queue.put(_Call(tool_name, args, fut))
            result = await fut
            texts = [getattr(c, "text", "") for c in (getattr(result, "content", None) or [])]
            text = "\n".join(t for t in texts if t)
            if not text and getattr(result, "structured_content", None) is not None:
                text = json.dumps(result.structured_content, ensure_ascii=False, default=str)
            if getattr(result, "is_error", False):
                raise ToolError(text[:400] or f"{tool_name} returned an error")
            return text or "(empty result)"
        return call


_SCOPE = re.compile(r"\b(org|repo|user):\S+", re.I)


def scope_github_search(org: str) -> Callable[[str, dict], dict]:
    """Unscoped GitHub searches run across ALL of GitHub. Live, "is:pr is:open
    review:required" (with owner="Relevan", which search ignores) matched 3,986,427
    PRs and Haru reported other people's repos as Relevan's. So any search query
    without its own org:/repo:/user: qualifier is pinned to the configured org."""
    def rewrite(tool: str, args: dict) -> dict:
        if tool.startswith("search_") and isinstance(args.get("query"), str) and not _SCOPE.search(args["query"]):
            args["query"] = f"{args['query']} org:{org}".strip()
        return args
    return rewrite


def github_server_config() -> MCPServerConfig | None:
    """The official github-mcp-server, read-only and in lockdown mode."""
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    binary = os.environ.get("HARU_GITHUB_MCP_BINARY", "/usr/local/bin/github-mcp-server")
    if not token or not os.path.exists(binary):
        return None
    allow = {t.strip() for t in os.environ.get("HARU_GITHUB_TOOLS", "").split(",") if t.strip()}
    if not allow:
        allow = DEFAULT_GITHUB_TOOLS
    return MCPServerConfig(
        name="github", command=binary, args=["stdio"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": token,
             "GITHUB_TOOLSETS": os.environ.get("HARU_GITHUB_TOOLSETS",
                                               "context,repos,issues,pull_requests,actions,notifications"),
             "GITHUB_READ_ONLY": "1",
             "GITHUB_LOCKDOWN_MODE": "1"},
        allow=allow,
        rewrite_args=scope_github_search(org) if (org := os.environ.get("HARU_GITHUB_ORG", "")) else None)
