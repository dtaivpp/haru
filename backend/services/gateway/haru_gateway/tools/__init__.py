"""Haru's tools: functions the LLM can call mid-reply.

A Tool is a name, a description, a JSON-schema for its arguments, and an async
handler. The registry turns tools into the OpenAI `tools` list and runs the
model's calls. Every call is checked against its schema first, and a bad call,
an unknown tool, a timeout or a crash comes back to the model as
{"error": ...}, not as an exception, so it can correct itself on the next round
(a lesson from an earlier build: local models make occasional malformed calls, and a
reliability layer that feeds errors back is what makes tools usable).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

log = logging.getLogger("haru.gateway.tools")

MAX_RESULT_CHARS = 6000


@dataclass
class ToolContext:
    device_id: str


Handler = Callable[[dict, ToolContext], Awaitable[object]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Handler
    timeout_s: float = 15.0
    group: str | None = None   # gated tools are only offered when their topic is active (see specs())

    def spec(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


class ToolError(Exception):
    """Raised by handlers for expected failures; the message goes back to the model."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str   # raw JSON text, as the model produced it


@dataclass
class ToolResult:
    call: ToolCall
    content: str
    ok: bool
    ms: float


_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
               "object": dict, "array": list}


def validate(args: object, schema: dict) -> list[str]:
    """Just enough JSON-schema for tool arguments: required, types, enums, bounds."""
    if not isinstance(args, dict):
        return ["arguments must be a JSON object"]
    problems = []
    props = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            problems.append(f"missing required argument {name!r}")
    for name, value in args.items():
        spec = props.get(name)
        if spec is None:
            if schema.get("additionalProperties") is False:
                problems.append(f"unknown argument {name!r}")
            continue
        expected = _JSON_TYPES.get(spec.get("type", ""))
        if expected and (not isinstance(value, expected) or (isinstance(value, bool) and spec["type"] != "boolean")):
            problems.append(f"{name!r} must be {spec['type']}, got {type(value).__name__}")
            continue
        if "enum" in spec and value not in spec["enum"]:
            problems.append(f"{name!r} must be one of {spec['enum']}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in spec and value < spec["minimum"]:
                problems.append(f"{name!r} must be >= {spec['minimum']}")
            if "maximum" in spec and value > spec["maximum"]:
                problems.append(f"{name!r} must be <= {spec['maximum']}")
    return problems


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, *tools: Tool) -> None:
        for t in tools:
            self.tools[t.name] = t

    def specs(self, active_groups: set[str] | None = None) -> list[dict]:
        """Ungrouped tools always; grouped ones (e.g. the ~16 GitHub tools, ~6k tokens of
        schema) only when their topic is active, so they don't slow every request."""
        active = active_groups or set()
        return [t.spec() for t in self.tools.values() if t.group is None or t.group in active]

    def groups(self) -> set[str]:
        return {t.group for t in self.tools.values() if t.group}

    def __bool__(self) -> bool:
        return bool(self.tools)

    async def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        t0 = time.monotonic()

        def done(payload: object, ok: bool) -> ToolResult:
            text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str)
            if len(text) > MAX_RESULT_CHARS:
                text = text[:MAX_RESULT_CHARS] + " …[truncated]"
            ms = (time.monotonic() - t0) * 1000
            try:   # one line, whatever whitespace the model used
                shown = json.dumps(json.loads(call.arguments or "{}"), ensure_ascii=False, separators=(",", ":"))
            except json.JSONDecodeError:
                shown = " ".join(call.arguments.split())
            (log.info if ok else log.warning)("tool %s(%s) -> %s in %.0f ms: %s", call.name, shown[:300],
                                              "ok" if ok else "error", ms, text[:160])
            return ToolResult(call, text, ok, ms)

        tool = self.tools.get(call.name)
        if tool is None:
            return done({"error": f"unknown tool {call.name!r}; available: {sorted(self.tools)}"}, False)
        try:
            args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as exc:
            return done({"error": f"arguments are not valid JSON ({exc}); send a JSON object"}, False)
        if problems := validate(args, tool.parameters):
            return done({"error": "invalid arguments: " + "; ".join(problems)}, False)
        try:
            result = await asyncio.wait_for(tool.handler(args, ctx), tool.timeout_s)
        except TimeoutError:
            return done({"error": f"{call.name} timed out after {tool.timeout_s:.0f}s"}, False)
        except ToolError as exc:
            return done({"error": str(exc)}, False)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("tool %s crashed", call.name)
            return done({"error": f"{call.name} failed: {type(exc).__name__}: {exc}"}, False)
        return done(result, True)
