"""Remote LLM (GPT-OSS-120B) connection settings + a reachability probe.

Stage 4 adds actual chat calls; for now the gateway only needs to know whether
the LLM is up so it can report it in backend.status (failure-behaviour spec:
the backend keeps running and says "unavailable" instead of dying).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

log = logging.getLogger("haru.gateway.llm")

# The endpoint sits behind Cloudflare, which 403s (error 1010, "browser
# signature") on Python's default "Python-urllib/x.y" User-Agent. Every request
# to the LLM must send an explicit one.
USER_AGENT = "haru-gateway/0.1"


@dataclass
class LLMConfig:
    base_url: str
    model: str
    api_key: str
    reasoning_effort: str
    health_interval_s: float

    @classmethod
    def from_env(cls) -> LLMConfig | None:
        base_url = os.environ.get("HARU_LLM_BASE_URL", "").rstrip("/")
        if not base_url:
            return None
        return cls(
            base_url=base_url,
            model=os.environ.get("HARU_LLM_MODEL", "gpt-oss-120b"),
            api_key=os.environ.get("HARU_LLM_API_KEY", ""),
            reasoning_effort=os.environ.get("HARU_LLM_REASONING_EFFORT", ""),
            health_interval_s=float(os.environ.get("HARU_LLM_HEALTH_INTERVAL_S", "30")),
        )


def probe(config: LLMConfig, timeout: float = 5.0) -> str:
    """One blocking health check. Returns a status string for backend.status:
    ok | no_api_key | unauthorized | model_missing | unavailable."""
    if not config.api_key:
        return "no_api_key"
    req = urllib.request.Request(f"{config.base_url}/models", headers={
        "Authorization": f"Bearer {config.api_key}",
        "User-Agent": USER_AGENT,
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            models = {m.get("id") for m in json.load(resp).get("data", [])}
    except urllib.error.HTTPError as exc:
        body = exc.read(200).decode(errors="replace").strip()
        log.debug("LLM probe HTTP %d: %s", exc.code, body)
        if exc.code in (401, 403):
            log.warning("LLM probe rejected: HTTP %d %s", exc.code, body)
            return "unauthorized"
        return "unavailable"
    except (OSError, ValueError):
        return "unavailable"
    return "ok" if config.model in models else "model_missing"


async def watch(config: LLMConfig, on_status) -> None:
    """Probe forever, calling on_status(status) whenever it changes."""
    last = None
    while True:
        status = await asyncio.to_thread(probe, config)
        if status != last:
            log.log(logging.INFO if status == "ok" else logging.WARNING,
                    "LLM %s (%s at %s)", status, config.model, config.base_url)
            last = status
            await on_status(status)
        await asyncio.sleep(config.health_interval_s)
