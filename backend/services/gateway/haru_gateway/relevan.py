"""Minimal async client for the Relevan search API (https://api.relevan.dev/docs).

Only what Haru's memory needs: list, search, ingest, update and feedback on one
index. Auth is the org-scoped key in the `x-api-key` header.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

log = logging.getLogger("haru.gateway.relevan")

USER_AGENT = "haru-gateway/0.1"


@dataclass
class RelevanConfig:
    api_key: str
    org: str
    index: str
    base_url: str = "https://api.relevan.dev"
    customer_id: str = "household"   # stable end-user id (Relevan hashes it before storage)

    @classmethod
    def from_env(cls) -> RelevanConfig | None:
        key = os.environ.get("RELEVAN_API_KEY", "")
        org = os.environ.get("RELEVAN_ORG", "")
        if not key or not org:
            return None
        return cls(api_key=key, org=org, index=os.environ.get("RELEVAN_MEMORY_INDEX", "haru-memories"),
                   base_url=os.environ.get("RELEVAN_BASE_URL", "https://api.relevan.dev").rstrip("/"),
                   customer_id=os.environ.get("RELEVAN_CUSTOMER_ID", "household"))


class RelevanError(Exception):
    pass


class Relevan:
    def __init__(self, config: RelevanConfig, timeout: float = 10.0) -> None:
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url, timeout=timeout,
            headers={"x-api-key": config.api_key, "User-Agent": USER_AGENT})
        self._index = f"/v1/{config.org}/indexes/{config.index}"

    async def close(self) -> None:
        await self.client.aclose()

    async def _call(self, method: str, path: str, **kw) -> dict:
        try:
            r = await self.client.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise RelevanError(f"{type(exc).__name__}: {exc}") from exc
        if r.status_code >= 400:
            raise RelevanError(f"HTTP {r.status_code} on {method} {path}: {r.text[:300]}")
        return r.json() if r.content else {}

    async def list_documents(self, max_docs: int = 1000) -> list[dict]:
        """All documents (id + content), paging 100 at a time."""
        out: list[dict] = []
        page = 1
        while len(out) < max_docs:
            data = await self._call("GET", f"{self._index}/documents", params={"page": page, "limit": 100})
            docs = data.get("documents") or []
            out += [{"id": d["id"], **(d.get("content") or {})} for d in docs]
            if len(docs) < 100:
                break
            page += 1
        return out[:max_docs]

    async def search(self, query: str, *, size: int = 8, session_id: str | None = None,
                     filters: list[dict] | None = None) -> dict:
        body: dict = {"query": query, "size": size, "customerId": self.config.customer_id}
        if session_id:
            body["sessionId"] = session_id
        if filters:
            body["filters"] = filters
        return await self._call("POST", f"{self._index}/search", json=body)

    async def ingest(self, docs: list[dict]) -> list[str]:
        """docs: [{"id": ..., **content}] -> accepted ids."""
        if not docs:
            return []
        payload = {"documents": [{"id": d["id"], "content": {k: v for k, v in d.items() if k != "id"}}
                                 for d in docs]}
        data = await self._call("POST", f"{self._index}/documents/_bulk", json=payload)
        return data.get("documentIds") or []

    async def update(self, doc_id: str, content: dict) -> None:
        await self._call("PUT", f"{self._index}/documents/{doc_id}", json={"content": content})

    async def feedback(self, *, action: str, object_id: str | None = None, query_id: str | None = None,
                       session_id: str | None = None, position: int | None = None) -> None:
        body = {"actionName": action, "customerId": self.config.customer_id}
        for k, v in (("objectId", object_id), ("queryId", query_id), ("sessionId", session_id), ("position", position)):
            if v is not None:
                body[k] = v
        await self._call("POST", f"/v1-beta/{self.config.org}/indexes/{self.config.index}/feedback", json=body)
