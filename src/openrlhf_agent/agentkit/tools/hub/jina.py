from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional
from urllib.parse import quote, urlparse

import aiohttp

from openrlhf_agent.agentkit.tools import ToolBase

JINA_SEARCH_URL = "https://s.jina.ai/"
JINA_READER_URL = "https://r.jina.ai/"
REQUEST_TIMEOUT_SECONDS = 30


async def _request_json(
    url: str,
    *,
    api_key: str,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    try:
        async with aiohttp.ClientSession(
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
        ) as session:
            async with session.get(url, params=params) as response:
                text = await response.text()
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("Jina returned invalid JSON.") from exc
                if response.status >= 400:
                    detail = ""
                    if isinstance(payload, dict):
                        detail = str(
                            payload.get("detail")
                            or payload.get("message")
                            or payload.get("title")
                            or ""
                        ).strip()
                    suffix = f": {detail}" if detail else ""
                    raise RuntimeError(f"Jina request failed with HTTP {response.status}{suffix}")
                return payload
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"Jina request timed out after {REQUEST_TIMEOUT_SECONDS} seconds."
        ) from exc
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"Jina request failed: {exc}") from exc


class JinaSearchTool(ToolBase):
    name = "jina_search"
    description = (
        "Search the web through the Jina API and return normalized result metadata "
        "such as title, URL, description, and published time."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string to look up information.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__()
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Jina AI API key is required. Set `JINA_API_KEY` or pass `api_key` explicitly."
            )

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("`query` must be a non-empty string.")

        payload = await _request_json(JINA_SEARCH_URL, api_key=self.api_key, params={"q": query})
        results = payload.get("data", payload)
        keys = ("title", "url", "description", "publishedTime")
        if isinstance(results, list):
            results = [
                {key: item.get(key) for key in keys}
                for item in results
                if isinstance(item, dict)
            ]
        elif isinstance(results, dict):
            results = {key: results.get(key) for key in keys}
        return json.dumps(results, ensure_ascii=False)


class JinaReadTool(ToolBase):
    name = "jina_read"
    description = (
        "Read a page through the Jina API and return normalized page data including "
        "title, URL, description, published time, warning, and content."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the web page to read and extract content from.",
            },
        },
        "required": ["url"],
    }

    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__()
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Jina AI API key is required. Set `JINA_API_KEY` or pass `api_key` explicitly."
            )

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        url = str(arguments.get("url") or "").strip()
        parsed = urlparse(url)
        if not url or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("`url` must be a valid http(s) URL.")

        payload = await _request_json(
            f"{JINA_READER_URL}{quote(url, safe=':/?&=%#')}",
            api_key=self.api_key,
        )
        data = payload.get("data", payload)
        normalized = {
            "title": data.get("title"),
            "url": data.get("url") or url,
            "description": data.get("description"),
            "publishedTime": data.get("publishedTime"),
            "warning": data.get("warning"),
            "content": data.get("content"),
        }
        return json.dumps(normalized, ensure_ascii=False)
