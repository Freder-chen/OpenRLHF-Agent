"""Jina web search and page reader tools."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from openrlhf_agent.agentkit.tools.base import Tool

_SEARCH_URL = "https://s.jina.ai/"
_READER_URL = "https://r.jina.ai/"
_REQUEST_TIMEOUT = 30.0


async def _get_json(
    url: str,
    *,
    api_key: str,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    async with httpx.AsyncClient(headers=headers, timeout=_REQUEST_TIMEOUT) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Jina returned invalid JSON")
    return payload


class JinaSearchTool(Tool):
    name = "jina_search"
    description = (
        "Search the web through the Jina API and return normalized result metadata "
        "such as title, URL, description, and published time."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string to look up information.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, *, api_key: str) -> None:
        if not api_key.strip():
            raise ValueError("api_key is required")
        self.api_key = api_key.strip()

    async def call(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        payload = await _get_json(
            _SEARCH_URL,
            api_key=self.api_key,
            params={"q": query.strip()},
        )
        results = payload.get("data")
        if not isinstance(results, list):
            raise RuntimeError("Jina returned invalid search results")

        keys = ("title", "url", "description", "publishedTime")
        return json.dumps(
            [
                {key: item.get(key) for key in keys}
                for item in results
                if isinstance(item, dict)
            ],
            ensure_ascii=False,
        )


class JinaReadTool(Tool):
    name = "jina_read"
    description = (
        "Read a page through the Jina API and return normalized page data including "
        "title, URL, description, published time, warning, and content."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the web page to read and extract content from.",
            },
        },
        "required": ["url"],
    }

    def __init__(self, *, api_key: str) -> None:
        if not api_key.strip():
            raise ValueError("api_key is required")
        self.api_key = api_key.strip()

    async def call(self, arguments: dict[str, Any]) -> str:
        url = arguments.get("url")
        if not isinstance(url, str):
            raise ValueError("url must be a valid http(s) URL")

        url = url.strip()
        parsed = urlparse(url)
        if not url or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url must be a valid http(s) URL")

        payload = await _get_json(
            f"{_READER_URL}{quote(url, safe=':/?&=%#')}",
            api_key=self.api_key,
        )
        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("Jina returned invalid reader data")

        return json.dumps(
            {
                "title": data.get("title"),
                "url": data.get("url") or url,
                "description": data.get("description"),
                "publishedTime": data.get("publishedTime"),
                "warning": data.get("warning"),
                "content": data.get("content"),
            },
            ensure_ascii=False,
        )
