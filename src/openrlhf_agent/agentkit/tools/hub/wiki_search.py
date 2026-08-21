"""Wiki search tool backed by a retriever and formatted output."""

from __future__ import annotations

from typing import Any, Mapping

import httpx

from openrlhf_agent.agentkit.tools.base import Tool


class WikiSearchTool(Tool):
    """Query a wiki-style retriever and return formatted passages."""

    name = "wiki_search"
    description = "Search a wiki retriever and return up to `topk` formatted passages."

    MIN_TOPK = 1
    MAX_TOPK = 10
    DEFAULT_TOPK = 3

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "topk": {
                "type": "integer",
                "description": "Maximum number of passages to return.",
                "minimum": MIN_TOPK,
                "maximum": MAX_TOPK,
                "default": DEFAULT_TOPK,
            },
        },
        "required": ["query"],
    }

    def __init__(self, *, base_url: str, timeout: float = 600.0):
        self.base_url = base_url
        self.timeout = timeout

    async def call(self, arguments: dict[str, Any]) -> str:
        # Validate arguments.
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        topk = arguments.get("topk", self.DEFAULT_TOPK)
        if isinstance(topk, bool) or not isinstance(topk, int):
            raise ValueError("topk must be an integer")
        if not self.MIN_TOPK <= topk <= self.MAX_TOPK:
            raise ValueError(f"topk must be between {self.MIN_TOPK} and {self.MAX_TOPK}")

        # Query the retriever.
        request = {
            "queries": [query.strip()],
            "topk": topk,
            "return_scores": True,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.base_url, json=request)
            response.raise_for_status()

        payload = response.json()
        results = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(results, list) or not results:
            raise RuntimeError("Retriever returned invalid results")

        passages = results[0]
        if not isinstance(passages, list):
            raise RuntimeError("Retriever returned invalid passages")

        # Format passages for the model.
        blocks = []
        for index, passage in enumerate(passages, start=1):
            if not isinstance(passage, Mapping):
                raise RuntimeError("Retriever returned an invalid passage")
            document = passage.get("document")
            if not isinstance(document, Mapping):
                raise RuntimeError("Retriever returned an invalid document")

            content = str(document.get("contents") or "").strip()
            lines = content.splitlines()
            title = lines[0].strip() if lines else ""
            body = "\n".join(lines[1:]).strip()

            header = f"Doc {index}"
            if title:
                header += f" — {title}"
            blocks.append(f"{header}\n{body}" if body else header)

        return "\n\n".join(blocks) or "No passages found."
