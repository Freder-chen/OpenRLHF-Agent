"""Wiki search tool backed by a retriever and formatted output."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from openrlhf_agent.agentkit.tools import ToolBase


class WikiSearchTool(ToolBase):
    """Query a wiki-style retriever and return formatted passages."""

    name = "wiki_search"
    description = "Search a wiki retriever and return up to `topk` formatted passages."

    MIN_TOPK = 1
    MAX_TOPK = 10
    DEFAULT_TOPK = 3

    parameters: Dict[str, Any] = {
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
        self.retriever_url = base_url
        self.timeout = float(timeout)

    @classmethod
    def _parse_topk(cls, value: Any) -> int:
        try:
            topk = int(value)
        except (TypeError, ValueError):
            topk = cls.DEFAULT_TOPK
        return max(cls.MIN_TOPK, min(cls.MAX_TOPK, topk))

    def _format_passages(self, passages: Sequence[Mapping[str, Any]]) -> str:
        """Format a list of retrieved passages into a readable string."""
        blocks: list[str] = []

        for i, passage in enumerate(passages, start=1):
            # Passages may contain {"document": {"contents": "..."}}, ignore any "score" fields.
            document = passage.get("document") or {}
            content = str(document.get("contents") or "").strip()

            header = f"Doc {i}"
            if not content:
                blocks.append(header)
                continue

            lines = content.splitlines()
            title = lines[0].strip() if lines else ""
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

            if title:
                header += f" — {title}"
            blocks.append(f"{header}\n{body}".rstrip() if body else header)

        return "\n\n".join(blocks).strip()

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        import httpx

        query = str(arguments.get("query", "")).strip()
        if not query:
            return "Missing required argument: `query`."

        topk = self._parse_topk(arguments.get("topk"))

        request_payload = {"queries": [query], "topk": topk, "return_scores": True}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.retriever_url, json=request_payload)
                response.raise_for_status()
                response_data = response.json()

            results = response_data.get("result")
            if not isinstance(results, list) or not results:
                return "No results returned by retriever."

            # Server usually returns: {"result": [[...docs...]]} for one query
            passages = results[0]
            if not isinstance(passages, list):
                return "Unexpected retriever response format: `result[0]` is not a list."

            return self._format_passages(passages) or "No passages found."

        except httpx.TimeoutException:
            return f"Request timed out after {self.timeout:.1f}s."
        except httpx.HTTPStatusError as exc:
            return f"Request failed with HTTP {exc.response.status_code}."
        except Exception as exc:
            return f"Request failed: {exc}"
