"""Search tool that calls a local retrieval server and formats passages."""

import asyncio
import json
from typing import Any, Dict, List

import httpx

from openrlhf_agent.agentkit.tools import ToolBase


class LocalSearchTool(ToolBase):
    """Calls a local Search-R1 style retriever and returns formatted passages."""

    name = "local_search"
    description = (
        "Retrieve supporting passages from a local search backend. "
        "Use this to gather references before drafting the answer."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "User query to send to the retriever.",
            },
            "topk": {
                "type": "integer",
                "description": "How many passages to return.",
                "minimum": 1,
                "maximum": 10,
                "default": 3,
            },
            "return_scores": {
                "type": "boolean",
                "description": "Whether to include similarity scores.",
                "default": True,
            },
        },
        "required": ["query"],
    }

    def __init__(self, *, base_url: str = "http://localhost:8000/retrieve", timeout: float = 10.0):
        self.base_url = base_url
        self.timeout = timeout

    def _passages_to_string(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieval results into a readable string."""

        formatted = []
        for idx, item in enumerate(results):
            doc = item.get("document", {}) or {}
            content = str(doc.get("contents", "")).split("\n")
            title = content[0].strip() if content else ""
            text = "\n".join(part for part in content[1:] if part).strip()
            score = item.get("score")

            parts = [f"Doc {idx + 1}"]
            if title:
                parts.append(f"(Title: {title})")
            if score is not None:
                parts.append(f"[score: {score}]")

            body = " ".join(parts)
            if text:
                body = f"{body} {text}"
            formatted.append(body)

        return "\n".join(formatted)

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return json.dumps({"ok": False, "error": "query is required"})

        topk = int(arguments.get("topk") or 3)
        return_scores = bool(arguments.get("return_scores", True))

        payload = {
            "queries": [query],
            "topk": topk,
            "return_scores": return_scores,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.base_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"request failed: {exc}"}, ensure_ascii=False)

        results = data.get("result", [[]])
        first_query_results = results[0] if results else []
        formatted = self._passages_to_string(first_query_results)

        return json.dumps({"ok": True, "references": formatted}, ensure_ascii=False)


__all__ = ["LocalSearchTool"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the LocalSearchTool against a retrieval server.")
    parser.add_argument("--base-url", default="http://localhost:8000/retrieve", help="Retriever endpoint.")
    parser.add_argument("--query", default="百年孤独的作者?", help="Query string to send.")
    parser.add_argument("--topk", type=int, default=3, help="Number of passages to fetch.")
    parser.add_argument(
        "--no-scores",
        action="store_true",
        help="Skip similarity scores in the request (return_scores=False).",
    )
    args = parser.parse_args()

    async def _run() -> None:
        tool = LocalSearchTool(base_url=args.base_url)
        payload = {
            "query": args.query,
            "topk": args.topk,
            "return_scores": not args.no_scores,
        }
        result = await tool.call(context={}, arguments=payload)
        print(result)

    asyncio.run(_run())
