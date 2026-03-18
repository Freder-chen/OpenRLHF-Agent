"""Control-plane tools for hidden reasoning, status updates, and explicit finals."""

from __future__ import annotations

import json
from typing import Any, Dict

from openrlhf_agent.agentkit.tools.base import ToolBase


class CommentaryTool(ToolBase):
    """Send a brief status update separate from the final answer."""

    name = "commentary"
    description = (
        "Send a short status update about current actions or progress. "
        "Do not use this tool for the final answer or key content."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": (
                    "Short status message about the current action, e.g. "
                    "\"Checking recent data\", \"Reviewing code\". "
                    "Do not include final answers or long explanations."
                ),
            },
        },
        "required": ["message"],
    }

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        return json.dumps({"ok": True}, ensure_ascii=False)


class FinalTool(ToolBase):
    """Explicit final-answer tool for structured outputs."""

    name = "final"
    description = "Return the final response that will be shown to the user."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "Final response to the user.",
            },
        },
        "required": ["response"],
    }

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        response = str(arguments.get("response", "")).strip()
        if not response:
            return json.dumps(
                {"ok": False, "error": "response must be a non-empty string."},
                ensure_ascii=False,
            )
        return json.dumps({"ok": True, "response": response}, ensure_ascii=False)


class ThinkTool(ToolBase):
    """Capture hidden reasoning, plans, and intermediate calculations."""

    name = "think"
    description = "Structured thinking tool for the model to capture reasoning, plans, and intermediate calculations."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "description": "Step-by-step reasoning, a concise plan, and intermediate calculations.",
            }
        },
        "required": ["note"],
    }

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        return ""
