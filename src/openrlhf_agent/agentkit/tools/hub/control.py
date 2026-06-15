"""Control-plane tools for reasoning and structured output."""

from __future__ import annotations

from typing import Any, Dict

from openrlhf_agent.agentkit.tools.base import ToolBase


class ThinkTool(ToolBase):
    """Concise reasoning visible to the user."""

    name = "think"
    description = "Use this tool to think a step before acting."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "A concrete reasoning step with specific details: what you found, what it means, what to do next.",
            }
        },
        "required": ["thought"],
    }

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        return ""


class CommentaryTool(ToolBase):
    """Emit a brief progress update visible to the user."""

    name = "commentary"
    description = "Send a short progress update to the user while you continue working."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "A brief progress status.",
            },
        },
        "required": ["message"],
    }

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        return ""


class FinalTool(ToolBase):
    """Explicitly mark the final response to the user."""

    name = "final"
    description = "Submit your final answer to the user. Use this when you are done."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "The final answer to present to the user.",
            },
        },
        "required": ["response"],
    }

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        response = str(arguments.get("response", "")).strip()
        if not response:
            return "InputValidationError: The required parameter `response` must be a non-empty string."
        return response
