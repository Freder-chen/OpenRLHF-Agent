"""Control-plane tools for reasoning and structured output."""

from __future__ import annotations

from typing import Any

from openrlhf_agent.agentkit.tools.base import Tool


class ThinkTool(Tool):
    """Concise reasoning visible to the user."""

    name = "think"
    description = "Use this tool to think a step before acting."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": (
                    "A concrete reasoning step: what you found, what it means, "
                    "and what to do next."
                ),
            }
        },
        "required": ["thought"],
    }

    async def call(self, arguments: dict[str, Any]) -> str:
        thought = arguments.get("thought")
        if not isinstance(thought, str) or not thought.strip():
            raise ValueError("thought must be a non-empty string")
        return ""


class CommentaryTool(Tool):
    """Emit a brief progress update visible to the user."""

    name = "commentary"
    description = "Send a short progress update to the user while you continue working."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "A brief progress status.",
            },
        },
        "required": ["message"],
    }

    async def call(self, arguments: dict[str, Any]) -> str:
        message = arguments.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        return ""


class FinalTool(Tool):
    """Explicitly mark the final response to the user."""

    name = "final"
    description = "Submit your final answer to the user. Use this when you are done."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "The final answer to present to the user.",
            },
        },
        "required": ["response"],
    }

    async def call(self, arguments: dict[str, Any]) -> str:
        response = arguments.get("response")
        if not isinstance(response, str) or not response.strip():
            raise ValueError("response must be a non-empty string")
        return response.strip()
