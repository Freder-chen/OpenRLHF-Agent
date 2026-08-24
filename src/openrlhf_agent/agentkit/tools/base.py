"""Base class for tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """A function that an agent can call."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_tool(self) -> dict[str, Any]:
        """Return a schema that matches OpenAI's function tool format."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def call(
        self,
        arguments: dict[str, Any],
    ) -> str | list[dict[str, Any]]:
        """Execute the tool and return message content."""
