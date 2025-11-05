"""Tool abstractions and the default think tool."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

import json


class ToolBase(ABC):
    """Minimal function-style tool definition."""

    name: str
    description: str
    parameters: Dict[str, Any]

    def openai_tool(self) -> Dict[str, Any]:
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
    def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        raise NotImplementedError


class ToolRegistry:
    """Simple name-to-tool lookup table."""

    def __init__(self, tools: Sequence[ToolBase]):
        self._tools: Dict[str, ToolBase] = {tool.name: tool for tool in tools}

    def register(self, tool: ToolBase) -> None:
        self._tools[tool.name] = tool

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def list_openai_tools(self) -> List[Dict[str, Any]]:
        return [tool.openai_tool() for tool in self._tools.values()]

    def get(self, name: str) -> ToolBase:
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'.")
        return self._tools[name]


class ThinkTool(ToolBase):
    """Hidden planning tool used to capture private notes."""

    name = "think"
    description = "Write down private notes before taking a visible action."
    parameters = {
        "type": "object",
        "properties": {
            "notes": {
                "type": "string",
                "description": "Short plan or reasoning that stays internal.",
            }
        },
        "required": ["notes"],
    }

    def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        return json.dumps({"ok": True}, ensure_ascii=False)


__all__ = ["ToolBase", "ToolRegistry", "ThinkTool"]

