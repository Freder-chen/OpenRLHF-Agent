"""Shared interfaces for agent environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openrlhf_agent.utils.types import Action, ToolCall
from openrlhf_agent.agentkit.tools import ToolBase


class Environment(ABC):
    """Base interface describing the agent environment contract."""

    def __init__(
        self,
        *,
        system_prompt: str,
        tools: Sequence[ToolBase] = [],
    ) -> None:
        self.system_prompt = system_prompt

        tool_list = list(tools)
        if len({tool.name for tool in tool_list}) != len(tool_list):
            raise ValueError("Tool names must be unique.")
        self._tool_map: Dict[str, ToolBase] = {tool.name: tool for tool in tool_list}

    def tools_manifest(self) -> List[Dict[str, Any]]:
        return [tool.openai_tool() for tool in self._tool_map.values()]

    def tool_names(self) -> List[str]:
        return list(self._tool_map.keys())

    def register_tool(self, tool: ToolBase) -> None:
        if tool.name in self._tool_map:
            raise ValueError(f"Tool '{tool.name}' already exists.")
        self._tool_map[tool.name] = tool

    async def execute_tool(self, call: ToolCall, context: Dict[str, Any]) -> str:
        if call.name not in self._tool_map:
            raise KeyError(f"Unknown tool '{call.name}'.")
        tool = self._tool_map[call.name]
        if call.arguments is None or isinstance(call.arguments, dict):
            return await tool.call(context=context, arguments=call.arguments)
        raise TypeError("Tool arguments must be a JSON object.")

    @abstractmethod
    async def step(self, action: Action) -> Tuple[List[str], bool]:
        """Run one environment transition and return (observations, done)."""
