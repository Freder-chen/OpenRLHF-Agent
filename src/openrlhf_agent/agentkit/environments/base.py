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
        tools: Sequence[ToolBase],
        system_prompt: str,
        max_steps: int,
    ) -> None:
        # Tools
        tool_list = list(tools)
        if len({tool.name for tool in tool_list}) != len(tool_list):
            raise ValueError("Tool names must be unique.")
        self._tool_map: Dict[str, ToolBase] = {tool.name: tool for tool in tool_list}
        
        # Prompt
        self._system_prompt = system_prompt
        if self._system_prompt is None:
            raise NotImplementedError(
                f"{type(self).__name__} must supply a system prompt via __init__ or override this property."
            )

        # Step
        self._step_index = 0
        self._max_steps = max_steps

    def tools_manifest(self) -> List[Dict[str, Any]]:
        return [tool.openai_tool() for tool in self._tool_map.values()]

    async def execute_tool(self, call: ToolCall, context: Dict[str, Any]) -> str:
        """Execute one tool invocation."""

        if call.name not in self._tool_map:
            raise KeyError(f"Unknown tool '{call.name}'.")
        
        tool = self._tool_map[call.name]
        arguments = call.arguments
        
        if arguments is None or isinstance(arguments, dict):
            return await tool.call(context=context, arguments=arguments)

        raise TypeError("Tool arguments must be a JSON object.")

    def tool_names(self) -> List[str]:
        return list(self._tool_map.keys())

    def register_tool(self, tool: ToolBase) -> None:
        """Add a tool at runtime."""

        if tool.name in self._tool_map:
            raise ValueError(f"Tool '{tool.name}' already exists.")
        self._tool_map[tool.name] = tool

    @property
    def system_prompt(self) -> str:
        """Return the system prompt used for the agent."""
        return self._system_prompt

    def reset_step(self) -> None:
        self._step_index = 0
    
    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @abstractmethod
    async def step(self, action: Action) -> Tuple[List[str], bool]:
        """Run one environment transition and return (observations, done)."""
