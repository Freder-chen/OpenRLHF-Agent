"""Shared interfaces for agent environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from openrlhf_agent.utils.types import Action, Message, ToolCall
from openrlhf_agent.agentkit.tools import Tool


class Environment(ABC):
    """Base class for stateful agent environments."""

    def __init__(
        self,
        *,
        system_prompt: str,
        tools: Sequence[Tool] = (),
        max_steps: int | None = None,
    ) -> None:
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Tool names must be unique.")

        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be at least 1 or None")

        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.step_index = 0

    def tools_manifest(self) -> list[dict[str, Any]]:
        """Return tools in the OpenAI function-calling format."""

        return [tool.to_openai_tool() for tool in self.tools.values()]

    async def execute_tool(
        self,
        call: ToolCall,
    ) -> str | list[dict[str, Any]]:
        """Execute one tool invocation."""

        if call.name not in self.tools:
            raise KeyError(f"Unknown tool '{call.name}'.")

        tool = self.tools[call.name]
        return await tool.call(call.arguments or {})

    async def reset(self) -> list[Message]:
        """Reset one rollout and return its initial messages."""

        self.step_index = 0
        return [Message(role="system", content=self.system_prompt)]

    @abstractmethod
    async def step(self, action: Action) -> tuple[list[Message], bool]:
        """Run one environment transition and return (observations, done)."""
