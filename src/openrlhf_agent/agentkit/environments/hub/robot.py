"""Environment for iterative robot control through tools."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from openrlhf_agent.agentkit.environments.base import Environment
from openrlhf_agent.utils.types import Action, Message, ToolCall


DEFAULT_PROMPT = """
You are controlling a robot through tool calls.
""".strip()


class RobotClient(Protocol):
    """Robot connection used by an environment."""

    tools: Sequence[dict[str, Any]]

    @property
    def success(self) -> bool: ...

    async def new(self) -> Sequence[Message]: ...

    async def execute_tool(
        self,
        call: ToolCall,
    ) -> str | list[dict[str, Any]]: ...

    async def close(self) -> None: ...


class RobotEnvironment(Environment):
    """Run robot tools sequentially until one completes the task."""

    def __init__(
        self,
        *,
        client: RobotClient,
        system_prompt: str | None = None,
        max_steps: int | None = None,
    ) -> None:
        super().__init__(
            system_prompt=system_prompt or DEFAULT_PROMPT,
            max_steps=max_steps,
        )
        self.client = client

    async def __aenter__(self) -> RobotEnvironment:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    def tools_manifest(self) -> list[dict[str, Any]]:
        return list(self.client.tools)

    async def reset(self) -> list[Message]:
        return [
            *await super().reset(),
            *await self.client.new(),
        ]

    async def step(self, action: Action) -> tuple[list[Message], bool]:
        self.step_index += 1
        if not action.tool_calls:
            return [], True

        messages: list[Message] = []
        for call in action.tool_calls:
            result = await self.client.execute_tool(call)
            messages.append(
                Message(role="tool", content=result, tool_call_id=call.call_id)
            )
            if self.client.success:
                return messages, True

        return messages, (
            self.max_steps is not None and self.step_index >= self.max_steps
        )

    async def close(self) -> None:
        await self.client.close()
