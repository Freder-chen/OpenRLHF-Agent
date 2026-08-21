"""Environment for agents that call tools."""

from __future__ import annotations

import asyncio
import json
from typing import Sequence

from openrlhf_agent.utils.types import Action, Message, ToolCall
from openrlhf_agent.agentkit.environments.base import Environment
from openrlhf_agent.agentkit.tools import Tool


DEFAULT_PROMPT = """
You are a helpful assistant.
""".strip()


class FunctionCallEnvironment(Environment):
    """Function call environment with tool and plain-text finals."""

    def __init__(
        self,
        *,
        tools: Sequence[Tool] | None = None,
        system_prompt: str | None = None,
        max_steps: int | None = None,
    ) -> None:
        super().__init__(
            tools=tools if tools is not None else [],
            system_prompt=system_prompt if system_prompt is not None else DEFAULT_PROMPT,
            max_steps=max_steps,
        )

    async def step(self, action: Action) -> tuple[list[Message], bool]:
        """Apply one assistant action and return its observations."""

        self.step_index += 1
        if action.error:
            # Return malformed model output so the model can retry.
            observations = [
                Message(
                    role="user",
                    content=(
                        f"Invalid response: {action.error} "
                        "Use the tool-call format or reply with plain text."
                    ),
                )
            ]
        elif not action.tool_calls:
            return [], True
        else:
            observations = await asyncio.gather(
                *(self._run_tool_call(call) for call in action.tool_calls)
            )

        # Stop at the configured limit. None means unlimited steps.
        reached_limit = self.max_steps is not None and self.step_index >= self.max_steps
        return observations, reached_limit

    async def _run_tool_call(self, tool_call: ToolCall) -> Message:
        if tool_call.error:
            return Message(
                role="tool",
                content=f"Invalid tool call: {tool_call.error}",
                tool_call_id=tool_call.call_id,
            )

        name = (tool_call.name or "").strip()
        if not name:
            return Message(
                role="tool",
                content="Tool name is required.",
                tool_call_id=tool_call.call_id,
            )

        if name not in self.tools:
            available_tools = ", ".join(self.tools) or "none"
            return Message(
                role="tool",
                content=f"Unknown tool '{name}'. Available tools: {available_tools}.",
                tool_call_id=tool_call.call_id,
            )

        try:
            outcome = await self.execute_tool(tool_call)
        except Exception as error:
            return Message(
                role="tool",
                content=f"Tool '{name}' failed: {error}. Revise the arguments.",
                tool_call_id=tool_call.call_id,
            )

        content = outcome if isinstance(outcome, str) else json.dumps(outcome, ensure_ascii=False)
        return Message(role="tool", content=content, tool_call_id=tool_call.call_id)
