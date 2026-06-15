"""Environment that supports function calls."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import List, Optional, Sequence, Set, Tuple

from openrlhf_agent.utils.types import Action, ToolCall
from openrlhf_agent.agentkit.environments.base import Environment
from openrlhf_agent.agentkit.tools import ThinkTool, ToolBase


SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant.

Current date: {date}
""".strip()


class FunctionCallEnvironment(Environment):
    """Function call environment with tool and plain-text finals."""

    def __init__(
        self,
        *,
        system_prompt: Optional[str] = None,
        tools: Optional[Sequence[ToolBase]] = None,
    ) -> None:
        super().__init__(
            system_prompt=system_prompt or SYSTEM_PROMPT_TEMPLATE.format(
                date=datetime.now().strftime("%Y-%m-%d")
            ),
            tools=list(tools or [ThinkTool()]),
        )

    async def step(self, action: Action) -> Tuple[List[str], bool]:
        # Parse failure — feed the error back so the model can retry.
        if action.refusal:
            observations = [f"ParseError: Failed to parse the assistant response: {action.refusal}"]
            terminated = False

        # Tool calls — validate and execute each one concurrently.
        elif action.tool_calls:
            allowed = set(self.tool_names())
            observations = await asyncio.gather(*[
                self._execute_tool_call(tc, allowed) for tc in action.tool_calls
            ])
            terminated = False

        # No tool calls and no parse error — check for empty response.
        elif not (action.content or "").strip():
            observations = ["EmptyResponseError: The assistant response is empty. Please provide a response."]
            terminated = False

        # Final reply.
        else:
            observations = []
            terminated = True

        return observations, terminated

    async def _execute_tool_call(self, tc: ToolCall, allowed: Set[str]) -> str:
        # Malformed tool call from the parser.
        if tc.refusal:
            return f"ToolCallParseError: Failed to parse tool call arguments: {tc.refusal}"

        name = (tc.name or "").strip()
        if not name:
            return "MissingToolName: The required parameter `name` is missing from the tool call."
        if name not in allowed:
            return f"InvalidToolName: Tool `{name}` is not available. Available tools: {', '.join(sorted(allowed))}"

        arguments = tc.arguments or {}
        if not isinstance(arguments, dict):
            return f"InvalidArguments: Tool `{name}` expects arguments as a JSON object, got {type(arguments).__name__}."

        # Run the tool.
        try:
            return str(await self.execute_tool(call=tc, context={}))
        except Exception as exc:
            return f"ToolRuntimeError: Tool `{name}` raised an exception: {exc}"
