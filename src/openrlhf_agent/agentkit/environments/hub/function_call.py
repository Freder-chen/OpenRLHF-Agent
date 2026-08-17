"""Environment that supports function calls plus a hidden status tool."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from openrlhf_agent.utils.types import Action, ToolCall
from openrlhf_agent.agentkit.environments.base import Environment
from openrlhf_agent.agentkit.tools import ThinkTool, ToolBase


SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant.
""".strip()


class FunctionCallEnvironment(Environment):
    """Function call environment with tool and plain-text finals."""

    def __init__(
        self,
        *,
        tools: Optional[Sequence[ToolBase]] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 9999,
    ) -> None:
        tools = tools or []
        super().__init__(
            tools=list(tools),
            system_prompt=system_prompt or SYSTEM_PROMPT_TEMPLATE,
            max_steps=max_steps,
        )

    async def step(self, action: Action) -> Tuple[List[str], bool]:
        observations: List[str] = []
        terminated = False
        tool_calls = action.tool_calls or []

        if action.refusal:
            # Handle a parsing or refusal error from the model.
            observations.append(
                self._internal_message(
                    code="parse_error",
                    message=action.refusal,
                    hint="Wrap tool calls in the tool call tags or reply with plain text only.",
                )
            )

        else:
            # If there are no tool calls, treat this as the final reply.
            if not tool_calls:
                terminated = True

            # Run tool calls if they exist.
            else:
                observations.extend(await self._run_tool_calls(tool_calls))

        # Bump the step counter and enforce the max step limit.
        self._step_index += 1
        if self._step_index >= self.max_steps:
            terminated = True

        return observations, terminated

    async def _run_tool_calls(self, tool_calls: Sequence[ToolCall]) -> List[str]:
        allowed = set(self.tool_names())
        tasks = [
            self._handle_tool_call(tool_call, allowed_tools=allowed) for tool_call in tool_calls
        ]
        return await asyncio.gather(*tasks)

    async def _handle_tool_call(
        self,
        tool_call: ToolCall,
        *,
        allowed_tools: Set[str],
    ) -> str:
        if tool_call.refusal:
            return self._internal_message(
                code="tool_call_error",
                message=tool_call.refusal,
                hint="Fix the tool call JSON payload.",
                extras={"tool_call_id": tool_call.call_id},
            )

        name = (tool_call.name or "").strip()
        if not name:
            return self._internal_message(
                code="missing_tool_name",
                message="Tool name is required.",
                hint="Provide a function name inside the tool call payload.",
                extras={"tool_call_id": tool_call.call_id},
            )

        if name not in allowed_tools:
            return self._internal_message(
                code="invalid_tool",
                message=f"Tool '{name}' is not available.",
                hint="Choose one of the allowed tools.",
                extras={"tool_call_id": tool_call.call_id},
            )

        arguments = tool_call.arguments or {}
        if not isinstance(arguments, dict):
            return self._internal_message(
                code="invalid_arguments",
                message="Tool arguments must be a JSON object.",
                hint="Use key/value pairs when building tool arguments.",
                extras={"tool_call_id": tool_call.call_id},
            )

        try:
            outcome = await self.execute_tool(call=tool_call, context={})
        except Exception as exc:  # pragma: no cover - defensive guard
            return self._internal_message(
                code="tool_runtime_error",
                message=f"Tool '{name}' raised an exception.",
                hint="Revise the arguments.",
                extras={
                    "tool_call_id": tool_call.call_id,
                    "exception": str(exc),
                },
            )

        return str(outcome)

    def _internal_message(
        self,
        *,
        code: str,
        message: str,
        hint: Optional[str] = None,
        tool: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "ok": False,
            "error": {"code": code, "message": message},
            "allowed_tools": self.tool_names(),
        }
        if hint:
            payload["hint"] = hint
        if tool:
            payload["tool"] = tool
        if extras:
            payload.update(extras)
        return json.dumps(payload, ensure_ascii=False)
