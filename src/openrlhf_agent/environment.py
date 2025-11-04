"""Environment orchestration logic used by OpenRLHF agents."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openrlhf_agent.types import ParsedAssistantAction


# ---------------------------------------------------------------------------
# Tool helpers


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
    """Simple name to tool lookup table."""

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
        return str(arguments.get("notes", ""))


# ---------------------------------------------------------------------------
# Reward helpers


def extract_verdict(text: str) -> Optional[str]:
    """Return [[A]] or [[B]] if found; otherwise None."""

    verdicts = [match for match in ("[[A]]", "[[B]]") if match in text]
    return verdicts[-1] if verdicts else None


def compute_reward(
    response: str,
    target: str,
    *,
    correct_score: float = 1.0,
    verdict_score: float = 0.1,
    miss_score: float = 0.0,
) -> float:
    """Score the response against a ground-truth label string."""

    gold = target.strip()
    if response.strip() == gold:
        return correct_score
    if extract_verdict(response) == gold:
        return verdict_score
    return miss_score


# ---------------------------------------------------------------------------
# Environment base


SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful agent assistant.\n\n"
    "You may call tools to plan privately, but anything outside tool calls is visible to the user.\n\n"
    "Knowledge cutoff: 2023-06\n"
    "Current date: {date}\n\n"
    "Rules:\n"
    "- Use think(notes=...) when you need to plan internally; its output is hidden from the user.\n"
    "- To answer the user, provide plain text outside tool calls. That text ends the session.\n"
    "- Tool calls must be JSON objects within <tool_call></tool_call> tags."
)


class Environment(ABC):
    """Abstract interface for agent environments."""

    @property
    @abstractmethod
    def max_steps(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def tools_manifest(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def execute_tool(self, name: str, args: Dict[str, Any], context: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def reward_hook(self, tool_name: str, tool_args: Dict[str, Any], label: Optional[str]) -> float:
        raise NotImplementedError

    @abstractmethod
    def reset(self, observation: Optional[Sequence[Dict[str, str]]] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        action: ParsedAssistantAction,
        label: Optional[str] = None,
        runtime: bool = False,
    ) -> Tuple[List[str], float, bool, Optional[str]]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete environment


class FunctionCallEnvironment(Environment):
    """Default environment with a private think tool and plain-text finals."""

    def __init__(
        self,
        *,
        max_steps: int = 32,
        reward_config: Optional[Dict[str, float]] = None,
    ) -> None:
        self.registry = ToolRegistry([ThinkTool()])
        self._max_steps = max_steps
        self._step_index = 0
        self._initial_messages: Optional[Sequence[Dict[str, str]]] = None

        self._reward_config = {
            "correct_score": 1.0,
            "verdict_score": 0.1,
            "miss_score": 0.0,
        }
        if reward_config:
            self._reward_config.update(reward_config)

    # ------------------------------------------------------------------ props

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(date=datetime.now().strftime("%Y-%m-%d"))

    # ----------------------------------------------------------------- tooling

    def tools_manifest(self) -> List[Dict[str, Any]]:
        return self.registry.list_openai_tools()

    def execute_tool(self, name: str, args: Dict[str, Any], context: Dict[str, Any]) -> str:
        tool = self.registry.get(name)
        return tool.call(context=context, arguments=args)

    # ---------------------------------------------------------------- lifecycle

    def reset(self, observation: Optional[Sequence[Dict[str, str]]] = None) -> None:
        self._step_index = 0
        self._initial_messages = observation

    def reward_hook(self, tool_name: str, tool_args: Dict[str, Any], label: Optional[str]) -> float:
        if label is None:
            return 0.0
        if tool_name != "final":
            return 0.0

        answer = str(tool_args.get("answer", "")).strip()
        if not answer:
            return 0.0
        return compute_reward(answer, label, **self._reward_config)

    # ----------------------------------------------------------------- helpers

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
            "__internal": True,
            "visible_to_user": False,
            "ok": False,
            "error": {"code": code, "message": message},
            "policy": {
                "planning_requires_tools": True,
                "final_response_must_be_plain_text": True,
            },
            "allowed_tools": self.registry.names(),
        }
        if hint:
            payload["hint"] = hint
        if tool:
            payload["tool"] = tool
        if extras:
            payload.update(extras)
        return json.dumps(payload, ensure_ascii=False)

    # ------------------------------------------------------------------- steps

    def step(
        self,
        action: ParsedAssistantAction,
        label: Optional[str] = None,
        runtime: bool = False,
    ) -> Tuple[List[str], float, bool, Optional[str]]:
        observations: List[str] = []
        reward = 0.0
        terminated = False
        final_response: Optional[str] = None

        if action.refusal:
            observations.append(
                self._internal_message(
                    code="parse_error",
                    message=action.refusal,
                    hint="Wrap tool calls in <tool_call> tags or reply with plain text only.",
                )
            )
            self._step_index += 1
            return observations, reward, terminated, final_response

        if not action.tool_calls:
            response = (action.content or "").strip()
            if not response:
                observations.append(
                    self._internal_message(
                        code="empty_final",
                        message="Final response cannot be empty when no tool calls are provided.",
                        hint="Reply with plain text to finish or call think(...) first.",
                    )
                )
                self._step_index += 1
                return observations, reward, terminated, None

            final_response = response
            terminated = True
            if not runtime:
                reward += self.reward_hook("final", {"answer": final_response}, label)
            self._step_index += 1
            return observations, reward, terminated, final_response

        allowed_tools = set(self.registry.names())

        for index, tool_call in enumerate(action.tool_calls):
            if tool_call is None:
                continue

            if tool_call.refusal:
                observations.append(
                    self._internal_message(
                        code="tool_call_error",
                        message=tool_call.refusal,
                        hint="Fix the tool call JSON payload and try again.",
                        extras={"tool_call_id": tool_call.id, "action_index": index},
                    )
                )
                continue

            name = (tool_call.name or "").strip()
            if not name:
                observations.append(
                    self._internal_message(
                        code="missing_tool_name",
                        message="Tool name is required.",
                        hint="Provide a function name inside the tool call payload.",
                        extras={"tool_call_id": tool_call.id, "action_index": index},
                    )
                )
                continue

            if name not in allowed_tools:
                observations.append(
                    self._internal_message(
                        code="invalid_tool",
                        message=f"Tool '{name}' is not available.",
                        hint="Choose one of the allowed tools.",
                        extras={"tool_call_id": tool_call.id, "action_index": index},
                    )
                )
                continue

            arguments = tool_call.arguments or {}
            if not isinstance(arguments, dict):
                observations.append(
                    self._internal_message(
                        code="invalid_arguments",
                        message="Tool arguments must be a JSON object.",
                        tool=name,
                        hint="Use key/value pairs when building tool arguments.",
                        extras={
                            "tool_call_id": tool_call.id,
                            "action_index": index,
                            "arguments": arguments,
                        },
                    )
                )
                continue

            context = {
                "step_index": self._step_index,
                "action_index": index,
                "initial_messages": self._initial_messages,
            }

            try:
                outcome = self.execute_tool(name=name, args=arguments, context=context)
            except Exception as exc:  # pragma: no cover - defensive guard
                observations.append(
                    self._internal_message(
                        code="tool_runtime_error",
                        message=f"Tool '{name}' raised an exception.",
                        tool=name,
                        hint="Revise the arguments or plan with think(...) before retrying.",
                        extras={
                            "tool_call_id": tool_call.id,
                            "action_index": index,
                            "exception": str(exc),
                        },
                    )
                )
                continue

            if name == ThinkTool.name:
                observations.append(
                    self._internal_message(
                        code="think_notes",
                        message="Captured private notes via think().",
                        tool=name,
                        extras={"notes": outcome, "action_index": index},
                    )
                )
            else:
                observations.append(str(outcome))

            if not runtime:
                reward += self.reward_hook(name, arguments, label)

        self._step_index += 1

        if self._step_index >= self.max_steps:
            terminated = True

        return observations, reward, terminated, final_response


def make_environment(name: Optional[str] = None, **kwargs: Any) -> Environment:
    if name in (None, "default"):
        return FunctionCallEnvironment(**kwargs)
    raise ValueError(f"Unknown environment '{name}'.")


__all__ = ["Environment", "FunctionCallEnvironment", "make_environment"]
