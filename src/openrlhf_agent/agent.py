"""Agent orchestration for tool-using language models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from openrlhf_agent.environment import Environment
from openrlhf_agent.model import LLMEngine
from openrlhf_agent.template import Template
from openrlhf_agent.types import ChatMessage, ParsedAssistantAction


@dataclass
class AgentStepResult:
    """Container for one agent-environment interaction."""

    assistant_message: ChatMessage
    tool_messages: List[ChatMessage] = field(default_factory=list)
    reward: float = 0.0
    terminated: bool = False
    final_response: Optional[str] = None
    parse_error: bool = False
    raw_action_text: str = ""


class AgentSession:
    """Tracks conversation state and mediates between model and environment."""

    def __init__(self, environment: Environment, template: Template) -> None:
        self.environment = environment
        self.template = template
        self.history: List[ChatMessage] = []

    # ------------------------------------------------------------------ helpers

    def _coerce_message(self, message: Any) -> ChatMessage:
        if isinstance(message, ChatMessage):
            return message
        if isinstance(message, dict):
            return ChatMessage(**message)
        raise TypeError(f"Unsupported message type: {type(message)!r}")

    def _dump_history(self) -> List[Dict[str, Any]]:
        return [msg.model_dump(exclude_none=True) for msg in self.history]

    def _compute_prompt(self) -> str:
        return self.template.render_messages(
            messages=self._dump_history(),
            tools_manifest=self.environment.tools_manifest(),
            add_generation_prompt=True,
        )

    @staticmethod
    def _has_parse_error(action: ParsedAssistantAction) -> bool:
        if action.refusal:
            return True
        return any(call.refusal for call in action.tool_calls)

    # ---------------------------------------------------------------- lifecycle

    def initialize(self, messages: Sequence[Dict[str, Any]]) -> str:
        """Reset environment state and return the initial rendered prompt."""

        self.environment.reset(messages)
        self.history = [
            ChatMessage(role="system", content=self.environment.system_prompt)
        ]
        for message in messages:
            self.history.append(self._coerce_message(message))

        return self._compute_prompt()

    def render_prompt(self) -> str:
        return self._compute_prompt()

    # ------------------------------------------------------------------- stepping

    def step(
        self,
        action: ParsedAssistantAction,
        *,
        label: Optional[str] = None,
        runtime: bool = False,
        raw_text: Optional[str] = None,
    ) -> AgentStepResult:
        """Apply a parsed assistant action to the environment."""

        assistant_message = ChatMessage(
            role="assistant",
            content=action.content,
            tool_calls=list(action.tool_calls),
            refusal=action.refusal,
        )
        self.history.append(assistant_message)

        observations, reward, terminated, final_response = self.environment.step(
            action,
            label=label,
            runtime=runtime,
        )

        tool_messages: List[ChatMessage] = []
        for observation in observations:
            tool_msg = ChatMessage(role="tool", content=observation)
            tool_messages.append(tool_msg)
            self.history.append(tool_msg)

        parse_error = self._has_parse_error(action)
        if parse_error and not action.tool_calls and raw_text is not None:
            assistant_message.content = raw_text

        if final_response and not assistant_message.content:
            assistant_message.content = final_response

        return AgentStepResult(
            assistant_message=assistant_message,
            tool_messages=tool_messages,
            reward=reward,
            terminated=terminated,
            final_response=final_response,
            parse_error=parse_error,
            raw_action_text=raw_text or "",
        )

    def step_from_text(
        self,
        action_text: str,
        *,
        label: Optional[str] = None,
        runtime: bool = False,
    ) -> AgentStepResult:
        parsed_action = self.template.parse_assistant_text(action_text)
        return self.step(
            parsed_action,
            label=label,
            runtime=runtime,
            raw_text=action_text,
        )


class AgentRuntime:
    """Lightweight runtime that streams tool calls and observations."""

    def __init__(
        self,
        engine: LLMEngine,
        environment: Environment,
        template: Template,
        *,
        max_new_tokens_per_step: int = 10240,
    ) -> None:
        self.engine = engine
        self.session = AgentSession(environment, template)
        self.environment = self.session.environment
        self.template = self.session.template
        self.max_new_tokens_per_step = max_new_tokens_per_step

    @staticmethod
    def _is_internal_obs(text: Optional[str]) -> bool:
        if not text:
            return False
        try:
            payload = json.loads(text)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        if payload.get("__internal") is True:
            return True
        return payload.get("visible_to_user") is False

    @staticmethod
    def _message_to_dict(message: ChatMessage) -> Dict[str, Any]:
        return message.model_dump(exclude_none=True)

    def run_steps(self, messages: Sequence[Dict[str, Any]]):
        """Yield chat messages emitted during the interaction loop."""

        prompt = self.session.initialize(messages)
        prompt_ids = self.engine.tokenize(prompt)

        for _ in range(self.environment.max_steps):
            _, action_text = self.engine.generate(
                prompt_ids,
                max_tokens=self.max_new_tokens_per_step,
            )

            step_result = self.session.step_from_text(action_text, runtime=True)

            assistant_msg = step_result.assistant_message
            if step_result.parse_error and assistant_msg.tool_calls:
                assistant_msg = ChatMessage(role="assistant", content=step_result.raw_action_text)

            yield self._message_to_dict(assistant_msg)

            for tool_msg in step_result.tool_messages:
                if not self._is_internal_obs(tool_msg.content):
                    yield self._message_to_dict(tool_msg)

            if step_result.terminated:
                return

            prompt = self.session.render_prompt()
            prompt_ids = self.engine.tokenize(prompt)

        yield self._message_to_dict(
            ChatMessage(role="assistant", content="Max steps reached without final response.")
        )

    def run_final(self, messages: Sequence[Dict[str, Any]]) -> Optional[str]:
        """Convenience wrapper that returns the last assistant content."""

        final_text: Optional[str] = None
        for message in self.run_steps(messages):
            role = message.get("role") if isinstance(message, dict) else None
            if role != "assistant":
                continue
            tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
            if tool_calls:
                continue
            content = message.get("content") if isinstance(message, dict) else None
            final_text = content or final_text
        return final_text


__all__ = ["AgentRuntime", "AgentSession", "AgentStepResult"]
