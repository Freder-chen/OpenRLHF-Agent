"""Session management for the tool-using agent."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from openrlhf_agent.core import AgentStepResult, ChatMessage, ParsedAssistantAction
from openrlhf_agent.environment import Environment
from openrlhf_agent.template import Template
from openrlhf_agent.orchestrator.history import ChatHistory


class AgentSession:
    """Maintains chat history and bridges the template with the environment."""

    def __init__(self, environment: Environment, template: Template) -> None:
        self.environment = environment
        self.template = template
        self.history = ChatHistory()

    # ------------------------------------------------------------------ helpers

    def _coerce_message(self, message: Any) -> ChatMessage:
        """Turn dictionaries into `ChatMessage` objects when needed."""

        if isinstance(message, ChatMessage):
            return message
        if isinstance(message, dict):
            return ChatMessage(**message)
        raise TypeError(f"Unsupported message type: {type(message)!r}")

    @staticmethod
    def _has_parse_error(action: ParsedAssistantAction) -> bool:
        if action.refusal:
            return True
        return any(call.refusal for call in action.tool_calls)

    # ---------------------------------------------------------------- lifecycle

    def initialize(self, messages: Sequence[Dict[str, Any]]) -> str:
        """Reset environment state and return the first prompt."""

        self.environment.reset_step()
        self.history.reset(system_prompt=self.environment.system_prompt)
        self.history.extend(self._coerce_message(message) for message in messages)

        return self.history.render_prompt(
            self.template,
            tools_manifest=self.environment.tools_manifest(),
        )

    def render_prompt(self) -> str:
        return self.history.render_prompt(
            self.template,
            tools_manifest=self.environment.tools_manifest(),
        )

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
            tool_calls=action.tool_calls,
        )
        parse_error = self._has_parse_error(action)
        if parse_error and not action.tool_calls and raw_text is not None:
            # Preserve the unparsed text so the user can see what went wrong.
            assistant_message.content = raw_text
        self.history.add(assistant_message)

        observations, reward, terminated, _ = self.environment.step(
            action, label=label, runtime=runtime
        )

        tool_messages = []
        for observation in observations:
            tool_messages.append(self.history.add_tool_message(observation))

        return AgentStepResult(
            idx=self.environment.step_index,
            assistant_message=assistant_message,
            tool_messages=tool_messages,
            reward=reward,
            terminated=terminated,
        )

    def step_from_text(
        self,
        action_text: str,
        *,
        label: Optional[str] = None,
        runtime: bool = False,
    ) -> AgentStepResult:
        """Parse a raw model response and forward to `step`."""

        parsed_action = self.template.parse_assistant_text(action_text)
        return self.step(
            parsed_action,
            label=label,
            runtime=runtime,
            raw_text=action_text,
        )
