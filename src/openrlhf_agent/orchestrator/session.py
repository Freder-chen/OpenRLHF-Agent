"""Session management for the tool-using agent."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

from openrlhf_agent.chat_protocol import ChatProtocol
from openrlhf_agent.core import AgentStepResult, ChatMessage, ParsedAssistantAction
from openrlhf_agent.environment import Environment
from openrlhf_agent.orchestrator.history import ChatHistory


class AgentSession:
    """Maintains chat history and bridges the protocol with the environment."""

    def __init__(self, environment: Environment, protocol: ChatProtocol) -> None:
        self.environment = environment
        self.protocol = protocol
        self.history = ChatHistory()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _has_parse_error(action: ParsedAssistantAction) -> bool:
        if action.refusal:
            return True
        return any(call.refusal for call in action.tool_calls)

    # ---------------------------------------------------------------- lifecycle

    def initialize(
        self, payload: Optional[Union[Sequence[Dict[str, Any]], str]] = None
    ) -> str:
        """Reset environment state and return the first prompt."""

        self.environment.reset_step()
        self.history.reset(system_prompt=self.environment.system_prompt)

        if isinstance(payload, str) and payload.strip():
            parsed_messages = self.protocol.parse_messages_from_completion_text(payload)
            # NOTE: skip system message
            if parsed_messages and parsed_messages[0].role == "system":
                parsed_messages = parsed_messages[1:]
            if parsed_messages:
                self.history.extend(parsed_messages)
        elif payload:
            self.history.extend(payload)

        return self.history.render_prompt(
            self.protocol,
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

        parsed_action = self.protocol.parse_assistant_text(action_text)
        return self.step(
            parsed_action,
            label=label,
            runtime=runtime,
            raw_text=action_text,
        )
