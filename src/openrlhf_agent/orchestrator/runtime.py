"""Streaming runtime loop for the tool-using agent."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from openrlhf_agent.core import AgentStepResult, ChatMessage
from openrlhf_agent.environment import Environment
from openrlhf_agent.engine import LLMEngine
from openrlhf_agent.template import Template

from openrlhf_agent.orchestrator.session import AgentSession


class AgentRuntime:
    """Coordinates the language model with the environment at inference time."""

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
    def _message_to_dict(message: ChatMessage) -> Dict[str, Any]:
        return message.model_dump(exclude_none=True)

    # ------------------------------------------------------------------- helpers

    def _initialize_tokens(self, messages: Sequence[Dict[str, Any]]) -> List[int]:
        prompt = self.session.initialize(messages)
        return list(self.engine.tokenize(prompt))

    def _generate_tokens(self, prompt_ids: Sequence[int]) -> tuple[List[int], str]:
        token_ids, text = self.engine.generate(
            prompt_ids,
            max_tokens=self.max_new_tokens_per_step,
        )
        return list(token_ids), text

    def _emit_step_messages(
        self, step_result: AgentStepResult
    ) -> Iterable[Dict[str, Any]]:
        yield self._message_to_dict(step_result.assistant_message)
        for tool_msg in step_result.tool_messages:
            yield self._message_to_dict(tool_msg)

    def _tool_response_tokens(self, tool_messages: Sequence[ChatMessage]) -> List[int]:
        if not tool_messages:
            # If the environment yields nothing we still need the next assistant prefix.
            prompt_snippet = self.template.render_messages(
                messages=[],
                add_generation_prompt=True,
            )
        else:
            prompt_snippet = self.template.render_messages(
                messages=[msg.model_dump(exclude_none=True) for msg in tool_messages],
                add_generation_prompt=True,
            )
        return list(self.engine.tokenize(prompt_snippet))

    # ------------------------------------------------------------------- runtime

    def run_steps(self, messages: Sequence[Dict[str, Any]]):
        """Yield chat messages emitted during the interaction loop."""

        prompt_ids = self._initialize_tokens(messages)

        for _ in range(self.environment.max_steps):
            # Ask the model for the next action and track the emitted tokens.
            action_ids, action_text = self._generate_tokens(prompt_ids)
            prompt_ids.extend(action_ids)

            action_result = self.session.step_from_text(action_text, runtime=True)

            # Emit the assistant reply and any tool observations.
            for payload in self._emit_step_messages(action_result):
                yield payload

            if action_result.terminated:
                return

            # Append tool outputs (plus the next assistant prefix) before continuing.
            prompt_ids.extend(self._tool_response_tokens(action_result.tool_messages))

        yield self._message_to_dict(
            ChatMessage(
                role="assistant",
                content="Max steps reached without final response.",
            )
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

