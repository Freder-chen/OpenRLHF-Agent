import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .environment import Environment
from .model import LLMEngine
from .template import Template
from .types import ParsedAssistantMessage, ToolCall


@dataclass
class AgentStepResult:
    observations: List[str]
    reward: float
    terminated: bool
    rendered_observation: str
    actions: Optional[List[Optional[ToolCall]]]
    final_response: Optional[str] = None
    parse_error: bool = False


class AgentSession:
    """Shared logic for applying tool calls in both training and inference."""

    def __init__(self, environment: Environment, template: Template) -> None:
        self.environment = environment
        self.template = template

    def reset(self, initial_observation: Optional[str] = None) -> None:
        self.environment.reset(initial_observation)

    def build_system_prompt(self) -> str:
        return self.template.render_system(
            text=self.environment.system_prompt,
            tools_manifest=self.environment.tools_manifest(),
        )

    def render_observations(self, observations: Sequence[str]) -> str:
        blocks = [self.template.render_tool_response(obs) for obs in observations]
        return "\n".join(blocks)

    def build_user_turn(self, observations: Sequence[str]) -> str:
        return self.template.render_turn(
            role="user",
            text=self.render_observations(observations),
            add_generation_prompt=True,
        )

    def parse_actions(self, text: str) -> ParsedAssistantMessage:
        return self.template.parse_assistant_message(text)

    def step(
        self,
        actions: Optional[Sequence[ToolCall | None]],
        *,
        final_response: Optional[str] = None,
        label: Optional[str] = None,
        runtime: bool = False,
        parse_error: bool = False,
    ) -> AgentStepResult:
        actions_list = list(actions) if actions is not None else None
        observations, reward, terminated, resolved_final = self.environment.step(
            actions_list,
            final_response,
            label,
            runtime=runtime,
        )
        return AgentStepResult(
            observations=observations,
            reward=reward,
            terminated=terminated,
            rendered_observation=self.build_user_turn(observations),
            actions=actions_list,
            final_response=resolved_final,
            parse_error=parse_error,
        )

    def step_from_text(
        self,
        action_text: str,
        *,
        label: Optional[str] = None,
        runtime: bool = False,
    ) -> AgentStepResult:
        parsed = self.parse_actions(action_text)

        if parsed.parse_error and not parsed.tool_calls and parsed.final_response is None:
            return self.step(None, label=label, runtime=runtime, parse_error=True)

        actions: Optional[Sequence[ToolCall | None]]
        if parsed.tool_calls:
            actions = parsed.tool_calls
        else:
            actions = []

        return self.step(
            actions,
            final_response=parsed.final_response,
            label=label,
            runtime=runtime,
            parse_error=parsed.parse_error,
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
    ):
        self.engine = engine
        self.session = AgentSession(environment, template)
        self.environment = self.session.environment
        self.template = self.session.template
        self.max_new_tokens_per_step = max_new_tokens_per_step

    @staticmethod
    def _is_internal_obs(text: str) -> bool:
        """Return True if the observation should stay hidden from the user."""
        try:
            data = json.loads(text)
            return isinstance(data, dict) and (data.get("__internal") is True or data.get("visible_to_user") is False)
        except Exception:
            return False

    def run_steps(self, messages: List[Dict[str, str]]):
        """Stream messages that mirror the openai-python Responses format."""
        self.session.reset()

        prompt = self.template.render_messages(
            messages=[{"role": "system", "content": self.environment.system_prompt}, *messages],
            tools_manifest=self.environment.tools_manifest(),
            add_generation_prompt=True,
        )
        prompt_ids = self.engine.tokenize(prompt)

        for _ in range(self.environment.max_steps):
            action_ids, action_text = self.engine.generate(
                prompt_ids, max_tokens=self.max_new_tokens_per_step
            )

            step_result = self.session.step_from_text(action_text, runtime=True)

            if step_result.parse_error and not step_result.actions:
                # Surface the raw model text so downstream logs can inspect it.
                yield {
                    "role": "assistant",
                    "content": action_text,
                }
                observation_ids = self.engine.tokenize(step_result.rendered_observation)
                prompt_ids += action_ids + observation_ids
                continue

            actions = step_result.actions or []
            tool_calls: List[Dict[str, Any]] = []
            for action in actions:
                if action is None:
                    continue
                tool_calls.append(
                    {
                        "id": action.id,
                        "type": "function",
                        "function": {
                            "name": action.name,
                            "arguments": action.arguments or "{}",
                        },
                    }
                )

            if tool_calls:
                yield {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                }

            for idx, obs in enumerate(step_result.observations):
                action = actions[idx] if idx < len(actions) else None
                if action is None:
                    continue
                if not self._is_internal_obs(obs):
                    yield {
                        "role": "tool",
                        "tool_call_id": action.id,
                        "content": obs,
                    }

            observation_ids = self.engine.tokenize(step_result.rendered_observation)
            prompt_ids += action_ids + observation_ids

            if step_result.terminated:
                final_text = step_result.final_response or ""
                yield {
                    "role": "assistant",
                    "content": final_text,
                }
                return

        yield {
            "role": "assistant",
            "content": "Max steps reached without final.",
        }

    def run_final(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Convenience wrapper: returns the final assistant content or None.
        """
        final_text: Optional[str] = None
        for step in self.run_steps(messages):
            if step.get("role") != "assistant":
                continue
            if "tool_calls" in step:
                continue
            content = step.get("content")
            if isinstance(content, str):
                final_text = content
        return final_text


__all__ = ["AgentRuntime", "AgentSession", "AgentStepResult"]
