from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from openrlhf_agent.environment import Environment
from openrlhf_agent.template import Template
from openrlhf_agent.utils.types import ToolCall


@dataclass
class AgentStepResult:
    observations: List[str]
    reward: float
    terminated: bool
    rendered_observation: str
    actions: Optional[List[Optional[ToolCall]]]
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
            self.environment.system_prompt,
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

    def parse_actions(self, text: str) -> Tuple[bool, Optional[List[Optional[ToolCall]]]]:
        parse_error, actions = self.template.extract_tool_calls_from_text(text)
        return parse_error, actions

    def step(
        self,
        actions: Optional[Sequence[Optional[ToolCall]]],
        *,
        label: Optional[str] = None,
        runtime: bool = False,
        parse_error: bool = False,
    ) -> AgentStepResult:
        env_actions: Optional[List[Optional[ToolCall]]] = None
        if actions is not None:
            env_actions = list(actions)
        observations, reward, terminated = self.environment.step(env_actions, label, runtime=runtime)
        return AgentStepResult(
            observations=observations,
            reward=reward,
            terminated=terminated,
            rendered_observation=self.build_user_turn(observations),
            actions=env_actions,
            parse_error=parse_error,
        )

    def step_from_text(
        self,
        action_text: str,
        *,
        label: Optional[str] = None,
        runtime: bool = False,
    ) -> AgentStepResult:
        parse_error, actions = self.parse_actions(action_text)
        if parse_error or not actions:
            return self.step(None, label=label, runtime=runtime, parse_error=True)
        return self.step(actions, label=label, runtime=runtime)
