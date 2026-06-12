"""Core agent session — one continuous segment of conversation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

from openrlhf_agent.utils.types import (
    Message, Conversation,
    Action, Status, Observation,
)
from openrlhf_agent.agentkit.environments import Environment
from openrlhf_agent.agentkit.protocols import ChatProtocol
from openrlhf_agent.agentkit.rewards import RewardPipeline


def has_parse_error(action: Action) -> bool:
    if action.refusal:
        return True
    return action.tool_calls and any(call.refusal for call in action.tool_calls)


class AgentSession:
    """One continuous segment of conversation. Pure message-level state."""

    def __init__(
        self,
        *,
        environment: Environment,
        protocol: ChatProtocol,
        reward_pipeline: Optional[RewardPipeline] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        self.environment = environment
        self.protocol = protocol
        self.reward_pipeline = reward_pipeline
        self.max_steps = max_steps

        self.history: Optional[Conversation] = None
        self.step_index = 0

    async def initialize(self, payload: Optional[Union[Sequence[Dict[str, Any]], str]] = None) -> str:
        self.step_index = 0

        # Parse payload into Message objects.
        if payload is None:
            messages = []
        elif isinstance(payload, str):
            messages = self.protocol.parse_messages_from_completion_text(payload) or []
        elif isinstance(payload, list):
            messages = [Message(**m) for m in payload]
        else:
            raise NotImplementedError

        self.history = Conversation(
            system_prompt=self.environment.system_prompt,
            messages=messages,
        )
        return self.protocol.render_messages(
            messages=self.history.messages,
            tools_manifest=self.environment.tools_manifest(),
            add_generation_prompt=True,
        )

    async def step(
        self,
        action: Action,
        *,
        label: Optional[Any] = None,
    ) -> Tuple[Observation, Optional[float]]:
        # Environment step
        obs_list, done = await self.environment.step(action)

        # Step counting
        self.step_index += 1
        if self.max_steps is not None and self.step_index >= self.max_steps:
            done = True

        # Prepare feedback message
        action_message = Message(
            role="assistant",
            content=action.content or None,
            tool_calls=action.tool_calls or None,
            reasoning_content=action.reasoning_content or None,
        )
        obs_messages = [Message(role="tool", content=obs) for obs in obs_list]
        self.history.extend([action_message, *obs_messages])

        # Prepare feedback text
        feedback_text = self.protocol.render_messages(
            messages=[m.model_dump(exclude_none=True) for m in obs_messages],
            add_generation_prompt=True,
        ) if obs_messages else ""

        observation = Observation(
            feedback_messages=[action_message, *obs_messages],
            feedback_text=feedback_text,
            status=Status.DONE if done else Status.CONTINUE,
        )

        # Reward
        reward = None
        if label is not None and self.reward_pipeline:
            reward = await self.reward_pipeline.score(
                action=action,
                label=label,
                done=done,
                history=self.history,
            )

        return observation, reward

    async def step_from_text(
        self,
        action_text: str,
        *,
        label: Optional[Any] = None,
    ) -> Tuple[Observation, Optional[float]]:
        parsed_action = self.protocol.parse_assistant_text(action_text)
        # On parse error with no tool calls, preserve the raw text in the action.
        if has_parse_error(parsed_action) and not parsed_action.tool_calls:
            parsed_action.content = action_text
            parsed_action.reasoning_content = None
        return await self.step(parsed_action, label=label)
