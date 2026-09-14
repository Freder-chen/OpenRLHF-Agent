"""Protocol-independent state for one agent rollout."""

from __future__ import annotations

from typing import Any, Sequence

from openrlhf_agent.utils.types import (
    Action,
    Conversation,
    Message,
    Observation,
)
from openrlhf_agent.agentkit.environments import Environment
from openrlhf_agent.agentkit.rewards import RewardPipeline


class AgentSession:
    """Connect an environment, conversation history, and optional rewards."""

    def __init__(
        self,
        *,
        environment: Environment,
        reward_pipeline: RewardPipeline | None = None,
    ) -> None:
        self.environment = environment
        self.reward_pipeline = reward_pipeline

        self.history = Conversation()
        self._initial_question: list[Message] = []

    async def reset(
        self,
        question: Sequence[dict[str, Any]] | str,
    ) -> list[dict[str, Any]]:
        """Start a rollout with the environment and user question."""

        if isinstance(question, str):
            self._initial_question = [Message(role="user", content=question)]
        else:
            self._initial_question = [Message(**message) for message in question]

        self.history = Conversation(
            [*await self.environment.reset(), *self._initial_question]
        )
        return self.history.messages

    async def step(
        self,
        action: Action,
        *,
        label: Any = None,
    ) -> tuple[Observation, float | None]:
        """Apply one structured action to the environment."""

        action_message = action.to_message()
        messages, done = await self.environment.step(action)
        self.history.extend([action_message, *messages])

        reward = None
        if self.reward_pipeline is not None:
            reward = await self.reward_pipeline.score(
                action=action,
                label=label,
                done=done,
                question=self._initial_question,
            )

        return Observation(
            step_index=self.environment.step_index,
            feedback_messages=[action_message, *messages],
            done=done,
        ), reward
