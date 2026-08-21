"""Completion session used by training and text-generation inference."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from openrlhf_agent.backends.openai.vllm.protocols import Protocol
from openrlhf_agent.utils.types import Action, Conversation, Message, Observation
from openrlhf_agent.agentkit.environments import Environment
from openrlhf_agent.agentkit.rewards import RewardPipeline


class AgentSession:
    """Connect a completion protocol, environment, history, and rewards."""

    def __init__(
        self,
        *,
        environment: Environment,
        protocol: Protocol,
        reward_pipeline: Optional[RewardPipeline] = None,
    ) -> None:
        self.environment = environment
        self.protocol = protocol
        self.reward_pipeline = reward_pipeline

        self.history = Conversation()
        self._initial_question: list[Message] = []

    async def initialize(
        self,
        question: Sequence[dict[str, Any]] | str,
    ) -> str:
        """Start a rollout with the environment and user question."""

        if isinstance(question, str):
            self._initial_question = [Message(role="user", content=question)]
        else:
            self._initial_question = [Message(**message) for message in question]

        self.history.reset(await self.environment.reset())
        self.history.extend(self._initial_question)

        return self.protocol.render(
            messages=self.history.messages,
            tools=self.environment.tools_manifest(),
            add_generation_prompt=True,
        )

    async def step(
        self,
        action: Action,
        *,
        label: Optional[Any] = None,
        raw_text: Optional[str] = None,
    ) -> tuple[Observation, Optional[float]]:
        """Apply a parsed assistant action to the environment."""

        # Action message
        action_message = action.to_message()
        action_error = action.error or any(call.error for call in action.tool_calls or [])
        if action_error and not action.tool_calls and raw_text is not None:
            # Preserve the unparsed text so the user can see what went wrong.
            action_message.content = raw_text
            action_message.reasoning_content = None
        self.history.append(action_message)

        # Observation messages
        obs_messages, done = await self.environment.step(action)
        self.history.extend(obs_messages)
        if obs_messages:
            # TODO: Pass multimodal observations through completion training.
            if any(isinstance(message.content, list) for message in obs_messages):
                raise TypeError("Completion sessions only support text observations")

            feedback_text = self.protocol.render(
                messages=[message.model_dump(exclude_none=True) for message in obs_messages],
                add_generation_prompt=True,
            )
        else:
            feedback_text = ""

        # Make observation
        observation = Observation(
            step_index=self.environment.step_index,
            feedback_messages=[action_message, *obs_messages],  # for runtime, with action
            feedback_text=feedback_text,  # for train, without action
            done=done,
        )

        # Reward action
        reward = None
        if self.reward_pipeline:
            reward = await self.reward_pipeline.score(
                action=action,
                label=label,
                done=done,
                question=self._initial_question,
            )

        return observation, reward

    async def step_from_text(
        self,
        action_text: str,
        *,
        label: Optional[Any] = None,
    ) -> tuple[Observation, Optional[float]]:
        """Parse a raw model response and forward to `step`."""

        parsed_action = self.protocol.parse_action(action_text)
        return await self.step(
            parsed_action,
            label=label,
            raw_text=action_text,
        )
