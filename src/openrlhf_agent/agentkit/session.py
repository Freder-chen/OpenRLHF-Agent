"""Completion session used by training and text-generation inference."""

from __future__ import annotations

from typing import Any, Sequence

from openrlhf_agent.model.protocols.base import CompletionProtocol, RenderedPrompt
from openrlhf_agent.utils.types import (
    Conversation,
    Message,
    Observation,
)
from openrlhf_agent.agentkit.environments import Environment
from openrlhf_agent.agentkit.rewards import RewardPipeline


class AgentSession:
    """Connect a completion protocol, environment, history, and rewards."""

    def __init__(
        self,
        *,
        environment: Environment,
        protocol: CompletionProtocol,
        reward_pipeline: RewardPipeline | None = None,
    ) -> None:
        self.environment = environment
        self.protocol = protocol
        self.reward_pipeline = reward_pipeline

        self.history = Conversation()
        self._initial_question: list[Message] = []

    def _render_qwen_observation_suffix(
        self,
        previous_messages: Sequence[dict[str, Any]],
    ) -> RenderedPrompt:
        """Render an observation suffix for the bundled Qwen chat templates."""

        tools = self.environment.tools_manifest()
        messages = self.history.messages
        before = self.protocol.render(messages=previous_messages, tools=tools)
        after = self.protocol.render(
            messages=messages, tools=tools, add_generation_prompt=True
        )
        # Qwen stops before the template's trailing newline.
        prefix = before.text.removesuffix("\n")
        separator = before.text[len(prefix) :]
        if after.text.startswith(prefix):
            return RenderedPrompt(
                text=after.text[len(prefix) :],
                images=after.images[len(before.images) :],
            )

        feedback = self.protocol.render(
            messages=messages[len(previous_messages) :],
            add_generation_prompt=True,
        )
        feedback.text = separator + feedback.text
        return feedback

    async def reset(
        self,
        question: Sequence[dict[str, Any]] | str,
    ) -> RenderedPrompt:
        """Start a rollout with the environment and user question."""

        if isinstance(question, str):
            self._initial_question = [Message(role="user", content=question)]
        else:
            self._initial_question = [Message(**message) for message in question]

        self.history = Conversation(
            [*await self.environment.reset(), *self._initial_question]
        )
        return self.protocol.render(
            messages=self.history.messages,
            tools=self.environment.tools_manifest(),
            add_generation_prompt=True,
        )

    async def step(
        self,
        action_text: str,
        *,
        label: Any = None,
    ) -> tuple[Observation, float | None]:
        """Parse and apply one completion."""

        action = self.protocol.parse_action(action_text)
        action_message = action.to_message()
        if action.error and not action.tool_calls:
            action_message.content = action_text
            action_message.reasoning_content = None
        self.history.append(action_message)

        history_before_observation = self.history.messages
        messages, done = await self.environment.step(action)
        self.history.extend(messages)
        rendered_feedback = (
            self._render_qwen_observation_suffix(history_before_observation)
            if messages
            else RenderedPrompt(text="")
        )

        observation = Observation(
            step_index=self.environment.step_index,
            feedback_messages=[action_message, *messages],
            feedback_text=rendered_feedback.text,
            done=done,
            environment_images=rendered_feedback.images,
        )

        reward = None
        if self.reward_pipeline is not None:
            reward = await self.reward_pipeline.score(
                action=action,
                label=label,
                done=done,
                question=self._initial_question,
            )

        return observation, reward
