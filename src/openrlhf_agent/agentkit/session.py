"""Session management for the tool-using agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from openrlhf_agent.utils.types import (
    Message, Conversation,
    Action, Observation, RewardSample,
)
from openrlhf_agent.agentkit.environments import Environment
from openrlhf_agent.agentkit.protocols import ChatProtocol
from openrlhf_agent.agentkit.rewards import RewardPipeline


def has_parse_error(action: Action) -> bool:
    if action.refusal:
        return True
    return action.tool_calls and any(call.refusal for call in action.tool_calls)


class AgentSession:
    """Maintains chat history and bridges the protocol with the environment."""

    def __init__(
        self,
        *,
        environment: Environment,
        protocol: ChatProtocol,
        reward_pipeline: Optional[RewardPipeline] = None,
    ) -> None:
        self.environment = environment
        self.protocol = protocol
        self.reward_pipeline = reward_pipeline

        self.history = Conversation()
        self._initial_question: List[Message] = []

    def _parse_messages(self, payload: Optional[Union[Sequence[Dict[str, Any]], str]]) -> List[Message]:
        """Reset the chat history and optionally seed prior turns."""

        if payload is None:
            return []

        if isinstance(payload, str):
            parsed_messages = self.protocol.parse_messages_from_completion_text(payload)
            # TODO(future): parsing completion text currently keeps both the
            # environment system prompt and the one embedded in the completion,
            # so we end up with two system messages. Leave the duplication for
            # now and revisit when prompt seeding is refactored.
            return parsed_messages if parsed_messages else []

        if isinstance(payload, list):
            return [Message(**message) for message in payload]

        raise NotImplementedError

    async def initialize(self, payload: Optional[Union[Sequence[Dict[str, Any]], str]] = None) -> str:
        """Reset environment state and return the first prompt."""

        self.environment.reset_step()

        self._initial_question = self._parse_messages(payload)

        self.history.reset(system_prompt=self.environment.system_prompt)
        self.history.extend(self._initial_question)

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
        raw_text: Optional[str] = None,
    ) -> Observation:
        """Apply a parsed assistant action to the environment."""

        # Action message
        action_message = Message(
            role="assistant",
            content=action.content or None,
            tool_calls=action.tool_calls or None,
            reasoning_content=action.reasoning_content or None,
        )
        parse_error = has_parse_error(action)
        if parse_error and not action.tool_calls and raw_text is not None:
            # Preserve the unparsed text so the user can see what went wrong.
            action_message.content = raw_text
            action_message.reasoning_content = None
        self.history.append(action_message)

        # Observation messages
        obs_list, done = await self.environment.step(action)

        obs_messages = [Message(role="tool", content=obs) for obs in obs_list]
        if obs_messages:
            tool_payload = [m.model_dump(exclude_none=True) for m in obs_messages]
            feedback_text = self.protocol.render_messages(
                messages=tool_payload,
                add_generation_prompt=True,
            )
        else:
            feedback_text = ""

        # Make observation
        observation = Observation(
            step_index=self.environment.step_index,
            feedback_messages=[action_message, *obs_messages], # for runtime, with action
            feedback_text=feedback_text,  # for train, without action
            done=done,
        )

        # Reward action
        reward = None
        if label is not None and self.reward_pipeline:
            reward = await self.reward_pipeline.score(
                action=action,
                label=label,
                done=done,
                sample=RewardSample(
                    question=self._initial_question,
                    process_messages=self.history.messages[len(self._initial_question):], # filter system + input
                ),
            )

        return observation, reward

    async def step_from_text(
        self,
        action_text: str,
        *,
        label: Optional[Any] = None,
    ) -> Observation:
        """Parse a raw model response and forward to `step`."""

        parsed_action = self.protocol.parse_assistant_text(action_text)
        return await self.step(
            parsed_action,
            label=label,
            raw_text=action_text,
        )
