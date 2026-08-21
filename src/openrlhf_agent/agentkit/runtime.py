"""Run an agent against a model backend and environment."""

from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

from openrlhf_agent.agentkit.environments import Environment
from openrlhf_agent.agentkit.session import AgentSession
from openrlhf_agent.backends import ChatBackend, CompletionBackend
from openrlhf_agent.utils.types import Conversation, Message


class AgentRuntime:
    """Run inference with either a completion or chat backend."""

    def __init__(
        self,
        backend: CompletionBackend | ChatBackend,
        environment: Environment,
    ) -> None:
        self.backend = backend
        self.environment = environment

    async def _run_completion(
        self,
        backend: CompletionBackend,
        messages: Sequence[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        session = AgentSession(
            environment=self.environment,
            protocol=backend.protocol,
        )
        prompt = await session.initialize(messages)
        prompt_ids = await backend.tokenize(prompt)

        while (
            self.environment.max_steps is None
            or self.environment.step_index < self.environment.max_steps
        ):
            # Generate and append the next assistant action.
            action_ids, action_text = await backend.generate(prompt_ids)
            prompt_ids.extend(action_ids)

            # Execute the action and emit the new messages.
            observation, _ = await session.step_from_text(action_text)
            for message in observation.feedback_messages or []:
                yield message.model_dump(exclude_none=True)

            if observation.done:
                return

            # Append tool feedback before the next generation.
            if observation.feedback_text:
                feedback_ids = await backend.tokenize(observation.feedback_text)
                prompt_ids.extend(feedback_ids)

        yield Message(
            role="assistant",
            content="Max steps reached without final response.",
        ).model_dump(exclude_none=True)

    async def _run_chat(
        self,
        backend: ChatBackend,
        messages: Sequence[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        history = Conversation()
        history.reset(await self.environment.reset())
        history.extend(messages)
        tools = self.environment.tools_manifest()

        while (
            self.environment.max_steps is None
            or self.environment.step_index < self.environment.max_steps
        ):
            # Generate and record the next assistant action.
            action = await backend.generate_chat(history.messages, tools=tools)
            action_message = action.to_message()
            history.append(action_message)

            # Execute the action and record its observations.
            observation_messages, done = await self.environment.step(action)
            history.extend(observation_messages)

            for message in [action_message, *observation_messages]:
                yield message.model_dump(exclude_none=True)
            if done:
                return

        yield Message(
            role="assistant",
            content="Max steps reached without final response.",
        ).model_dump(exclude_none=True)

    async def run_steps(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield each assistant action and environment observation."""

        if isinstance(self.backend, CompletionBackend):
            runner = self._run_completion(self.backend, messages)
        elif isinstance(self.backend, ChatBackend):
            runner = self._run_chat(self.backend, messages)
        else:
            raise TypeError(f"Unsupported backend: {type(self.backend).__name__}")

        async for message in runner:
            yield message

    async def run_final(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> str | None:
        """Return the last text response produced by the assistant."""

        answer = None
        async for message in self.run_steps(messages):
            if message["role"] == "assistant" and isinstance(
                message.get("content"), str
            ):
                answer = message["content"]
        return answer
