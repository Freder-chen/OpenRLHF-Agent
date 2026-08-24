"""Run an agent against a model backend and environment."""

from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

from openrlhf_agent.agentkit.environments import Environment
from openrlhf_agent.agentkit.session import AgentSession
from openrlhf_agent.model.backends.base import ActionBackend, CompletionBackend
from openrlhf_agent.model.protocols.base import CompletionProtocol
from openrlhf_agent.utils.types import Conversation


class AgentRuntime:
    """Run inference with either a completion or action backend."""

    def __init__(
        self,
        backend: CompletionBackend | ActionBackend,
        environment: Environment,
        *,
        protocol: CompletionProtocol | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.backend = backend
        self.environment = environment
        self.protocol = protocol
        self.max_tokens = max_tokens

    async def _run_completion(
        self,
        backend: CompletionBackend,
        protocol: CompletionProtocol,
        messages: Sequence[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        session = AgentSession(
            environment=self.environment,
            protocol=protocol,
        )

        rendered = await session.reset(messages)
        token_ids = await backend.tokenize(
            rendered.text,
            add_special_tokens=False,
        )
        images = rendered.images
        while True:
            result = await backend.generate(
                token_ids,
                max_tokens=self.max_tokens,
                images=images,
            )
            token_ids.extend(result.token_ids)

            observation, _ = await session.step(result.text)
            for message in observation.feedback_messages:
                yield message.model_dump(exclude_none=True)
            if observation.done:
                return
            token_ids.extend(
                await backend.tokenize(
                    observation.feedback_text,
                    add_special_tokens=False,
                )
            )
            images.extend(observation.environment_images)

    async def _run_action(
        self,
        backend: ActionBackend,
        messages: Sequence[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        history = Conversation([*await self.environment.reset(), *messages])
        tools = self.environment.tools_manifest()

        while True:
            action = await backend.generate(
                history.messages,
                tools=tools,
                max_tokens=self.max_tokens,
            )
            action_message = action.to_message()
            history.append(action_message)

            observation_messages, done = await self.environment.step(action)
            history.extend(observation_messages)

            for message in [action_message, *observation_messages]:
                yield message.model_dump(exclude_none=True)
            if done:
                return

    async def run_steps(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield each assistant action and environment observation."""

        if isinstance(self.backend, CompletionBackend):
            if self.protocol is None:
                raise ValueError("CompletionBackend requires a CompletionProtocol.")
            runner = self._run_completion(self.backend, self.protocol, messages)
        else:
            runner = self._run_action(self.backend, messages)

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
