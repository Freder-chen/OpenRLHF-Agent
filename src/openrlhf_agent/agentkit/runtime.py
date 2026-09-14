"""Run an agent against a model backend and environment."""

from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

from openrlhf_agent.agentkit.environments import Environment
from openrlhf_agent.agentkit.session import AgentSession
from openrlhf_agent.model.backends.base import ActionBackend, CompletionBackend


class AgentRuntime:
    """Run inference with either a completion or action backend."""

    def __init__(
        self,
        backend: CompletionBackend | ActionBackend,
        environment: Environment,
        *,
        max_tokens: int | None = None,
    ) -> None:
        self.backend = backend
        self.environment = environment
        self.max_tokens = max_tokens

    async def _run_completion(
        self,
        backend: CompletionBackend,
        messages: Sequence[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        session = AgentSession(
            environment=self.environment,
        )

        await session.reset(messages)
        tools = self.environment.tools_manifest()
        rendered = backend.render_prompt(
            messages=session.history.messages,
            tools=tools,
        )
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

            observation, _ = await session.step(backend.parse_action(result.text))
            for message in observation.feedback_messages:
                yield message.model_dump(exclude_none=True)
            if observation.done:
                return

            feedback = backend.render_feedback(
                messages=session.history.messages,
                environment_messages=[
                    message.model_dump(exclude_none=True)
                    for message in observation.environment_messages
                ],
                tools=tools,
            )
            token_ids.extend(
                await backend.tokenize(feedback.text, add_special_tokens=False)
            )
            images.extend(feedback.images)

    async def _run_action(
        self,
        backend: ActionBackend,
        messages: Sequence[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        session = AgentSession(environment=self.environment)
        await session.reset(messages)
        while True:
            action = await backend.generate(
                session.history.messages,
                tools=self.environment.tools_manifest(),
                max_tokens=self.max_tokens,
            )
            observation, _ = await session.step(action)
            for message in observation.feedback_messages:
                yield message.model_dump(exclude_none=True)
            if observation.done:
                return

    async def run_steps(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield each assistant action and environment observation."""

        if isinstance(self.backend, CompletionBackend):
            runner = self._run_completion(self.backend, messages)
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
