"""Streaming runtime loop for the tool-using agent."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional, Sequence

from openrlhf_agent.utils.types import Status
from openrlhf_agent.backends import LLMEngine
from openrlhf_agent.agentkit.session import AgentSession


class AgentRuntime:
    """Coordinates the language model with the environment at inference time.

    Owns the token-level state (prompt_ids). When ``max_context_tokens`` is
    set and the session supports ``request_compact()``, the runtime triggers
    compaction: appends the compact instruction, lets the model generate a
    summary, then re-initializes the session.
    """

    def __init__(
        self,
        engine: LLMEngine,
        session: AgentSession,
        *,
        max_new_tokens_per_step: int = 10240,
        max_context_tokens: Optional[int] = None,
    ) -> None:
        self.engine = engine
        self.session = session
        self.max_new_tokens_per_step = max_new_tokens_per_step
        self.max_context_tokens = max_context_tokens

    async def run_steps(self, messages: Sequence[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        prompt_text = await self.session.initialize(messages)
        prompt_ids = await self.engine.tokenize(prompt_text)

        while True:
            action_ids, action_text = await self.engine.generate(
                prompt_ids,
                max_tokens=self.max_new_tokens_per_step,
            )

            observation, _ = await self.session.step_from_text(action_text)
            for message in observation.feedback_messages or []:
                yield message.model_dump(exclude_none=True)

            if observation.status == Status.DONE:
                return

            prompt_ids.extend(
                action_ids + await self.engine.tokenize(observation.feedback_text)
            )

            # Compact: append instruction, model generates summary, re-initialize.
            if (
                self.max_context_tokens is not None
                and len(prompt_ids) > self.max_context_tokens
                and hasattr(self.session, 'request_compact')
            ):
                compact_feedback = self.session.request_compact()
                _, summary_text = await self.engine.generate(
                    prompt_ids + await self.engine.tokenize(compact_feedback),
                    max_tokens=self.max_new_tokens_per_step,
                )
                new_prompt = await self.session.finish_compact(summary_text)
                prompt_ids = await self.engine.tokenize(new_prompt)

    async def run_final(self, messages: Sequence[Dict[str, Any]]) -> Optional[str]:
        final_text: Optional[str] = None
        async for msg in self.run_steps(messages):
            if msg.get("role") == "assistant" and not msg.get("tool_calls"):
                final_text = msg.get("content") or final_text
        return final_text
