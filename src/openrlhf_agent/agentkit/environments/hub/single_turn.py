"""Environment that ends after one plain-text assistant reply."""

from __future__ import annotations

from openrlhf_agent.utils.types import Action
from openrlhf_agent.agentkit.environments.base import Environment


DEFAULT_PROMPT = """
You are a helpful assistant.
""".strip()


class SingleTurnEnvironment(Environment):
    """Minimal environment that accepts only one assistant reply."""

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(
            system_prompt=system_prompt if system_prompt is not None else DEFAULT_PROMPT,
            max_steps=1,
        )

    async def step(self, action: Action) -> tuple[list[str], bool]:
        self.step_index += 1

        return [], True
