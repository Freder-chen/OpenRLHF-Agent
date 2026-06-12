"""Environment that ends after one plain-text assistant reply."""

from __future__ import annotations

from typing import List, Optional, Tuple

from openrlhf_agent.utils.types import Action
from openrlhf_agent.agentkit.environments.base import Environment


DEFAULT_SINGLE_TURN_PROMPT = """
You are a helpful assistant.
""".strip()


class SingleTurnEnvironment(Environment):
    """Minimal environment that accepts only one assistant reply."""

    def __init__(
        self,
        *,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(
            system_prompt=system_prompt or DEFAULT_SINGLE_TURN_PROMPT,
        )

    async def step(self, action: Action) -> Tuple[List[str], bool]:
        self._step_index += 1
        return [], True
