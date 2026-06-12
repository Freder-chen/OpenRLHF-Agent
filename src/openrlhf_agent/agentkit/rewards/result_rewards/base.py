"""Base abstraction for result reward strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from openrlhf_agent.utils.types import Action, Conversation


class ResultRewardStrategy(ABC):
    """Scores the final user-visible reply."""

    @abstractmethod
    async def score(
        self,
        *,
        action: Action,
        label: Optional[Any],
        history: Optional[Conversation] = None,
    ) -> float:
        """Return the reward for the assistant's final response."""

    def extract_final_response(self, action: Action) -> Optional[str]:
        """Return the assistant-visible final response from an action."""

        final_text = (action.content or "").strip()
        if final_text and not action.tool_calls:
            return final_text

        return None
