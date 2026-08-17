"""Base abstraction for result reward strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from openrlhf_agent.utils.types import Action, RewardSample


class ResultRewardStrategy(ABC):
    """Scores the final user-visible reply."""

    final_tool_name: str = "final"

    @abstractmethod
    async def score(
        self,
        *,
        action: Action,
        label: Optional[Any],
        sample: Optional[RewardSample] = None,
    ) -> float:
        """Return the reward for the assistant's final answer."""

    def extract_final_response(self, action: Action) -> Optional[str]:
        """Return the assistant-visible final answer from an action."""

        final_text = (action.content or "").strip()
        if final_text and not action.tool_calls:
            return final_text

        for tool_call in action.tool_calls or []:
            if tool_call is None:
                continue

            name = (tool_call.name or "").strip()
            if name != self.final_tool_name:
                continue

            arguments = tool_call.arguments or {}
            if not isinstance(arguments, dict):
                continue

            response = str(arguments.get("response", "")).strip()
            if response:
                return response

        return None
