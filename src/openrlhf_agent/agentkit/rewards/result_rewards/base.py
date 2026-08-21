"""Base class for result rewards."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from openrlhf_agent.utils.types import Action, Message


class ResultReward(ABC):
    """Scores the final user-visible reply."""

    final_tool_name: str = "final"

    @abstractmethod
    async def score(
        self,
        *,
        action: Action,
        label: Any,
        question: Sequence[Message] = (),
    ) -> float:
        """Return the reward for the assistant's final answer."""

    def extract_final_response(self, action: Action) -> str | None:
        """Return the assistant-visible final answer from an action."""

        if not action.tool_calls:
            return action.content.strip() if action.content else None

        for tool_call in action.tool_calls:
            if tool_call.name != self.final_tool_name:
                continue

            arguments = tool_call.arguments or {}
            response = arguments.get("response")
            if isinstance(response, str) and response.strip():
                return response.strip()

        return None
