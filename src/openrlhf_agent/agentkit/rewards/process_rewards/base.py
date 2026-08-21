"""Base class for process rewards."""

from __future__ import annotations

from abc import ABC, abstractmethod

from openrlhf_agent.utils.types import Action


class ProcessReward(ABC):
    """Scores intermediate planning/tool steps."""

    @abstractmethod
    async def score(
        self,
        *,
        action: Action,
    ) -> float:
        """Return the reward associated with the latest tool usage."""
