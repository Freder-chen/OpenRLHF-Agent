"""Lightweight tool process reward: penalize parse errors and invalid tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from openrlhf_agent.agentkit.rewards.process_rewards.base import ProcessRewardStrategy
from openrlhf_agent.utils.types import Action


@dataclass
class ToolFormatReward(ProcessRewardStrategy):
    """Penalize an action if it fails to parse or contains any invalid tool call."""

    penalty: float = -0.1

    async def score(
        self,
        *,
        action: Action,
        label: Optional[Any],
    ) -> float:
        """Return a fixed penalty on any format error, else 0."""

        invalid = action.refusal or any(
            call is None or call.refusal or not (call.name or "").strip()
            for call in action.tool_calls or []
        )
        return self.penalty if invalid else 0.0
