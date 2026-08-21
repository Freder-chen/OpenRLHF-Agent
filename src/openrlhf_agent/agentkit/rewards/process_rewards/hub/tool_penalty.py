"""Process reward for calling a specific tool."""

from __future__ import annotations

from dataclasses import dataclass

from openrlhf_agent.agentkit.rewards.process_rewards.base import ProcessReward
from openrlhf_agent.utils.types import Action


@dataclass
class ToolCallPenalty(ProcessReward):
    """Apply a fixed penalty when an action calls one named tool."""

    tool_name: str
    penalty: float = -0.1

    async def score(
        self,
        *,
        action: Action,
    ) -> float:
        called = any(call.name == self.tool_name for call in action.tool_calls or [])
        return self.penalty if called else 0.0
