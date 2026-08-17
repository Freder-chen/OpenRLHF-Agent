"""Reward pipeline orchestrating process/result strategies."""

from __future__ import annotations

from typing import Any, Optional

from openrlhf_agent.utils.types import Action, RewardSample

from .process_rewards.base import ProcessRewardStrategy
from .result_rewards.base import ResultRewardStrategy


class RewardPipeline:
    """Combines optional process/result reward components."""

    def __init__(
        self,
        *,
        process_reward: Optional[ProcessRewardStrategy] = None,
        result_reward: Optional[ResultRewardStrategy] = None,
    ) -> None:
        self._process_reward = process_reward
        self._result_reward = result_reward

        assert (
            self._process_reward is not None or self._result_reward is not None
        ), "RewardPipeline requires at least one reward strategy"

    async def score(
        self,
        *,
        action: Action,
        label: Optional[Any],
        done: bool,
        sample: Optional[RewardSample] = None,
    ) -> float:
        """Compute a scalar reward for the latest action."""

        reward = 0.0

        if self._process_reward and not done:
            reward += await self._process_reward.score(action=action, label=label)

        if self._result_reward and done:
            reward += await self._result_reward.score(action=action, label=label, sample=sample)

        return reward
