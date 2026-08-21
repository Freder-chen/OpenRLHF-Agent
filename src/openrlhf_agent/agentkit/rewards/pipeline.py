"""Reward pipeline orchestrating process/result strategies."""

from __future__ import annotations

from typing import Any, Sequence

from openrlhf_agent.utils.types import Action, Message

from .process_rewards.base import ProcessReward
from .result_rewards.base import ResultReward


class RewardPipeline:
    """Combine process and result rewards for one rollout."""

    def __init__(
        self,
        *,
        process_rewards: Sequence[ProcessReward] = (),
        result_rewards: Sequence[ResultReward] = (),
    ) -> None:
        if not process_rewards and not result_rewards:
            raise ValueError("RewardPipeline requires at least one reward")
        self.process_rewards = list(process_rewards)
        self.result_rewards = list(result_rewards)

    async def score(
        self,
        *,
        action: Action,
        label: Any,
        done: bool,
        question: Sequence[Message] = (),
    ) -> float:
        """Compute a scalar reward for the latest action."""

        if label is None:
            raise ValueError("label is required to score a trajectory")

        if done:
            scores = [
                await reward.score(action=action, label=label, question=question)
                for reward in self.result_rewards
            ]
        else:
            scores = [
                await reward.score(action=action)
                for reward in self.process_rewards
            ]

        return sum(scores)
