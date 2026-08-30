"""Use OpenRLHF-Agent's math checker as VIME's reward function."""

from typing import Any

from openrlhf_agent.agentkit.rewards.result_rewards import MathMatchingReward


reward = MathMatchingReward(correct_score=1.0, format_score=0.1, miss_score=0.0)


async def reward_func(args: Any, sample: Any, **kwargs: Any) -> float:
    del args, kwargs
    if sample.response is None:
        return 0.0
    return float(reward.score_response(sample.response, sample.label))
