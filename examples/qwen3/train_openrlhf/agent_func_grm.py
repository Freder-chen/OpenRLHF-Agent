"""OpenRLHF integration example that scores rollouts with the GRM judge."""

from typing import Any, Dict

import torch

from openrlhf_agent.agentkit.rewards import RewardPipeline
from openrlhf_agent.agentkit.session import AgentSession
from openrlhf_agent.utils.types import Status
from openrlhf_agent.agentkit.environments import FunctionCallEnvironment
from openrlhf_agent.agentkit.protocols import Qwen3ThinkingProtocol
from openrlhf_agent.agentkit.rewards.result_rewards import GRMJudgeReward

from openrlhf.utils.agent import MultiTurnAgentExecutor, AgentInstanceBase


class AgentInstance(AgentInstanceBase):
    """Exact same agent wiring as examples/qwen3/agent_func.py, but GRM-scored."""

    def __init__(self, *_, **__):
        environment = FunctionCallEnvironment()
        protocol = Qwen3ThinkingProtocol()
        pipeline = RewardPipeline(result_reward=GRMJudgeReward(
            model="qwen3",
            base_url="http://0.0.0.0:8009/v1",
            api_key="empty",
            correct_score=1.0,
            format_score=0.0,
            error_score=-0.1,
        ))
        self.session = AgentSession(
            environment=environment,
            protocol=protocol,
            reward_pipeline=pipeline,
        )

    async def reset(self, states: dict, **_) -> Dict[str, Any]:
        prompt = await self.session.initialize(states.get("observation"))
        return {"observation": prompt}

    async def step(self, states: dict, **_) -> Dict[str, Any]:
        action_text: str = states.get("action_text", "")
        label = states.get("label")

        observation, reward = await self.session.step_from_text(action_text, label=label)

        reward = float(reward) if reward is not None else 0.0
        reward = max(reward, -1.0)

        done = observation.status == Status.DONE
        return {
            "rewards": torch.tensor(reward),
            "scores": torch.tensor(reward),
            "environment_feedback": "" if done else observation.feedback_text,
            "done": done,
            "sampling_params": states.get("sampling_params"),
            "extra_logs": {
                "dummy_scores": torch.tensor(reward),
                "turn_count": torch.tensor(self.session.step_index),
            },
        }


class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)
