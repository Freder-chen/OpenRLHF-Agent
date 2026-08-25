"""OpenRLHF adapter for the Qwen3 math environment."""

from typing import Any

import torch
from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor

from openrlhf_agent.agentkit import AgentSession
from openrlhf_agent.agentkit.environments import SingleTurnEnvironment
from openrlhf_agent.agentkit.rewards import RewardPipeline
from openrlhf_agent.agentkit.rewards.result_rewards import MathMatchingReward
from openrlhf_agent.model import Qwen3Protocol


TRAIN_SYSTEM_PROMPT = """
You are a helpful assistant operating in TRAINING mode.

Rules:
1. Before finishing, verify both the answer and the exact boxed-answer format.

## Output Rules
- First provide a clear markdown explanation of the solution.
- Then end exactly with:
  `Answer: \\boxed{<final_answer>}`
- The boxed expression must contain only the final answer in canonical form.
- Do not add any text after the boxed answer.
""".strip()


class AgentInstance(AgentInstanceBase):
    """One isolated math rollout."""

    def __init__(self, *args, **kwargs):
        self.session = AgentSession(
            environment=SingleTurnEnvironment(system_prompt=TRAIN_SYSTEM_PROMPT),
            protocol=Qwen3Protocol(enable_thinking=True),
            reward_pipeline=RewardPipeline(
                result_rewards=[MathMatchingReward(correct_score=1.0, miss_score=0.0)]
            ),
        )

    async def reset(self, states: dict[str, Any], **kwargs) -> dict[str, str]:
        prompt = await self.session.reset(states["observation"])
        return {"observation": prompt.text}

    async def step(self, states: dict[str, Any], **kwargs) -> dict[str, Any]:
        observation, reward = await self.session.step(
            states["action_text"],
            label=states["label"],
        )
        reward = float(reward or 0.0)

        return {
            "rewards": torch.tensor(reward),
            "scores": torch.tensor(reward),
            "environment_feedback": (
                "" if observation.done else observation.feedback_text
            ),
            "done": observation.done,
            "extra_logs": {
                "turn_count": torch.tensor(observation.step_index),
            },
        }


class AgentExecutor(MultiTurnAgentExecutor):
    """Entrypoint loaded by OpenRLHF rollout workers."""

    def __init__(self):
        super().__init__(AgentInstance)
