import torch

from typing import Any, Dict

from openrlhf_agent.agentkit.rewards import RewardPipeline
from openrlhf_agent.agentkit.session import AgentSession
from openrlhf_agent.agentkit.environments import SingleTurnEnvironment
from openrlhf_agent.agentkit.protocols import Qwen3ThinkingProtocol
from openrlhf_agent.agentkit.rewards.result_rewards import MathMatchingReward

from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor


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
    def __init__(self, *args, **kwargs):
        self.session = AgentSession(
            environment=SingleTurnEnvironment(system_prompt=TRAIN_SYSTEM_PROMPT),
            protocol=Qwen3ThinkingProtocol(),
            reward_pipeline=RewardPipeline(
                result_reward=MathMatchingReward(correct_score=1.0, miss_score=0.0)
            ),
        )

    async def reset(self, states: dict, **kwargs):
        prompt = await self.session.initialize(states.get("observation"))
        return {"observation": prompt}

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        action_text: str = states.get("action_text", "")
        label = states.get("label")

        observation, reward = await self.session.step_from_text(action_text, label=label)
        reward = float(reward) if reward is not None else 0.0

        done = True # observation.done
        return {
            "rewards": torch.tensor(reward),
            "scores": torch.tensor(reward),
            "environment_feedback": "" if done else observation.feedback_text,
            "done": done,
            "sampling_params": states.get("sampling_params", None),
            "extra_logs": {
                "dummy_scores": torch.tensor(reward),
                "turn_count": torch.tensor(observation.step_index),
            },
        }


class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)
