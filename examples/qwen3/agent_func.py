import torch

from typing import Any, Dict

from openrlhf_agent.agentkit.rewards import RewardPipeline, ToolCallReward
from openrlhf_agent.agentkit.session import AgentSession
from openrlhf_agent.utils.types import Status
from openrlhf_agent.agentkit.environments import FunctionCallEnvironment
from openrlhf_agent.agentkit.protocols import Qwen3ThinkingProtocol
from openrlhf_agent.agentkit.rewards.result_rewards import MatchingReward

from openrlhf.utils.agent import MultiTurnAgentExecutor, AgentInstanceBase


class AgentInstance(AgentInstanceBase):
    def __init__(self, *args, **kwargs):
        environment = FunctionCallEnvironment()
        protocol = Qwen3ThinkingProtocol()
        pipeline = RewardPipeline(
            process_reward=ToolCallReward(
                parse_error_penalty=-0.2,
                penalty_for_refused=-0.1,
                tool_policies={
                    "commentary": dict(
                        max_calls=1,
                        reward_per_call=0.1,
                        overuse_penalty=-0.1,
                    ),
                },
            ),
            result_reward=MatchingReward(
                correct_score=1.0,
                miss_score=0.0,
            ),
        )
        self.session = AgentSession(environment=environment, protocol=protocol, reward_pipeline=pipeline)

    async def reset(self, states: dict, **kwargs):
        prompt = await self.session.initialize(states.get("observation"))
        return {"observation": prompt}

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
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
            "sampling_params": states.get("sampling_params", None),
            "extra_logs": {
                "dummy_scores": torch.tensor(reward),
                "turn_count": torch.tensor(self.session.step_index),
            },
        }


class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)
