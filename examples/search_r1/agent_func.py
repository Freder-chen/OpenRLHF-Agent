import logging
from datetime import datetime
import torch
from typing import Any, Dict

from openrlhf_agent.agentkit.rewards import RewardPipeline
from openrlhf_agent.agentkit.session import AgentSession
from openrlhf_agent.agentkit.factory import (
    build_environment,
    build_protocol,
    build_result_reward,
)
from openrlhf_agent.agentkit.tools import CommentaryTool, LocalSearchTool

from openrlhf.utils.agent import AgentExecutorBase, AgentInstanceBase

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


CUSTOM_SYSTEM_PROMPT = """
Answer the given question. First, think step by step inside <think> and </think> whenever you receive new information. 
After reasoning, decide whether to use tools. Use tools to verify specific aspects of your reasoning or to fetch missing knowledge; 
do not rely on tools to write the final answer. Call the commentary tool only for brief progress updates.

If the conditions for solving the problem have been met, directly provide the final answer inside <final> and </final> without extra illustrations. 
Example: <final> ... </final>.

Knowledge cutoff: 2023-06
Current date: {date}
""".strip()


class AgentInstance(AgentInstanceBase):
    def __init__(self, *args, **kwargs):
        environment = build_environment(
            name="function_call",
            tools=[CommentaryTool(), LocalSearchTool()],
            system_prompt=CUSTOM_SYSTEM_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d")),
        )
        protocol = build_protocol(name="qwen3_thinking")
        pipeline = RewardPipeline(
            result_reward=build_result_reward(
                name="matching",
                config=dict(
                    correct_score=1.0,
                    miss_score=0.0,
                )
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
        if reward < -1:
            reward = -1.0

        done = observation.done
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


class AgentExecutor(AgentExecutorBase):
    def __init__(self, max_length, llm_engine, hf_tokenizer):
        super().__init__(AgentInstance, max_length, llm_engine, hf_tokenizer)
