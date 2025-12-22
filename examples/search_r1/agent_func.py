import logging
from datetime import datetime
import torch
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from typing import Any, Dict

from openrlhf_agent.agentkit.rewards import RewardPipeline
from openrlhf_agent.agentkit.session import AgentSession
from openrlhf_agent.agentkit.factory import (
    build_environment,
    build_protocol,
    build_process_reward,
    build_result_reward,
)
from openrlhf_agent.agentkit.tools.hub.commentary import CommentaryTool
from search_tool import LocalSearchTool

from openrlhf.utils.agent import AgentExecutorBase, AgentInstanceBase

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


CUSTOM_SYSTEM_PROMPT = """
Answer the given question. First, think step by step inside <think> and </think> whenever you receive new information. After reasoning, decide whether to use tools. Use tools to verify specific aspects of your reasoning or to fetch missing knowledge; do not rely on tools to write the final answer. Call the commentary tool only for brief progress updates.

If the conditions for solving the problem have been met, directly provide the final answer inside <answer> and </answer> without extra illustrations. Example: <answer> ... </answer>.

Knowledge cutoff: 2023-06
Current date: {date}
""".strip().format(date=datetime.now().strftime("%Y-%m-%d"))


class AgentInstance(AgentInstanceBase):
    def __init__(self, *args, **kwargs):
        environment = build_environment(
            name="function_call",
            tools=[CommentaryTool(), LocalSearchTool()],
            system_prompt=CUSTOM_SYSTEM_PROMPT,
        )
        protocol = build_protocol(name="qwen3_thinking")
        pipeline = RewardPipeline(
            process_reward=build_process_reward(
                name="tool_call",
                config=dict(
                    parse_error_penalty=-0.2,
                    penalty_for_refused=-0.1,
                    tool_policies={
                        "commentary": dict(
                            max_calls=1,
                            reward_per_call=0.1,
                            overuse_penalty=-0.1,
                        ),
                    },
                )
            ),
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
