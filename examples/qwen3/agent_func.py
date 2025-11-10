import logging
import torch

from typing import Any, Dict
from openrlhf_agent import AgentSession, make_environment, make_chat_protocol
from openrlhf.utils.agent import AgentExecutorBase, AgentInstanceBase

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AgentInstance(AgentInstanceBase):
    def __init__(self, *args, **kwargs):
        environment = make_environment(name="default")
        protocol = make_chat_protocol("qwen3_instruct")
        self.session = AgentSession(environment, protocol)

    async def reset(self, states: dict, **kwargs):
        prompt = self.session.initialize(states.get("observation"))
        return {"observation": prompt}

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        action_text: str = states.get("action_text", "")
        label = states.get("label")

        step_outcome = self.session.step_from_text(action_text, label=label)

        reward = step_outcome.reward
        if reward < -1:
            reward = -1.0

        done = step_outcome.terminated
        step_idx = step_outcome.step_index

        return {
            "rewards": torch.tensor(reward),
            "scores": torch.tensor(reward),
            "environment_feedback": "" if done else step_outcome.feedback_text,
            "done": done,
            "sampling_params": states.get("sampling_params", None),
            "extra_logs": {
                "dummy_scores": torch.tensor(reward),
                "turn_count": torch.tensor(step_idx),
            },
        }


class AgentExecutor(AgentExecutorBase):
    def __init__(self, max_steps, max_length, llm_engine, hf_tokenizer, result_queue):
        super().__init__(AgentInstance, max_steps, max_length, llm_engine, hf_tokenizer, result_queue)
