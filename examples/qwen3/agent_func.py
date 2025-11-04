import logging
import torch

from typing import Any, Dict
from openrlhf_agent import AgentSession, make_environment, make_template
from openrlhf.utils.agent import AgentExecutorBase, AgentInstanceBase

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AgentInstance(AgentInstanceBase):
    async def __init__(self, *args, **kwargs):
        self.step_idx = 0
        environment = make_environment(name="default")
        template = make_template("qwen3")
        self.session = AgentSession(environment, template)

    async def reset(self, states: dict, **kwargs):
        self.step_idx = 0
        self.session.initialize_prompt(states.get("observation"))

        system_prompt = self.session.build_system_prompt()
        observation = system_prompt + states["observation"]
        return {"observation": observation}

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        action_text: str = states.get("action_text", "")
        label = states.get("label")

        step_result = self.session.step_from_text(action_text, label=label)

        reward = step_result.reward
        if reward < -1:
            reward = -1.0

        self.step_idx += 1
        done = step_result.terminated

        return {
            "rewards": torch.tensor(reward),
            "scores": torch.tensor(reward),
            "environment_feedback": "" if done else step_result.rendered_observation,
            "done": done,
            "sampling_params": states.get("sampling_params", None),
            "extra_logs": {
                "dummy_scores": torch.tensor(reward),
                "turn_count": torch.tensor(self.step_idx),
            },
        }


class AgentExecutor(AgentExecutorBase):
    def __init__(self, max_steps, max_length, llm_engine, hf_tokenizer, result_queue):
        super().__init__(AgentInstance, max_steps, max_length, llm_engine, hf_tokenizer, result_queue)

    async def execute(self, prompt, label, sampling_params):
        return await super().execute(prompt, label, sampling_params)
