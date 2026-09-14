import os
import torch
from typing import Any, Dict

from openrlhf_agent.agentkit.rewards import RewardPipeline
from openrlhf_agent.agentkit import AgentSession
from openrlhf_agent.agentkit.environments import FunctionCallEnvironment
from openrlhf_agent.model import Qwen3Protocol
from openrlhf_agent.agentkit.rewards.result_rewards import SearchMatchingReward
from openrlhf_agent.agentkit.rewards.process_rewards import ToolFormatReward
from openrlhf_agent.agentkit.tools import WikiSearchTool

from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor

RETRIEVER_URL = os.environ.get("RETRIEVER_URL", "http://localhost:8000/retrieve")
MAX_AGENT_STEPS = 32

SYSTEM_PROMPT = f"""
You are a helpful assistant operating in TRAINING mode.

Rules:
1. Complete each user question within {MAX_AGENT_STEPS} assistant turns.
2. Use `wiki_search` to look up external facts or verify uncertain claims.
3. Before finalizing, verify both the answer and the exact final-line format.

## Output Rules
- First provide a clear markdown explanation of the solution.
- Then end exactly with:
  `Answer: <final_answer>`
- The answer line must contain only the final answer in canonical form.
- Do not add any text after the final answer line.
""".strip()


class AgentInstance(AgentInstanceBase):
    def __init__(self, *args, **kwargs):
        self.protocol = Qwen3Protocol(enable_thinking=True)
        self.session = AgentSession(
            environment=FunctionCallEnvironment(
                system_prompt=SYSTEM_PROMPT,
                tools=[WikiSearchTool(base_url=RETRIEVER_URL)],
                max_steps=MAX_AGENT_STEPS,
            ),
            reward_pipeline=RewardPipeline(
                process_rewards=[ToolFormatReward(penalty=-0.1)],
                result_rewards=[
                    SearchMatchingReward(
                        correct_score=1.0,
                        format_score=0.1,
                        miss_score=0.0,
                    ),
                ],
            ),
        )

    async def reset(self, states: dict, **kwargs):
        messages = await self.session.reset(states.get("observation"))
        prompt = self.protocol.render(
            messages=messages,
            tools=self.session.environment.tools_manifest(),
            add_generation_prompt=True,
        )
        return {"observation": prompt.text}

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        action_text: str = states.get("action_text", "")
        label = states.get("label")

        action = self.protocol.parse_action(action_text)
        observation, reward = await self.session.step(
            action,
            label=label,
        )
        reward = float(reward) if reward is not None else 0.0

        done = observation.done
        feedback_text = ""
        if not done:
            environment_messages = [
                message.model_dump(exclude_none=True)
                for message in observation.environment_messages
            ]
            feedback_text = self.protocol.render_feedback(
                messages=self.session.history.messages,
                environment_messages=environment_messages,
                tools=self.session.environment.tools_manifest(),
            ).text
        return {
            "rewards": torch.tensor(reward),
            "scores": torch.tensor(reward),
            "environment_feedback": feedback_text,
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
