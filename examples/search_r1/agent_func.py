from datetime import datetime
from typing import Any, Dict

import re
import string

from openrlhf_agent.agentkit.rewards import RewardPipeline
from openrlhf_agent.agentkit.session import AgentSession
from openrlhf_agent.agentkit.environments import FunctionCallEnvironment
from openrlhf_agent.agentkit.protocols import Qwen3ThinkingProtocol
from openrlhf_agent.agentkit.rewards.result_rewards import MatchingReward
from openrlhf_agent.agentkit.tools import WikiSearchTool

from openrlhf.utils.agent import MultiTurnAgentExecutor, AgentInstanceBase


CUSTOM_SYSTEM_PROMPT = """
You are a helpful assistant.

## Output Rules
- First provide a clear markdown explanation of the solution.
- Then end exactly with:
  `Answer: <final_answer>`
- The answer line must contain only the final answer in canonical form.
- Do not add any text after the final answer line.
""".strip()


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_FINAL_ANSWER_RE = re.compile(r"(?im)^\s*Answer:\s*(.+?)\s*$")
_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


# Synchronize with search-r1
def normalize_answer(text: str) -> str:
    text = text.lower().translate(_PUNCT_TRANSLATION)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def extract_final_answer(response: str) -> str | None:
    matches = _FINAL_ANSWER_RE.findall(response)
    if matches:
        answer = matches[-1].strip()
        return answer or None
    return None


class EMMatchingReward(MatchingReward):
    def score_response(self, response: str, label) -> float:
        if isinstance(label, str):
            labels = [label]
        elif isinstance(label, list):
            labels = label
        else:
            raise NotImplementedError(f"Unsupported label type: {type(label)!r}")

        pred_answer = extract_final_answer(response)
        if pred_answer is None:
            return self.miss_score

        normalized_labels = {normalize_answer(str(item)) for item in labels}
        normalized_pred_answer = normalize_answer(pred_answer)
        return self.correct_score if normalized_pred_answer in normalized_labels else self.miss_score


class AgentInstance(AgentInstanceBase):
    def __init__(self):
        self.session = AgentSession(
            environment=FunctionCallEnvironment(
                system_prompt=CUSTOM_SYSTEM_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d")),
                tools=[
                    WikiSearchTool(base_url="http://localhost:8000/retrieve"),
                ],
            ),
            protocol=Qwen3ThinkingProtocol(),
            reward_pipeline=RewardPipeline(
                result_reward=EMMatchingReward(
                    correct_score=1.0, miss_score=0.0
                ),
            )
        )

    async def reset(self, states: dict):
        prompt = await self.session.initialize(states.get("observation"))
        return {"observation": prompt}

    async def step(self, states: dict) -> Dict[str, Any]:
        action_text: str = states.get("action_text", "")
        label = states.get("label")

        observation, reward = await self.session.step_from_text(action_text, label=label)

        reward = float(reward) if reward is not None else 0.0
        reward = max(reward, -1.0)

        done = observation.done
        return {
            "rewards": reward,
            "scores": reward,
            "environment_feedback": "" if done else observation.feedback_text,
            "done": done,
            "sampling_params": states.get("sampling_params", None),
            "extra_logs": {
                "dummy_scores": reward,
                "turn_count": observation.step_index,
            },
        }


class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)
