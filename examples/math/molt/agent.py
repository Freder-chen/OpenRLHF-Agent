"""Molt adapter for the Qwen3 math environment."""

from molt.agents import Env, Result, StepEnvRunner

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


class MathEnv(Env):
    """Adapt one OpenRLHF-Agent session to Molt's step interface."""

    def __init__(self):
        self.session = AgentSession(
            environment=SingleTurnEnvironment(system_prompt=TRAIN_SYSTEM_PROMPT),
            protocol=Qwen3Protocol(enable_thinking=True),
            reward_pipeline=RewardPipeline(
                result_rewards=[MathMatchingReward(correct_score=1.0, miss_score=0.0)]
            ),
        )

    async def reset(self, state: dict, **kwargs) -> dict:
        prompt = await self.session.reset(state["observation"])
        return {"observation": prompt.text}

    async def step(self, state: dict, **kwargs) -> Result:
        observation, reward = await self.session.step(
            state["action_text"],
            label=state["label"],
        )
        reward = float(reward or 0.0)

        return Result(
            reward=reward,
            score=reward,
            observation="" if observation.done else observation.feedback_text,
            terminated=observation.done,
            info={"turn_count": observation.step_index},
        )


class AgentRunner(StepEnvRunner):
    # AgentSession renders the Qwen template, so Molt must pass raw messages here.
    PRERENDER_PROMPT = False

    def __init__(self):
        super().__init__(MathEnv)
