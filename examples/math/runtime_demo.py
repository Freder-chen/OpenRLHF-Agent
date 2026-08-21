import asyncio

from openrlhf_agent.backends import VLLMCompletionBackend
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.environments import SingleTurnEnvironment
from openrlhf_agent.backends.openai.vllm.protocols import Qwen3Protocol


EVAL_SYSTEM_PROMPT = """
You are a helpful assistant.

## Output Rules
- First provide a clear markdown explanation of the solution.
- Then end exactly with:
  `Answer: \\boxed{<final_answer>}`
- The boxed expression must contain only the final answer in canonical form.
- Do not add any text after the boxed answer.
""".strip()


async def main() -> None:
    agent_runtime = AgentRuntime(
        backend=VLLMCompletionBackend(
            model="qwen3",
            base_url="http://localhost:8009/v1",
            api_key="empty",
            protocol=Qwen3Protocol(enable_thinking=True),
        ),
        environment=SingleTurnEnvironment(system_prompt=EVAL_SYSTEM_PROMPT),
    )
    messages = [{"role": "user", "content": "1+1=?"}]
    async for message in agent_runtime.run_steps(messages):
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
