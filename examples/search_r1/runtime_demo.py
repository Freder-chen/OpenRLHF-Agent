import asyncio
from datetime import datetime
from openrlhf_agent.backends import OpenAIEngine
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.environments import FunctionCallEnvironment
from openrlhf_agent.agentkit.protocols import Qwen3ThinkingProtocol
from openrlhf_agent.agentkit.tools import WikiSearchTool


CUSTOM_SYSTEM_PROMPT = """
You are a helpful assistant.

## Output Rules
- First provide a clear markdown explanation of the solution.
- Then end exactly with:
  `Answer: <final_answer>`
- The answer line must contain only the final answer in canonical form.
- Do not add any text after the final answer line.
""".strip()


async def main() -> None:
    agent_runtime = AgentRuntime(
        protocol=Qwen3ThinkingProtocol(),
        engine=OpenAIEngine(
            model="qwen3", 
            base_url="http://localhost:8009/v1",
            api_key="empty"
        ),
        environment=FunctionCallEnvironment(
            system_prompt=CUSTOM_SYSTEM_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d")),
            tools=[
                WikiSearchTool(base_url="http://localhost:8000/retrieve"),
            ],
        ),
    )
    messages = [{"role": "user", "content": "what's your goal?"}]
    async for message in agent_runtime.run_steps(messages):
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
