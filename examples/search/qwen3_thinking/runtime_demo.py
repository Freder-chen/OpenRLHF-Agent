import asyncio

from openrlhf_agent.backends import OpenAIEngine
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.environments import FunctionCallEnvironment, SingleTurnEnvironment
from openrlhf_agent.agentkit.protocols import Qwen3ThinkingProtocol, Qwen3InstructProtocol
from openrlhf_agent.agentkit.tools import PrivilegedFeedbackTool, WikiSearchTool, ThinkTool, CommentaryTool

RETRIEVER_URL = "http://localhost:8000/retrieve"
BASE_URL = "http://localhost:8009/v1"
API_KEY = "empty"
MODEL = "qwen3"

SYSTEM_PROMPT = """
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
        protocol=Qwen3ThinkingProtocol(), # qwen3-thinking
        engine=OpenAIEngine(base_url=BASE_URL, api_key=API_KEY, model=MODEL),
        environment=FunctionCallEnvironment(
            system_prompt=SYSTEM_PROMPT,
            tools=[WikiSearchTool(base_url=RETRIEVER_URL)],
        ),
        max_model_len=32768,
    )
    messages = [{"role": "user", "content": "What did the technical device made by the british aeronautical engineer during the second world war do?"}]
    async for message in agent_runtime.run_steps(messages):
        print("-"*50)
        print(message)
    
    print("="*50)
    print(message["content"])


if __name__ == "__main__":
    asyncio.run(main())
