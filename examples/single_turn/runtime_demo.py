import asyncio

from openrlhf_agent.backends import OpenAIEngine
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.session import AgentSession
from openrlhf_agent.agentkit.environments import SingleTurnEnvironment
from openrlhf_agent.agentkit.protocols import Qwen3ThinkingProtocol


async def main() -> None:
    engine = OpenAIEngine(
        model="qwen3",
        base_url="http://localhost:8009/v1",
        api_key="empty",
    )
    session = AgentSession(
        environment=SingleTurnEnvironment(),
        protocol=Qwen3ThinkingProtocol(),
    )

    rt = AgentRuntime(engine, session)
    messages = [{"role": "user", "content": "Tell me a joke about programming."}]
    async for message in rt.run_steps(messages):
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
