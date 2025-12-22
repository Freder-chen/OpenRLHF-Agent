import asyncio

from openrlhf_agent.backends import OpenAIEngine
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.factory import build_environment, build_protocol


async def main() -> None:
    engine = OpenAIEngine(
        model="qwen3",
        base_url="http://localhost:8009/v1",
        api_key="empty",
    )
    env = build_environment(name="single_turn")
    protocol = build_protocol(name="qwen3_thinking")

    rt = AgentRuntime(engine, env, protocol)
    messages = [{"role": "user", "content": "Tell me a joke about programming."}]
    async for step in rt.run_steps(messages):
        print(step)


if __name__ == "__main__":
    asyncio.run(main())