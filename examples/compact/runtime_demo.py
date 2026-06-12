"""Runtime demo with compaction — inference version.

Unlike training (where the rollout model generates the summary),
at inference the runtime generates the summary itself via engine.generate().
"""

import asyncio

from openrlhf_agent.backends import OpenAIEngine
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.session import CompactableSession
from openrlhf_agent.agentkit.environments import FunctionCallEnvironment
from openrlhf_agent.agentkit.protocols import Qwen3ThinkingProtocol


async def main() -> None:
    engine = OpenAIEngine(
        model="qwen3",
        base_url="http://localhost:8009/v1",
        api_key="empty",
    )
    session = CompactableSession(
        environment=FunctionCallEnvironment(),
        protocol=Qwen3ThinkingProtocol(),
    )

    rt = AgentRuntime(engine, session, max_context_tokens=8192)
    messages = [{"role": "user", "content": "Research the history of Python programming language in detail."}]
    async for message in rt.run_steps(messages):
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
