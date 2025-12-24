import asyncio
from datetime import datetime
from openrlhf_agent.backends import OpenAIEngine
from openrlhf_agent.agentkit.factory import build_environment, build_protocol
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.tools import CommentaryTool, LocalSearchTool


CUSTOM_SYSTEM_PROMPT = """
Answer the given question. First, think step by step inside <think> and </think> whenever you receive new information. 
After reasoning, decide whether to use tools. Use tools to verify specific aspects of your reasoning or to fetch missing knowledge; 
do not rely on tools to write the final answer. Call the commentary tool only for brief progress updates.

If the conditions for solving the problem have been met, directly provide the final answer inside <final> and </final> without extra illustrations. 
Example: <final> ... </final>.

Knowledge cutoff: 2023-06
Current date: {date}
""".strip()


async def main() -> None:
    engine = OpenAIEngine(
        model="qwen3", 
        base_url="http://localhost:8009/v1",
        api_key="empty"
    )
    env = build_environment(
        name="function_call",
        tools=[
            CommentaryTool(),
            LocalSearchTool(base_url="http://localhost:8000/retrieve"),
        ], # Available Tools,
        system_prompt=CUSTOM_SYSTEM_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d")),
    )
    protocol = build_protocol(name="qwen3_thinking")
    rt = AgentRuntime(engine, env, protocol)
    messages = [{"role": "user", "content": "Please use the commentary function to share your thoughts, and also help me search what Python is?"}]
    async for step in rt.run_steps(messages):
        print(step)
        print("-" * 100)    


if __name__ == "__main__":
    asyncio.run(main())
