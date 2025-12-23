import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from openrlhf_agent.backends import OpenAIEngine
from openrlhf_agent.agentkit.factory import build_environment, build_protocol
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.tools.hub.commentary import CommentaryTool
from examples.webui.gradio_ui import launch_runtime_ui
from examples.search_r1.search_tool import LocalSearchTool


CUSTOM_SYSTEM_PROMPT = """
Answer the given question. First, think step by step inside <think> and </think> whenever you receive new information. 
After reasoning, decide whether to use tools. Use tools to verify specific aspects of your reasoning or to fetch missing knowledge; 
do not rely on tools to write the final answer. Call the commentary tool only for brief progress updates.

If the conditions for solving the problem have been met, directly provide the final answer inside <final> and </final> without extra illustrations. 
Example: <final> ... </final>.

Knowledge cutoff: 2023-06
Current date: {date}
""".strip().format(date=datetime.now().strftime("%Y-%m-%d"))

TOOLS = [CommentaryTool(), LocalSearchTool()] # Available Tools

async def main() -> None:
    engine = OpenAIEngine(
        model="qwen3",
        base_url="http://localhost:8009/v1",
        api_key="empty",
    )
    env = build_environment(
        name="function_call",
        tools=TOOLS,
        system_prompt=CUSTOM_SYSTEM_PROMPT,
    )
    protocol = build_protocol(name="qwen3_thinking")
    launch_runtime_ui(
        rt=AgentRuntime(engine, env, protocol),
        tools=TOOLS,
        title="## Search Chat · Qwen3",
        description="### UI Demo for R1 with Qwen3 model.",
        port=7867,
        open_browser=True,
        default_question="Please use the commentary function to share your thoughts, and also help me search what Python is?",
    )  

if __name__ == "__main__":
    # python -m examples.webui.runtime_demo
    asyncio.run(main())
