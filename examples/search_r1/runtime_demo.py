import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openrlhf_agent.backends import OpenAIEngine
from openrlhf_agent.agentkit.factory import build_environment, build_protocol
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.tools.hub.commentary import CommentaryTool
from examples.gui import launch_runtime_ui, run_console
from search_tool import LocalSearchTool


CUSTOM_SYSTEM_PROMPT = """
Answer the given question. First, think step by step inside <think> and </think> whenever you receive new information. After reasoning, decide whether to use tools. Use tools to verify specific aspects of your reasoning or to fetch missing knowledge; do not rely on tools to write the final answer. Call the commentary tool only for brief progress updates.

If the conditions for solving the problem have been met, directly provide the final answer inside <answer> and </answer> without extra illustrations. Example: <answer> ... </answer>.

Knowledge cutoff: 2023-06
Current date: {date}
""".strip().format(date=datetime.now().strftime("%Y-%m-%d"))


def build_runtime(model: str, base_url: str, api_key: str) -> AgentRuntime:
    engine = OpenAIEngine(model=model, base_url=base_url, api_key=api_key)
    env = build_environment(
        name="function_call",
        tools=[CommentaryTool(), LocalSearchTool()],
        system_prompt=CUSTOM_SYSTEM_PROMPT,
    )
    protocol = build_protocol(name="qwen3_thinking")
    return AgentRuntime(engine, env, protocol)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Search-R1 runtime demo.")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--base-url", default="http://localhost:8009/v1")
    parser.add_argument("--api-key", default="empty")
    parser.add_argument("--question", default="Curious is a women’s fragrance by a singer born in what city and state?")
    parser.add_argument("--ui", action="store_true", default=True, help="Launch web UI (Gradio) instead of console print.")
    parser.add_argument("--port", type=int, default=7867, help="Port for the UI.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio public link.")
    parser.add_argument("--open-browser", action="store_true", default=True, help="Open browser automatically when UI starts.")
    args = parser.parse_args()

    if args.ui:
        launch_runtime_ui(
            runtime_builder=lambda: build_runtime(args.model, args.base_url, args.api_key),
            title="## Search-R1 Chat · Qwen3",
            description="### UI Demo for R1 with Qwen3 model.",
            port=args.port,
            share=args.share,
            open_browser=args.open_browser,
            default_question=args.question,
        )
    else:
        runtime = build_runtime(model=args.model, base_url=args.base_url, api_key=args.api_key)
        await run_console(runtime, args.question)


if __name__ == "__main__":
    asyncio.run(main())
