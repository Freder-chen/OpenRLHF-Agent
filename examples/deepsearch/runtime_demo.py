"""Deep search demo using Jina Search + Read tools.

The agent iteratively searches the web and reads pages to answer
complex questions that require multi-step research.

Usage:
    export JINA_API_KEY=your_key
    python examples/deepsearch/runtime_demo.py
"""

import asyncio

from openrlhf_agent.backends import OpenAIEngine
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.session import CompactableSession
from openrlhf_agent.agentkit.environments import FunctionCallEnvironment
from openrlhf_agent.agentkit.protocols import Qwen3ThinkingProtocol
from openrlhf_agent.agentkit.tools import JinaSearchTool, JinaReadTool, ThinkTool, CommentaryTool


SYSTEM_PROMPT = """
You are a deep research agent. Answer questions by systematically searching and reading web sources.

## Tools
- `think`: Reason about what you've found and decide next steps. Be specific — name facts, compare sources, identify gaps.
- `commentary`: Brief progress update for the user (e.g. "Found 3 papers on X, reading the most cited one").
- `jina_search`: Web search. Use specific queries; vary keywords across searches to cover different angles.
- `jina_read`: Read a URL for full content. Use on the most relevant search results.

## Process
1. Think about what information is needed to answer the question.
2. Search with a specific query.
3. Think about results — what's useful, what's missing, which URLs to read.
4. Read 1-2 most promising URLs.
5. Think — do I have enough evidence? Are sources consistent? What's still unclear?
6. Repeat 2-5 with refined queries until confident.
7. Reply directly with your final synthesis (cite sources with URLs).

Use `commentary` to keep the user informed between long tool sequences.
""".strip()


async def main() -> None:
    rt = AgentRuntime(
        engine=OpenAIEngine(
            model="CAPF-Qwen3-4B-Thinking-Search",
            base_url="http://localhost:8009/v1",
            api_key="empty",
        ),
        session=CompactableSession(
            environment=FunctionCallEnvironment(
                system_prompt=SYSTEM_PROMPT,
                tools=[ThinkTool(), CommentaryTool(), JinaSearchTool(), JinaReadTool()],
            ),
            protocol=Qwen3ThinkingProtocol(),
            max_steps=999,
        ),
        max_context_tokens=40960,
        max_new_tokens_per_step=10240,
    )

    question = input("Question: ") or "What are the key differences between Rust and Zig, and when should you choose one over the other?"
    messages = [{"role": "user", "content": question}]

    print(f"\n🔍 Researching: {question}\n")
    async for message in rt.run_steps(messages):
        role = message.get("role")
        content = message.get("content", "")
        tool_calls = message.get("tool_calls")

        if role == "assistant" and tool_calls:
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("arguments", {})
                if name == "jina_search":
                    print(f"  🔎 Searching: {args.get('query', '')}")
                elif name == "jina_read":
                    print(f"  📄 Reading: {args.get('url', '')[:80]}")
                elif name == "think":
                    pass
        elif role == "assistant" and content and not tool_calls:
            print(f"\n{'='*60}\n📝 Answer:\n{'='*60}\n{content}")


if __name__ == "__main__":
    asyncio.run(main())
