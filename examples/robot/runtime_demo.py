"""Run one LIBERO episode with a VLM agent."""

import argparse
import asyncio
import json

from openrlhf_agent.agentkit import AgentRuntime
from openrlhf_agent.agentkit.environments import RobotEnvironment
from openrlhf_agent.model import Qwen3p8Protocol, VLLMCompletionBackend

from examples.robot.libero.client import LiberoClient


LIBERO_SYSTEM_PROMPT = """
You are controlling a simulated Franka Panda robot through tool calls.
Complete the user's task within {max_steps} turns.

# Workflow

1. Inspect the latest task status, gripper opening, and camera images.
2. Briefly state what you observe and why you chose the next motion.
3. Make one small, deliberate motion by calling `act` exactly once.
4. Inspect the new observation before choosing the next motion.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--env-url", default="http://127.0.0.1:8010")
    parser.add_argument("--model", default="qwen3.8-vl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8009")
    parser.add_argument("--api-key", default="empty")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    environment = RobotEnvironment(
        client=LiberoClient(
            suite=args.suite,
            task_id=args.task_id,
            init_state_id=args.init_state_id,
            base_url=args.env_url,
        ),
        system_prompt=LIBERO_SYSTEM_PROMPT.format(max_steps=args.max_steps),
        max_steps=args.max_steps,
    )
    backend = VLLMCompletionBackend(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        protocol=Qwen3p8Protocol(
            enable_thinking=True,
            preserve_thinking=True,
            reasoning_effort="medium",
        ),
    )

    async with environment, backend:
        runtime = AgentRuntime(backend=backend, environment=environment)
        async for message in runtime.run_steps([]):
            if isinstance(message.get("content"), list):
                message["content"] = [
                    part
                    if part.get("type") != "image_url"
                    else {"type": "image_url", "image_url": "<omitted>"}
                    for part in message["content"]
                ]
            print(json.dumps(message, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
