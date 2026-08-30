import asyncio
from argparse import ArgumentParser
from typing import Any

from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

from openrlhf_agent.model import Qwen3Protocol, VLLMCompletionBackend
from openrlhf_agent.agentkit import AgentRuntime
from openrlhf_agent.agentkit.environments import SingleTurnEnvironment
from openrlhf_agent.agentkit.rewards.result_rewards import MathMatchingReward

_REWARD = MathMatchingReward(correct_score=1.0, miss_score=0.0)


EVAL_SYSTEM_PROMPT = """
You are a helpful assistant.

## Output Rules
- First provide a clear markdown explanation of the solution.
- Then end exactly with:
  `Answer: \\boxed{<final_answer>}`
- The boxed expression must contain only the final answer in canonical form.
- Do not add any text after the boxed answer.
""".strip()


def normalize_golds(label: Any) -> list[str]:
    if label is None or label == "":
        return []
    if isinstance(label, (str, int, float)):
        text = str(label).strip()
        return [text] if text else []
    try:
        return [text for x in list(label) if (text := str(x).strip())]
    except TypeError:
        return []


async def evaluate(
    dataset_name: str,
    split: str,
    *,
    repeat: int,
    concurrency: int,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_name, split=split)
    if repeat > 1:
        dataset = concatenate_datasets([dataset] * repeat)

    async with VLLMCompletionBackend(
        model="qwen3",
        base_url="http://localhost:8009/v1",
        api_key="empty",
    ) as backend:
        sem = asyncio.Semaphore(concurrency)

        async def run_item(item: dict[str, Any]) -> tuple[bool, bool]:
            question = item["problem"]
            golds = normalize_golds(item["answer"])

            async with sem:
                try:
                    runtime = AgentRuntime(
                        backend=backend,
                        protocol=Qwen3Protocol(enable_thinking=True),
                        environment=SingleTurnEnvironment(system_prompt=EVAL_SYSTEM_PROMPT),
                    )
                    pred = await runtime.run_final(
                        [{"role": "user", "content": str(question)}]
                    )
                    ok = _REWARD.score_response(pred, golds) >= _REWARD.correct_score
                except Exception as error:
                    tqdm.write(f"Error: {error}")
                    return False, True
                return ok, False

        total = len(dataset)
        tasks = [run_item(item) for item in dataset]
        results = []
        with tqdm(total=total, desc="Evaluating") as pbar:
            for task in asyncio.as_completed(tasks):
                results.append(await task)
                pbar.update(1)

    correct = sum(ok for ok, _ in results)
    return {
        "num_samples": total,
        "num_errors": sum(failed for _, failed in results),
        "metrics": {"math_match": correct / total if total else 0.0},
    }


def parse_args():
    parser = ArgumentParser(description="Evaluate a math model served by vLLM.")
    parser.add_argument("dataset", nargs="?", default="MathArena/apex_2025")
    parser.add_argument("--split", default="train")
    parser.add_argument("--repeat", type=int, default=1, help="Runs per problem")
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()
    if args.repeat < 1 or args.concurrency < 1:
        parser.error("--repeat and --concurrency must be positive")
    return args


async def main() -> None:
    args = parse_args()
    result = await evaluate(
        args.dataset,
        args.split,
        repeat=args.repeat,
        concurrency=args.concurrency,
    )
    print(f"{args.dataset.split('/')[-1]}: {result}")


if __name__ == "__main__":
    asyncio.run(main())
