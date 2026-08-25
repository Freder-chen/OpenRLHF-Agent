# -*- coding: utf-8 -*-

import asyncio
from typing import Any, Dict, List

from datasets import load_dataset, concatenate_datasets
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


def normalize_golds(label: Any) -> List[str]:
    if label is None or label == "":
        return []
    if isinstance(label, (str, int, float)):
        text = str(label).strip()
        return [text] if text else []
    try:
        return [text for x in list(label) if (text := str(x).strip())]
    except TypeError:
        return []


async def run_one(backend: VLLMCompletionBackend, question: str) -> str:
    rt = AgentRuntime(
        backend=backend,
        protocol=Qwen3Protocol(enable_thinking=True),
        environment=SingleTurnEnvironment(system_prompt=EVAL_SYSTEM_PROMPT),
    )

    messages = [{"role": "user", "content": str(question)}]
    return await rt.run_final(messages)


async def evaluate(dataset_name, split, n_repeat=1, concurrency=50) -> Dict[str, Any]:
    dataset = load_dataset(dataset_name, split=split)
    if n_repeat > 1:
        dataset = concatenate_datasets([dataset] * n_repeat)

    async with VLLMCompletionBackend(
        model="qwen3",
        base_url="http://localhost:8009/v1",
        api_key="empty",
    ) as backend:
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()

        metric_sum = 0.0
        metric_cnt = 0
        num_errors = 0

        async def run_item(item: Dict[str, Any], pbar: tqdm) -> None:
            nonlocal metric_sum, metric_cnt, num_errors

            question = item["problem"]
            golds = normalize_golds(item["answer"])

            async with sem:
                infer_failed = False

                try:
                    pred = await run_one(backend, question)
                    ok = _REWARD.score_response(pred, golds) >= _REWARD.correct_score
                except Exception as e:
                    ok = False
                    infer_failed = True
                    print(f"Error: {e}")

            async with lock:
                metric_sum += 1.0 if ok else 0.0
                metric_cnt += 1
                if infer_failed:
                    num_errors += 1
                pbar.update(1)

        total = len(dataset)
        with tqdm(total=total, desc="Evaluating") as pbar:
            await asyncio.gather(*(run_item(it, pbar) for it in dataset))

    math_match = (metric_sum / metric_cnt) if metric_cnt else 0.0
    return {
        "num_samples": total,
        "num_errors": num_errors,
        "metrics": {"math_match": math_match},
    }


async def main() -> None:
    # Datasets:
    #   AIME 2024: https://huggingface.co/datasets/HuggingFaceH4/aime_2024
    #   AIME 2025: https://huggingface.co/datasets/MathArena/aime_2025
    #   AIME 2026: https://huggingface.co/datasets/MathArena/aime_2026
    #   HMMT Feb 2025: https://huggingface.co/datasets/MathArena/hmmt_feb_2025
    #   Apex 2025: https://huggingface.co/datasets/MathArena/apex_2025
    #   Apex Shortlist: https://huggingface.co/datasets/MathArena/apex-shortlist
    datasets = [
        # Math: (dataset, split, n_repeat) — n_repeat = runs per problem (avg@n)
        # {"dirname": "HuggingFaceH4/aime_2024", "split": "train", "n_repeat": 4},
        # {"dirname": "MathArena/aime_2025", "split": "train", "n_repeat": 4},
        # {"dirname": "MathArena/aime_2026", "split": "train", "n_repeat": 4},
        # {"dirname": "MathArena/hmmt_feb_2025", "split": "train", "n_repeat": 4},
        {"dirname": "MathArena/apex_2025", "split": "train", "n_repeat": 1},
        # {"dirname": "MathArena/apex-shortlist", "split": "train", "n_repeat": 4},
    ]
    for dataset in datasets:
        dataset_dirname = dataset["dirname"]
        dataset_split = dataset["split"]
        n_repeat = dataset.get("n_repeat", 1)
        result = await evaluate(
            dataset_dirname, dataset_split, n_repeat=n_repeat, concurrency=50
        )

        dataset_name = dataset_dirname.split("/")[-1]
        print(f"{dataset_name}: {result}")


if __name__ == "__main__":
    asyncio.run(main())
