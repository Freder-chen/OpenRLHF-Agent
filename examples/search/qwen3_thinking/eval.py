# -*- coding: utf-8 -*-

import asyncio
from typing import Any, Dict, List

from tqdm import tqdm
from datasets import load_dataset

from openrlhf_agent.backends import OpenAIEngine
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.environments import FunctionCallEnvironment
from openrlhf_agent.agentkit.protocols import Qwen3ThinkingProtocol
from openrlhf_agent.agentkit.tools import WikiSearchTool
from openrlhf_agent.agentkit.rewards.result_rewards import SearchMatchingReward

RETRIEVER_URL = "http://localhost:8000/retrieve"
FLASHRAG_DATASET_REPO = "RUC-NLPIR/FlashRAG_datasets"

_REWARD = SearchMatchingReward(correct_score=1.0, miss_score=0.0)

EVAL_SYSTEM_PROMPT = """
You are a helpful assistant.

## Output Rules
- First provide a clear markdown explanation of the solution.
- Then end exactly with:
  `Answer: <final_answer>`
- The answer line must contain only the final answer in canonical form.
- Do not add any text after the final answer line.
""".strip()


def _coerce_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    items = [value] if isinstance(value, str) else value
    try:
        return [text for item in items if (text := str(item).strip())]
    except TypeError:
        return []


async def run_one(engine: OpenAIEngine, question: str, labels) -> str:
    rt = AgentRuntime(
        engine=engine,
        environment=FunctionCallEnvironment(
            system_prompt=EVAL_SYSTEM_PROMPT,
            tools=[WikiSearchTool(base_url=RETRIEVER_URL)],
        ),
        protocol=Qwen3ThinkingProtocol(),
        max_model_len=32768,
    )

    messages = [{"role": "user", "content": str(question)}]
    return await rt.run_final(messages)


async def evaluate(dataset_name, data_dir, split, concurrency=50) -> Dict[str, Any]:
    dataset = load_dataset(dataset_name, data_dir=data_dir, split=split)
    engine = OpenAIEngine(
        base_url="http://localhost:8009/v1",
        api_key="empty",
        model="qwen3",
    )
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    metric_sum = 0.0
    metric_cnt = 0
    num_errors = 0

    async def run_item(item: Dict[str, Any], pbar: tqdm) -> None:
        nonlocal metric_sum, metric_cnt, num_errors

        question = item["question"]
        golds = _coerce_text_list(item["golden_answers"])

        async with sem:
            infer_failed = False

            try:
                pred = await run_one(engine, question, golds)
                ok = _REWARD.score_response(pred, golds) >= _REWARD.correct_score
            except Exception as e:
                ok = ""
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

    exact_match = (metric_sum / metric_cnt) if metric_cnt else 0.0
    return {
        "num_samples": total,
        "num_errors": num_errors,
        "metrics": {"exact_match": exact_match},
    }


async def main() -> None:
    datasets = [
        # General QA
        {"data_dir": "nq", "split": "test"},
        {"data_dir": "triviaqa", "split": "test"},
        {"data_dir": "popqa", "split": "test"},
        # Multi-Hop QA
        {"data_dir": "hotpotqa", "split": "validation"},
        {"data_dir": "2wikimultihopqa", "split": "validation"},
        {"data_dir": "musique", "split": "validation"},
        {"data_dir": "bamboogle", "split": "test"},
    ]
    for dataset in datasets:
        data_dir = dataset["data_dir"]
        dataset_split = dataset["split"]
        result = await evaluate(FLASHRAG_DATASET_REPO, data_dir, dataset_split, concurrency=48)

        print(f"{data_dir}: {result}")


if __name__ == "__main__":
    asyncio.run(main())
