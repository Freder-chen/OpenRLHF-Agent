"""Convert ArxivMath parquet rows to the chat JSONL expected by VIME."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

SYSTEM_PROMPT = """
You are a helpful assistant operating in TRAINING mode.

Rules:
1. Before finishing, verify both the answer and the exact boxed-answer format.

## Output Rules
- First provide a clear markdown explanation of the solution.
- Then end exactly with:
  `Answer: \\boxed{<final_answer>}`
- The boxed expression must contain only the final answer in canonical form.
- Do not add any text after the boxed answer.
""".strip()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: prepare_data.py SOURCE_PARQUET_OR_DIR OUTPUT_JSONL")

    source, output = map(Path, sys.argv[1:])
    parquet_files = [source] if source.is_file() else sorted(source.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0

    with temporary.open("w", encoding="utf-8") as stream:
        for parquet_path in parquet_files:
            parquet = pq.ParquetFile(parquet_path)
            for batch in parquet.iter_batches():
                for row in batch.to_pylist():
                    question = row.get("question")
                    answer = row.get("answer")
                    if not isinstance(question, str) or answer is None:
                        continue
                    record = {
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": question},
                        ],
                        "label": str(answer),
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

    temporary.replace(output)
    print(f"Wrote {count} samples to {output}")


if __name__ == "__main__":
    main()
