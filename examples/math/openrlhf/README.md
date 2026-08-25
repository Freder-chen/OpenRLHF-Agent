# OpenRLHF math training

This example connects `AgentSession` to the current OpenRLHF training interface.

## Install

```bash
python -m pip install -e .
python -m pip install "openrlhf[vllm]==0.11.0"
```

## Run

```bash
bash examples/math/openrlhf/train.sh
```

The script trains `Qwen/Qwen3-4B-Thinking-2507` on `MathArena/arxivmath-training`.
