# Molt math training

This example connects `AgentSession` to Molt's training interface.

## Install

```bash
python -m pip install -e .
python -m pip install "molt-rl[vllm]==0.1.6"
```

## Run

```bash
bash examples/math/molt/train.sh
```

The script trains `Qwen/Qwen3-4B-Thinking-2507` on `MathArena/arxivmath-training`.
