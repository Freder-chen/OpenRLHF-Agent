# Math examples

This directory contains math inference, evaluation, and training examples for `Qwen3-4B-Thinking-2507`.

## Inference

Start a compatible vLLM server on port 8009, then run:

```bash
python examples/math/runtime_demo.py
```

Evaluate a Hugging Face dataset containing `problem` and `answer` columns:

```bash
python examples/math/eval.py MathArena/apex_2025
```

## OpenRLHF training

Install the project and OpenRLHF, adjust the model and dataset paths in `openrlhf/train.sh`, then run:

```bash
python -m pip install -e .
python -m pip install "openrlhf[vllm]==0.11.0"
bash examples/math/openrlhf/train.sh
```

## VIME training

The VIME example uses one 8-GPU node for Megatron training and one 8-GPU node for rollout. Adjust the paths in `vime/prepare.sh` and `vime/train.sh`, then run:

```bash
conda activate vime
bash examples/math/vime/prepare.sh
bash examples/math/vime/train.sh
```

Preparation converts the model to a Megatron checkpoint and the ArxivMath dataset to VIME JSONL. Re-running it reuses the existing model checkpoint. Training output and TensorBoard logs are written to `exp/Qwen3-4B-Thinking-ArxivMath-vime/`.
