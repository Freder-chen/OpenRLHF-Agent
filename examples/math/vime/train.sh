#!/usr/bin/env bash

set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="$(realpath "$EXAMPLE_DIR/../../..")"

VIME_DIR="/root/vime"
MEGATRON_DIR="/root/Megatron-LM-vime"
MODEL_PATH="Qwen/Qwen3-4B-Thinking-2507"
SAVE_DIR="$PROJECT_DIR/exp/Qwen3-4B-Thinking-ArxivMath-vime"
TENSORBOARD_DIR="$SAVE_DIR/tensorboard"
TRAIN_DATA="$SAVE_DIR/data/train.jsonl"
CHECKPOINT_DIR="$SAVE_DIR/checkpoint"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$VIME_DIR:$MEGATRON_DIR:$EXAMPLE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MODEL_ARGS_ROTARY_BASE=5000000

if [[ ! -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" || ! -f "$TRAIN_DATA" ]]; then
  echo "Prepared model or data not found. Run this first:" >&2
  echo "  bash examples/math/vime/prepare.sh" >&2
  exit 1
fi

source "$VIME_DIR/scripts/models/qwen3-4B.sh"

# Model and checkpoint paths.
CKPT_ARGS=(
  --hf-checkpoint "$MODEL_PATH"
  --ref-load "$CHECKPOINT_DIR"
  --load "$SAVE_DIR"
  --save "$SAVE_DIR"
  --save-interval 10
)

# Generate 32 prompts x 8 answers, then train on all 256 answers.
ROLLOUT_ARGS=(
  --prompt-data "$TRAIN_DATA"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --custom-rm-path math_reward.reward_func
  --num-epoch 1
  --rollout-batch-size 32
  --over-sampling-batch-size 64
  --partial-rollout
  --n-samples-per-prompt 8
  --rollout-max-context-len 128000
  --rollout-max-response-len 128000
  --rollout-temperature 1.0
  --global-batch-size 256
  --balance-data
)

# Megatron training: TP=1, CP=4, and at most 32768 tokens per GPU.
TRAIN_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 4
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --no-gradient-accumulation-fusion
  --use-dynamic-batch-size
  --max-tokens-per-gpu 32768
)

# GRPO settings.
GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 1e-5
  --kl-loss-type k2
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28
  --use-tis
)

# Optimizer settings.
OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 5e-7
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

# The 4B model fits on one GPU, so the rollout node runs eight TP=1 engines.
VLLM_ARGS=(
  --rollout-num-gpus 8
  --rollout-num-gpus-per-engine 1
  --vllm-max-model-len 128000
  --vllm-gpu-memory-utilization 0.9
  # Suppress noisy Uvicorn access logs such as the periodic GET /health probes.
  --vllm-uvicorn-log-level warning
)

MISC_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node 8
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --use-tensorboard
)

# Both 8-GPU nodes must already belong to the same Ray cluster.
if ! ray status >/dev/null 2>&1; then
  echo "Ray is not running." >&2
  exit 1
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"$VIME_DIR:$MEGATRON_DIR:$EXAMPLE_DIR\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"TENSORBOARD_DIR\": \"$TENSORBOARD_DIR\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- python "$VIME_DIR/train_async.py" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${VLLM_ARGS[@]}" \
  "${MISC_ARGS[@]}"
