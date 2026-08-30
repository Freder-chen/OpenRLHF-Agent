#!/usr/bin/env bash

set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="$(realpath "$EXAMPLE_DIR/../../..")"

VIME_DIR="/root/vime"
MEGATRON_DIR="/root/Megatron-LM-vime"
DATASET_PATH="MathArena/arxivmath-training"
MODEL_PATH="Qwen/Qwen3-4B-Thinking-2507"

TRAIN_DATA="$PROJECT_DIR/exp/Qwen3-4B-Thinking-ArxivMath-vime/data/train.jsonl"
CHECKPOINT_DIR="$PROJECT_DIR/exp/Qwen3-4B-Thinking-ArxivMath-vime/checkpoint"

export PYTHONPATH="$MEGATRON_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_ARGS_ROTARY_BASE=5000000

if [[ ! -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]]; then
  echo "Converting the Hugging Face model to a Megatron checkpoint..."
  source "$VIME_DIR/scripts/models/qwen3-4B.sh"
  mkdir -p "$CHECKPOINT_DIR"

  cd "$VIME_DIR"
  python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --no-gradient-accumulation-fusion \
    --hf-checkpoint "$MODEL_PATH" \
    --save "$CHECKPOINT_DIR"
else
  echo "Model checkpoint already exists."
fi

echo "Converting the training data to VIME JSONL..."
python "$EXAMPLE_DIR/prepare_data.py" "$DATASET_PATH" "$TRAIN_DATA"

echo "Preparation complete."
