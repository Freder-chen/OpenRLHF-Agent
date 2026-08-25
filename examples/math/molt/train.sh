#!/usr/bin/env bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../../..")"

set -euo pipefail

MODEL_PATH="Qwen/Qwen3-4B-Thinking-2507"
DATASET_PATH="MathArena/arxivmath-training"
SAVE_PATH="$WORK_DIR/exp/Qwen3-4B-Thinking-ArxivMath"
AGENT_PATH="$(realpath "$SCRIPT_DIR/agent.py")"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

CKPT_ARGS=(
  --actor.model_name_or_path "$MODEL_PATH"

  --ckpt.load_enable
  --ckpt.output_dir "$SAVE_PATH/hf"
  --ckpt.path "$SAVE_PATH/state"
  --ckpt.save_steps 10
)

ROLLOUT_ARGS=(
  --train.agent_path "$AGENT_PATH"

  --data.prompt_dataset "$DATASET_PATH"
  --data.input_key question
  --data.label_key answer
  --data.apply_chat_template
  --data.max_len 128000
  --data.max_samples 128000000

  --rollout.batch_size 32
  --rollout.vllm_generate_batch_size 32
  --rollout.micro_batch_size 1
  --rollout.n_samples_per_prompt 32
  --rollout.temperature 1.0
  --rollout.max_tokens_per_gpu 65536

  --train.batch_size 1024
  --train.micro_batch_size 1
  --train.dynamic_batch_enable
  --train.max_tokens_per_gpu 40960
  --train.max_epochs 1
  --train.num_episodes 1
  --train.async_queue_size 1
  --train.partial_rollout_enable
)

ENGINE_ARGS=(
  --train.colocate_fsdp_models

  --actor.num_nodes 1
  --actor.num_gpus_per_node 4
  --ref.num_nodes 1
  --ref.num_gpus_per_node 4
  --actor.gradient_checkpoint full

  --vllm.num_engines 4
  --vllm.tensor_parallel_size 1
  --vllm.sync_backend nccl
  --vllm.gpu_memory_utilization 0.9
  --vllm.distributed_executor_backend mp

  --fsdp.param_dtype bf16
  --fsdp.attn_implementation flash_attention_2
  --fsdp.tp_size 1
  --fsdp.ep_size 1
  --fsdp.cp_size 4
  --fsdp.packing_samples
)

OPTIMIZER_ARGS=(
  --algo.advantage.estimator reinforce
  --actor.adam.lr 5e-7
  --actor.entropy_coef 0

  --algo.kl.use_loss
  --algo.kl.estimator k2
  --algo.kl.init_coef 1e-5
  --algo.advantage.is_correction_level token
  --algo.advantage.is_correction_mode mask
  --algo.advantage.is_correction_threshold 0.99 1.01
)

LOG_ARGS=(
  --logger.tensorboard_dir "$SAVE_PATH/runs"
  --logger.logging_steps 1
)

MOLT_STARTED_RAY=0
if ! ray status >/dev/null 2>&1; then
  ray start --head --num-gpus=8 --disable-usage-stats >/dev/null
  MOLT_STARTED_RAY=1
fi

cleanup() {
  if [[ "$MOLT_STARTED_RAY" == 1 ]]; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

cd "$WORK_DIR"
python3 -u -m molt.cli.train_rl_ray \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${ENGINE_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${LOG_ARGS[@]}"
