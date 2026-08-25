#!/usr/bin/env bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../../..")"

set -euo pipefail

MODEL_PATH="Qwen/Qwen3-4B-Thinking-2507"
DATASET_PATH="MathArena/arxivmath-training"
SAVE_PATH="$WORK_DIR/exp/Qwen3-4B-Thinking-ArxivMath"
AGENT_FUNC_PATH="$(realpath "$SCRIPT_DIR/agent.py")"

CKPT_ARGS=(
  --actor.model_name_or_path "${MODEL_PATH}"
  --ckpt.load_enable
  --ckpt.output_dir "${SAVE_PATH}"
  --ckpt.path "${SAVE_PATH}/ckpt"
  --ckpt.save_hf
  --ckpt.save_steps 10
  --ckpt.max_num 3
)

ROLLOUT_ARGS=(
  --train.agent_func_path "${AGENT_FUNC_PATH}"

  --data.prompt_dataset "${DATASET_PATH}"
  --data.input_key question
  --data.label_key answer
  --data.max_len 128000
  --data.max_samples 128000000
  --ds.packing_samples

  --rollout.vllm_generate_batch_size 32
  --rollout.batch_size 32
  --rollout.micro_batch_size 1
  --rollout.n_samples_per_prompt 32
  --rollout.max_tokens_per_gpu 65536

  --train.batch_size 1024
  --train.micro_batch_size 1
  --train.dynamic_batch_enable
  --train.max_tokens_per_gpu 40960
  --train.max_epochs 1
  --train.num_episodes 1
)

ENGINE_ARGS=(
  --train.async_enable
  --train.partial_rollout_enable

  --ref.num_nodes 1
  --ref.num_gpus_per_node 4
  --actor.num_nodes 1
  --actor.num_gpus_per_node 4
  --actor.gradient_checkpointing_enable

  --vllm.num_engines 4
  --vllm.tensor_parallel_size 1
  --vllm.gpu_memory_utilization 0.9
  --vllm.sync_backend nccl
  --vllm.enforce_eager

  --train.colocate_actor_ref
  --ds.enable_sleep
  --ds.zero_stage 3
  --ds.param_dtype bf16
  --ds.attn_implementation flash_attention_2
  --ds.ring_attn_size 4
  --ds.ring_attn_head_stride 2
)

OPTIMIZER_ARGS=(
  --algo.advantage.estimator reinforce
  --actor.adam.lr 5e-7
  --actor.entropy_coef 0

  --algo.kl.use_loss
  --algo.kl.init_coef 1e-5
  --algo.kl.estimator k2
  --algo.advantage.is_correction_enable
  --algo.advantage.is_correction_type icepop
  --algo.advantage.is_correction_threshold 0.99 1.01
)

LOG_ARGS=(
  --logger.tensorboard_dir "${SAVE_PATH}/runs"
  --logger.logging_steps 1
  --eval.steps -1
)

python3 -m openrlhf.cli.train_ppo_ray \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${ENGINE_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${LOG_ARGS[@]}"
