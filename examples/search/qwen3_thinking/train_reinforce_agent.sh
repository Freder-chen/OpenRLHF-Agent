#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../../..")"

# set -x

MODEL_PATH="Qwen/Qwen3-4B-Thinking-2507"
DATASET_PATH="PeterJinGo/nq_hotpotqa_train" # use nq_hotpotqa_train/train.parquet
SAVE_PATH="${WORK_DIR}/exp/Qwen3-4B-Search-baseline"

# Demo for search
AGENT_FUNC_PATH="$(realpath "$SCRIPT_DIR")/agent_func.py"


CKPT_ARGS=(
   --pretrain ${MODEL_PATH}
   # --reward_pretrain ${REWARD_MODEL} # not used in agent mode
   --load_checkpoint

   --save_path ${SAVE_PATH}
   --ckpt_path "${SAVE_PATH}/ckpt"
   --save_hf_ckpt
   --max_ckpt_num 3
   --save_steps 20
)

ROLLOUT_ARGS=(
   --agent_func_path ${AGENT_FUNC_PATH}

   --prompt_data ${DATASET_PATH}
   --input_key question
   --label_key golden_answers
   --prompt_max_len 8000
   --generate_max_len 120000
   --apply_chat_template
   --packing_samples

   --vllm_generate_batch_size 128
   --rollout_batch_size 128
   --n_samples_per_prompt 8
   --train_batch_size 1024

   --use_dynamic_batch
   --train_max_tokens_per_gpu 40960
   --rollout_max_tokens_per_gpu 65536

   --max_samples 128000000
   --max_epochs 1
   --num_episodes 1
)

ENGINE_ARGS=(
   --async_train
   --partial_rollout

   --ref_num_nodes 1
   --ref_num_gpus_per_node 8
   --actor_num_nodes 1
   --actor_num_gpus_per_node 8
   --vllm_num_engines 16
   --vllm_tensor_parallel_size 1
   --vllm_gpu_memory_utilization 0.95
   --colocate_actor_ref
   --deepspeed_enable_sleep
   --vllm_sync_backend nccl
   --enforce_eager

   --zero_stage 3
   --gradient_checkpointing
   --ring_attn_size 4
   --ring_head_stride 2
)

OPTIMIZER_ARGS=(
   --advantage_estimator reinforce
   --actor_learning_rate 5e-7
   --entropy_loss_coef 0.0

   --use_kl_loss
   --init_kl_coef 1e-5
   --kl_estimator k2
)

LOG_ARGS=(
   --use_tensorboard ${SAVE_PATH}/runs
   --logging_steps 1
   --eval_steps -1
)


ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${ENGINE_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${LOG_ARGS[@]}
