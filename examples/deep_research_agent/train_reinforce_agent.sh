#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../..")"

set -x

MODEL_PATH="YOUR_MODEL_PATH"
# e.g.
DATASET_PATH="${WORK_DIR}/examples/deep_research_agent/data/rl_train_q_rubrics.jsonl"

SAVE_PATH="${WORK_DIR}/exp/Qwen3-Search-R1"

# Demo for search
AGENT_FUNC_PATH="${WORK_DIR}/examples/search_r1/agent_func.py"


CKPT_ARGS=(
   --pretrain ${MODEL_PATH}
   # --reward_pretrain ${REWARD_MODEL} # agent mode no use
   --load_checkpoint

   --save_path ${SAVE_PATH}
   --ckpt_path "${SAVE_PATH}/ckpt"
   --save_hf_ckpt
   --max_ckpt_num 3
   --save_steps 10
)

ROLLOUT_ARGS=(
   --agent_func_path ${AGENT_FUNC_PATH}
   --prompt_data ${DATASET_PATH}
   --input_key question
   --label_key rubrics
   --prompt_max_len 10240
   --generate_max_len 64000
   --apply_chat_template
   --packing_samples

   --vllm_generate_batch_size 16
   --rollout_batch_size 128
   --n_samples_per_prompt 8
   --train_batch_size 1024

   --dynamic_filtering
   --dynamic_filtering_reward_range 0.0 1.0

   --use_dynamic_batch
   --train_max_tokens_per_gpu 16192
   --rollout_max_tokens_per_gpu 32768

   # --micro_train_batch_size 1
   # --micro_rollout_batch_size 8
   --max_samples 128000
   --max_epochs 1
   --num_episodes 1
)

ENGINE_ARGS=(
   --ref_num_nodes 1
   --ref_num_gpus_per_node 8
   --actor_num_nodes 1
   --actor_num_gpus_per_node 8
   --vllm_num_engines 2
   --vllm_tensor_parallel_size 4
   --vllm_gpu_memory_utilization 0.8
   --colocate_all_models
   --vllm_enable_sleep
   --deepspeed_enable_sleep
   --vllm_sync_backend nccl
   --enforce_eager

   --zero_stage 3
   --gradient_checkpointing
   # --adam_offload
   --ring_attn_size 4
   --ring_head_stride 2
   --param_dtype bf16
)

OPTIMIZER_ARGS=(
   --advantage_estimator reinforce_baseline
   --actor_learning_rate 5e-7
   # --critic_learning_rate 9e-6
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



python3 -m openrlhf.cli.train_ppo_ray \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${ENGINE_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${LOG_ARGS[@]}

# ray job submit --address="http://127.0.0.1:8265" \
#    -- python3 -m openrlhf.cli.train_ppo_ray \
#    ${CKPT_ARGS[@]} \
#    ${ROLLOUT_ARGS[@]} \
#    ${ENGINE_ARGS[@]} \
#    ${OPTIMIZER_ARGS[@]} \
#    ${LOG_ARGS[@]}
