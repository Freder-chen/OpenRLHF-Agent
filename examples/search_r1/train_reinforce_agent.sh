#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../..")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export OPENRLHF_ASYNC_NUM_TASKS=128
export NCCL_IB_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_SOCKET_TIMEOUT=1800000

MODEL_PATH="/mnt/workspace/liaoxinye/pretrain_models/Qwen/Qwen3-4B-Thinking-2507"
SAVE_PATH="${WORK_DIR}/exp/Qwen3-test"
AGNET_FUNC_PATH="${WORK_DIR}/examples/search_r1/agent_func.py"
DATASET_PATH="/mnt/workspace/liaoxinye/Search-R1/data/nq_hotpotqa_train" # TODO: set your dataset path here

set -x

# ray job submit --address="http://127.0.0.1:8265" \
   python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --vllm_gpu_memory_utilization 0.7 \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --colocate_all_models \
   --vllm_sync_backend nccl \
   --pretrain ${MODEL_PATH} \
   --save_path ${SAVE_PATH} \
   --ckpt_path "${SAVE_PATH}/ckpt" \
   --save_hf_ckpt \
   --load_checkpoint \
   --save_steps 10 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 64 \
   --n_samples_per_prompt 2 \
   --max_samples 128000 \
   --max_epochs 1 \
   --num_episodes 1 \
   --prompt_max_len 10240 \
   --generate_max_len 32000 \
   --zero_stage 3 \
   --ring_attn_size 2 \
   --ring_head_stride 2 \
   --gradient_checkpointing \
   --bf16 \
   --advantage_estimator reinforce_baseline \
   --actor_learning_rate 5e-7 \
   --entropy_loss_coef 0.00 \
   --init_kl_coef 0.00001 \
   --use_kl_loss \
   --kl_estimator k2 \
   --prompt_data ${DATASET_PATH} \
   --apply_chat_template \
   --input_key question \
   --label_key golden_answers \
   --normalize_reward \
   --packing_samples \
   --agent_func_path ${AGNET_FUNC_PATH} \
   --use_tensorboard ${SAVE_PATH}/runs
