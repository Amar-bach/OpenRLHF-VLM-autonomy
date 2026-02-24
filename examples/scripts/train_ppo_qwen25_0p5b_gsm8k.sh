#!/bin/bash
set -x

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
WANDB_API_KEY="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"
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
# GSM8K has multiple configs; specify one explicitly (main/socratic) to avoid builder_config errors.
DATASET_PATH="openai/gsm8k#main"
REWARD_FUNC_PATH="${WORK_DIR}/examples/python/gsm8k_reward_func.py"
SAVE_PATH="${WORK_DIR}/exp/qwen25-0p5b-gsm8k-ppo"

python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4\
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.20 \
   --pretrain ${MODEL_PATH} \
   --remote_rm_url ${REWARD_FUNC_PATH} \
   --save_path ${SAVE_PATH} \
   --ckpt_path "${SAVE_PATH}/ckpt" \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 16 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 32 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --num_episodes 1 \
   --prompt_max_len 512 \
   --generate_max_len 256 \
   --max_samples 2000 \
   --zero_stage 2 \
   --param_dtype bf16 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 5e-6 \
   --init_kl_coef 0.01 \
   --prompt_data ${DATASET_PATH} \
   --prompt_split train \
   --input_key question \
   --label_key answer \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --use_dynamic_batch \
   --train_max_tokens_per_gpu 8192 \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --use_wandb ${WANDB_API_KEY} \
   --wandb_project openrlhf_gsm8k \
   --wandb_group qwen25_0p5b \
   --wandb_run_name ppo_qwen25_0p5b_gsm8k_$(date +%m%dT%H%M) \
