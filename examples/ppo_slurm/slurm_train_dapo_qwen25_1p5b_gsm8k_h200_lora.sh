#!/bin/bash
#SBATCH --job-name=pretrain_dapo_lora
#SBATCH --output=/mnt/sandbox/amar.amarjyoti/outputs/%j-%x.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --partition=gen4
#SBATCH --mem-per-gpu=60G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amar.amarjyoti@bluerivertech.com
#SBATCH --time=100:00:00

set -x

# ===== Environment setup =====
module load miniforge
mamba activate /mnt/sandbox/amar.amarjyoti/conda_envs/rlvr_conda

# ===== Project paths =====
WORK_DIR="/mnt/sandbox/amar.amarjyoti/research_code/OpenRLHF-prorl-research"
cd "$WORK_DIR"

export WANDB_API_KEY="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"
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH="openai/gsm8k#main"
REWARD_FUNC_PATH="${WORK_DIR}/examples/python/gsm8k_reward_func.py"
SAVE_PATH="${WORK_DIR}/exp/qwen25-1p5b-gsm8k-dapo-lora"

INPUT_TEMPLATE='<|im_start|>system
Please reason step by step before answering the question, and put your final answer within \\boxed{{}}.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
'

# ===== Ray setup (single-node) =====
# Start Ray head on this node using all 8 GPUs
RAY_PORT=6379
ray start --head \
    --port=$RAY_PORT \
    --num-gpus=8 \
    --block &
RAY_PID=$!

# Wait for Ray to initialize
sleep 15
echo "Ray head started on $(hostname), PID=$RAY_PID"
ray status

# ===== Run training =====
python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.35 \
   --pretrain ${MODEL_PATH} \
   --remote_rm_url ${REWARD_FUNC_PATH} \
   --save_path ${SAVE_PATH} \
   --ckpt_path "${SAVE_PATH}/ckpt" \
   --save_hf_ckpt \
   --target_modules all-linear \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 256 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --num_episodes 1 \
   --prompt_max_len 512 \
   --generate_max_len 512 \
   --max_samples 8000 \
   --zero_stage 2 \
   --eps_clip_low_high 0.2 0.28 \
   --param_dtype bf16 \
   --actor_learning_rate 5e-6 \
   --init_kl_coef 1e-3 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --dynamic_filtering \
   --dynamic_filtering_reward_range 0 1 \
   --prompt_data ${DATASET_PATH} \
   --prompt_split train \
   --input_key question \
   --label_key answer \
   --input_template "${INPUT_TEMPLATE}" \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --use_dynamic_batch \
   --train_max_tokens_per_gpu 32768 \
   --vllm_sync_backend nccl \
   --attn_implementation flash_attention_2 \
   --enforce_eager \
   --use_wandb ${WANDB_API_KEY} \
   --wandb_project openrlhf_gsm8k \
   --wandb_group qwen25_1p5b_dapo \
   --wandb_run_name dapo_qwen25_1p5b_gsm8k_$(date +%m%dT%H%M)

TRAIN_EXIT_CODE=$?

# ===== Cleanup =====
ray stop
echo "Training finished with exit code: $TRAIN_EXIT_CODE"
exit $TRAIN_EXIT_CODE
