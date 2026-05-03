#!/bin/bash
# Run on the login node (compute nodes have no internet).
# Usage: bash download_model.sh

set -ex

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-235B-A22B-Thinking}"
LOCAL_DIR="${LOCAL_DIR:-/mnt/data4/shasta/amar.amarjyoti/research_data/models/$(basename "$MODEL_ID")}"

module load miniforge
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/sandbox/amar.amarjyoti/conda_envs/rlvr_conda

mkdir -p "$LOCAL_DIR"

export HF_HUB_ENABLE_HF_TRANSFER=1
pip show hf_transfer >/dev/null 2>&1 || pip install -q hf_transfer

huggingface-cli download "$MODEL_ID" \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False \
    --resume-download

echo "Model downloaded to: $LOCAL_DIR"
