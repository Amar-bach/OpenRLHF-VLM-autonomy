#!/bin/bash
# Run on login node (compute nodes lack internet access).
# Usage: bash download_mathverse.sh

set -e

export PATH="/mnt/sandbox/amar.amarjyoti/conda_envs/rlvr_conda/bin:$PATH"

DATA_DIR="/mnt/data4/shasta/amar.amarjyoti/research_data/raw/mathverse"
LOG_DIR="/mnt/data4/shasta/amar.amarjyoti/research_data/logs"
mkdir -p "$DATA_DIR" "$LOG_DIR"

export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=== Downloading MathVerse (AI4Math/MathVerse) ==="
echo "Start time: $(date)"
echo "Target: $DATA_DIR"

for i in 1 2 3; do
    echo "Attempt $i..."
    huggingface-cli download AI4Math/MathVerse \
        --repo-type dataset \
        --local-dir "$DATA_DIR" && break
    echo "Attempt $i failed, retrying in 60s..."
    sleep 60
done

echo ""
echo "=== Download complete ==="
echo "End time: $(date)"
du -sh "$DATA_DIR"
ls -la "$DATA_DIR"
