#!/bin/bash
# Run on login node (compute nodes lack internet access).
# Usage: bash download_mavis.sh

set -e

export PATH="/mnt/sandbox/amar.amarjyoti/conda_envs/rlvr_conda/bin:$PATH"

DATA_DIR="/mnt/data4/shasta/amar.amarjyoti/research_data/raw/mavis"
LOG_DIR="/mnt/data4/shasta/amar.amarjyoti/research_data/logs"
mkdir -p "$DATA_DIR" "$LOG_DIR"

export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=== Downloading MAVIS (CaraJ/MAVIS-Geometry + CaraJ/MAVIS-Function) ==="
echo "Start time: $(date)"
echo "Target: $DATA_DIR"

for repo in CaraJ/MAVIS-Geometry CaraJ/MAVIS-Function; do
    sub="${repo#CaraJ/}"
    echo ""
    echo "--- $repo ---"
    for i in 1 2 3; do
        echo "Attempt $i..."
        huggingface-cli download "$repo" \
            --repo-type dataset \
            --local-dir "$DATA_DIR/$sub" && break
        echo "Attempt $i failed, retrying in 60s..."
        sleep 60
    done
done

echo ""
echo "=== Download complete ==="
echo "End time: $(date)"
du -sh "$DATA_DIR"
ls -la "$DATA_DIR"
