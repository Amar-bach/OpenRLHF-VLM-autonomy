#!/bin/bash
# Run on login node (compute nodes lack internet access).
# Usage: bash download_mulberry.sh
# Pulls Mulberry-SFT images so the Vision-R1-cold mulberry portion is usable.

set -e

export PATH="/mnt/sandbox/amar.amarjyoti/conda_envs/rlvr_conda/bin:$PATH"

DATA_DIR="/mnt/data4/shasta/amar.amarjyoti/research_data/raw/mulberry"
LOG_DIR="/mnt/data4/shasta/amar.amarjyoti/research_data/logs"
mkdir -p "$DATA_DIR" "$LOG_DIR"

export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=== Downloading HuanjinYao/Mulberry-SFT ==="
echo "Start time: $(date)"
echo "Target: $DATA_DIR"

for i in 1 2 3; do
    echo "Attempt $i..."
    huggingface-cli download HuanjinYao/Mulberry-SFT \
        --repo-type dataset \
        --local-dir "$DATA_DIR" && break
    echo "Attempt $i failed, retrying in 60s..."
    sleep 60
done

echo ""
echo "=== Extracting mulberry_images.tar ==="
tar -xf "$DATA_DIR/mulberry_images.tar" -C "$DATA_DIR"

echo ""
echo "=== Symlinking into vision_r1_cold for path resolution ==="
VR1_DIR="/mnt/data4/shasta/amar.amarjyoti/research_data/raw/vision_r1_cold"
rm -rf "$VR1_DIR/mulberry_images"
ln -s "$DATA_DIR/mulberry_images" "$VR1_DIR/mulberry_images"

echo ""
echo "=== Download complete ==="
echo "End time: $(date)"
du -sh "$DATA_DIR"
ls -la "$DATA_DIR"
