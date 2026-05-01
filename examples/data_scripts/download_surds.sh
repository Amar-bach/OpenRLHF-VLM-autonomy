#!/bin/bash
#SBATCH --job-name=pretrain_model_21
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/data4/shasta/amar.amarjyoti/research_data/logs/job_21_%j.log

DATA_DIR="/mnt/data4/shasta/amar.amarjyoti/research_data/raw/surds"
mkdir -p "$DATA_DIR"
mkdir -p /mnt/data4/shasta/amar.amarjyoti/research_data/logs

export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=== Downloading SURDS (full: bonbon-rj/SURDS) ==="
echo "Start time: $(date)"
echo "Target: $DATA_DIR"

# Retry up to 3 times with resume support
for i in 1 2 3; do
    echo "Attempt $i..."
    huggingface-cli download bonbon-rj/SURDS \
        --repo-type dataset \
        --local-dir "$DATA_DIR" && break
    echo "Attempt $i failed, retrying in 60s..."
    sleep 60
done

echo ""
echo "=== Downloading SURDS eval (bonbon-rj/SURDS_eval) ==="

for i in 1 2 3; do
    echo "Attempt $i..."
    huggingface-cli download bonbon-rj/SURDS_eval \
        --repo-type dataset \
        --local-dir "$DATA_DIR/eval_only" && break
    echo "Attempt $i failed, retrying in 60s..."
    sleep 60
done

echo ""
echo "=== Download complete ==="
echo "End time: $(date)"
echo "Disk usage:"
du -sh "$DATA_DIR"
ls -la "$DATA_DIR"
