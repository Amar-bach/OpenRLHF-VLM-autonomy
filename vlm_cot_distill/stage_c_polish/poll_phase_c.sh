#!/usr/bin/env bash
# Background poller: every 5 minutes check Phase C output line count, run the
# audit script when a new 10% milestone is crossed, append to a log.
set -euo pipefail

JOB_ID="${JOB_ID:?set JOB_ID}"
DATA_DIR="${DATA_DIR:-/mnt/data4/shasta/amar.amarjyoti/research_data/vlm_cot_distill}"
OUT="${OUT:-${DATA_DIR}/phase_c_${JOB_ID}_deepseekv4_c2.jsonl}"
STAGE_B="${STAGE_B:-${DATA_DIR}/judge_1778033257_qwen32b.jsonl}"
STAGE_A="${STAGE_A:-${DATA_DIR}/cot_1058163_Qwen3-VL-235B-A22B-Thinking-FP8_train_N16_T0.8_grounding.jsonl}"
TOTAL="${TOTAL:-40645}"
WORK="/mnt/sandbox/amar.amarjyoti/research_code/OpenRLHF-prorl-research/vlm_cot_distill/stage_c_polish"
LOG="${LOG:-/mnt/sandbox/amar.amarjyoti/outputs/phase_c_${JOB_ID}_audit.log}"
SLEEP="${SLEEP:-300}"
MIN_FIRST_AUDIT="${MIN_FIRST_AUDIT:-20}"
FAST_SLEEP="${FAST_SLEEP:-30}"

WANDB_PROJECT="${WANDB_PROJECT:-vision cot distillation}"
WANDB_RUN_ID="${WANDB_RUN_ID:-phase_c_${JOB_ID}_c2}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-phase_c_${JOB_ID}_c2}"

# Use rlvr_conda python (has wandb installed); fall back to system python3.
PY="/mnt/sandbox/amar.amarjyoti/conda_envs/rlvr_conda/bin/python3"
[[ -x "$PY" ]] || PY="python3"

# Load WANDB_API_KEY (and any other secrets) from project .env if present.
ENV_FILE="${ENV_FILE:-/mnt/sandbox/amar.amarjyoti/research_code/OpenRLHF-prorl-research/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

mkdir -p "$(dirname "$LOG")"
last_bucket=-1

echo "[poller] started; job=$JOB_ID  out=$OUT  total=$TOTAL  log=$LOG" | tee -a "$LOG"

while true; do
  # Job state: stop polling once the job leaves the queue.
  state=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || true)
  [[ -z "$state" ]] && state="GONE"

  if [[ ! -f "$OUT" ]]; then
    n=0
  else
    n=$(wc -l < "$OUT")
  fi
  bucket=$(( n * 10 / TOTAL ))

  # Before first audit: wait on a faster cadence until we cross MIN_FIRST_AUDIT.
  if (( last_bucket == -1 && n < MIN_FIRST_AUDIT )) && [[ "$state" != "GONE" ]]; then
    sleep "$FAST_SLEEP"
    continue
  fi

  if [[ "$bucket" -gt "$last_bucket" || "$state" == "GONE" ]]; then
    {
      echo
      echo "===== $(date -Iseconds)  job_state=$state  lines=$n  bucket=${bucket}0% ====="
      "$PY" "$WORK/audit_phase_c.py" \
        --phase-c-jsonl "$OUT" \
        --stage-b-jsonl "$STAGE_B" \
        --stage-a-jsonl "$STAGE_A" \
        --total-expected "$TOTAL" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-id  "$WANDB_RUN_ID" \
        --wandb-run-name "$WANDB_RUN_NAME" \
        --wandb-step    "$bucket" 2>&1 || echo "[audit failed]"
    } | tee -a "$LOG"
    last_bucket="$bucket"
  fi

  [[ "$state" == "GONE" ]] && { echo "[poller] job gone, exiting" | tee -a "$LOG"; break; }
  sleep "$SLEEP"
done
