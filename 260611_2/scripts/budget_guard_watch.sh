#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${1:?usage: budget_guard_watch.sh LOG_FILE JOB_ID}"
JOB_ID="${2:?usage: budget_guard_watch.sh LOG_FILE JOB_ID}"
PATTERN="${BUDGET_GUARD_PATTERN:-STOPPED_BUDGET_GUARD|<<<<<<<<<< LM-Pretraining|<<<<<<<<<< LM-Pre-train Inference|Loading raw text|titleabs.tsv|trainLM.py|infLM.py}"
INTERVAL="${BUDGET_GUARD_INTERVAL:-5}"
MAX_SECONDS="${BUDGET_GUARD_MAX_SECONDS:-0}"
MAX_GPU_MEM_MB="${BUDGET_GUARD_MAX_GPU_MEM_MB:-0}"
MAX_RSS_MB="${BUDGET_GUARD_MAX_RSS_MB:-0}"
BREACHES_REQUIRED="${BUDGET_GUARD_BREACHES:-2}"
START_TS="$(date +%s)"

cancel_job() {
  local reason="$1"
  echo "[BUDGET_GUARD] $reason; cancelling job $JOB_ID"
  scancel "$JOB_ID" || true
  exit 0
}

job_active() {
  [[ -n "$(squeue -h -j "$JOB_ID" 2>/dev/null || true)" ]]
}

gpu_mem_mb() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return
  fi
  local output=""
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES:-}" != "NoDevFiles" ]]; then
    output="$(nvidia-smi --id="${CUDA_VISIBLE_DEVICES}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
  fi
  if [[ -z "$output" ]]; then
    output="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
  fi
  awk '{sum += $1} END {printf "%d\n", sum + 0}' <<<"$output"
}

python_rss_mb() {
  ps -u "${USER:-$(id -un)}" -o rss=,comm= 2>/dev/null \
    | awk '$2 ~ /^python/ {sum += $1} END {printf "%d\n", sum / 1024}'
}

echo "[BUDGET_GUARD] watching $LOG_FILE; job=$JOB_ID interval=${INTERVAL}s max_seconds=$MAX_SECONDS max_gpu_mem_mb=$MAX_GPU_MEM_MB max_python_rss_mb=$MAX_RSS_MB"
gpu_breaches=0
rss_breaches=0
while true; do
  if [[ -f "$LOG_FILE" ]] && grep -E "$PATTERN" "$LOG_FILE" >/dev/null 2>&1; then
    cancel_job "sentinel found in $LOG_FILE"
  fi

  now_ts="$(date +%s)"
  elapsed=$((now_ts - START_TS))
  if (( MAX_SECONDS > 0 && elapsed > MAX_SECONDS )); then
    cancel_job "elapsed ${elapsed}s exceeded BUDGET_GUARD_MAX_SECONDS=$MAX_SECONDS"
  fi

  if (( MAX_GPU_MEM_MB > 0 )); then
    used_gpu_mem="$(gpu_mem_mb)"
    if (( used_gpu_mem > MAX_GPU_MEM_MB )); then
      gpu_breaches=$((gpu_breaches + 1))
      echo "[BUDGET_GUARD] gpu_mem_mb=$used_gpu_mem exceeds max=$MAX_GPU_MEM_MB breach=$gpu_breaches/$BREACHES_REQUIRED"
    else
      gpu_breaches=0
    fi
    if (( gpu_breaches >= BREACHES_REQUIRED )); then
      cancel_job "GPU memory ${used_gpu_mem}MB exceeded ${MAX_GPU_MEM_MB}MB for ${gpu_breaches} checks"
    fi
  fi

  if (( MAX_RSS_MB > 0 )); then
    used_rss="$(python_rss_mb)"
    if (( used_rss > MAX_RSS_MB )); then
      rss_breaches=$((rss_breaches + 1))
      echo "[BUDGET_GUARD] python_rss_mb=$used_rss exceeds max=$MAX_RSS_MB breach=$rss_breaches/$BREACHES_REQUIRED"
    else
      rss_breaches=0
    fi
    if (( rss_breaches >= BREACHES_REQUIRED )); then
      cancel_job "Python RSS ${used_rss}MB exceeded ${MAX_RSS_MB}MB for ${rss_breaches} checks"
    fi
  fi

  if ! job_active; then
    exit 0
  fi
  sleep "$INTERVAL"
done
