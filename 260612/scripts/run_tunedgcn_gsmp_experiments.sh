#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$EXP_ROOT/.." && pwd)"
LARGE_GRAPH="$EXP_ROOT/upstream/tunedGNN/large_graph"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
RESULT_ROOT="${RESULT_ROOT:-$EXP_ROOT/results/tunedgcn_gsmp}"
CACHE_DIR="${CACHE_DIR:-$EXP_ROOT/cache/gsmp}"
DEVICE="${DEVICE:-0}"
SEEDS="${SEEDS:-0}"
GSMP_NORM="${GSMP_NORM:-scale_preserve}"
GCN_GSMP_MODE="${GCN_GSMP_MODE:-weighted_gcn_norm}"
GSMP_APPLY="${GSMP_APPLY:-all_layers}"

CPU_ARGS=()
if [[ "${CPU:-0}" == "1" ]]; then
  CPU_ARGS+=(--cpu)
fi

run_one() {
  local method="$1"
  local seed="$2"
  local epochs="$3"
  local eval_every="$4"
  local log_every="$5"
  shift 5

  mkdir -p "$RESULT_ROOT" "$CACHE_DIR"
  (
    cd "$LARGE_GRAPH"
    "$PYTHON_BIN" main-arxiv.py \
      --dataset ogbn-arxiv \
      --data_dir "$DATA_DIR" \
      --hidden_channels 512 \
      --epochs "$epochs" \
      --lr 0.0005 \
      --runs 1 \
      --local_layers 5 \
      --bn \
      --res \
      --device "$DEVICE" \
      --seed "$seed" \
      --eval-every "$eval_every" \
      --log-every "$log_every" \
      --gsmp-cache-dir "$CACHE_DIR" \
      --result-root "$RESULT_ROOT" \
      "${CPU_ARGS[@]}" \
      "$@"
  )
}

first_seed() {
  read -r seed _ <<< "$SEEDS"
  printf '%s\n' "${seed:-0}"
}

case "$MODE" in
  smoke)
    SEED="$(first_seed)"
    EPOCHS="${EPOCHS:-3}"
    EVAL_EVERY="${EVAL_EVERY:-1}"
    LOG_EVERY="${LOG_EVERY:-1}"
    echo "[RUN] smoke baseline seed=$SEED epochs=$EPOCHS"
    run_one baseline "$SEED" "$EPOCHS" "$EVAL_EVERY" "$LOG_EVERY"
    echo "[RUN] smoke GSMP seed=$SEED epochs=$EPOCHS"
    run_one gsmp "$SEED" "$EPOCHS" "$EVAL_EVERY" "$LOG_EVERY" \
      --use-gsmp \
      --gsmp-norm "$GSMP_NORM" \
      --gcn-gsmp-mode "$GCN_GSMP_MODE" \
      --gsmp-apply "$GSMP_APPLY"
    ;;
  baseline)
    EPOCHS="${EPOCHS:-2000}"
    EVAL_EVERY="${EVAL_EVERY:-10}"
    LOG_EVERY="${LOG_EVERY:-10}"
    for seed in $SEEDS; do
      echo "[RUN] tunedGNN GCN baseline seed=$seed epochs=$EPOCHS"
      run_one baseline "$seed" "$EPOCHS" "$EVAL_EVERY" "$LOG_EVERY"
    done
    ;;
  gsmp)
    EPOCHS="${EPOCHS:-2000}"
    EVAL_EVERY="${EVAL_EVERY:-10}"
    LOG_EVERY="${LOG_EVERY:-10}"
    for seed in $SEEDS; do
      echo "[RUN] tunedGNN GCN+GSMP seed=$seed epochs=$EPOCHS norm=$GSMP_NORM mode=$GCN_GSMP_MODE apply=$GSMP_APPLY"
      run_one gsmp "$seed" "$EPOCHS" "$EVAL_EVERY" "$LOG_EVERY" \
        --use-gsmp \
        --gsmp-norm "$GSMP_NORM" \
        --gcn-gsmp-mode "$GCN_GSMP_MODE" \
        --gsmp-apply "$GSMP_APPLY"
    done
    ;;
  all)
    if command -v sbatch >/dev/null 2>&1; then
      (cd "$EXP_ROOT" && sbatch slurm/run_tunedgcn_all_array.sbatch)
    else
      echo "[WARN] sbatch not found; running baseline then GSMP sequentially"
      "$0" baseline
      "$0" gsmp
    fi
    ;;
  *)
    echo "Usage: $0 {smoke|baseline|gsmp|all}" >&2
    exit 2
    ;;
esac
