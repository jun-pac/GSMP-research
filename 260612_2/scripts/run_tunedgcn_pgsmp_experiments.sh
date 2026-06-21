#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$EXP_ROOT/.." && pwd)"
LARGE_GRAPH="$EXP_ROOT/upstream/tunedGNN/large_graph"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
RESULT_ROOT="${RESULT_ROOT:-$EXP_ROOT/results/tunedgcn_pgsmp}"
CACHE_DIR="${CACHE_DIR:-$EXP_ROOT/cache/pgsmp}"
DEVICE="${DEVICE:-0}"
SEEDS="${SEEDS:-0}"
PGSMP_ALPHA="${PGSMP_ALPHA:-0.5}"
PGSMP_DEPTH="${PGSMP_DEPTH:-1}"
PGSMP_NORM="${PGSMP_NORM:-strict_observed}"
PGSMP_SELF_MODE="${PGSMP_SELF_MODE:-neighbor_only}"
PGSMP_CHUNK_SIZE="${PGSMP_CHUNK_SIZE:-200000}"

CPU_ARGS=()
if [[ "${CPU:-0}" == "1" ]]; then
  CPU_ARGS+=(--cpu)
fi

FORCE_ARGS=()
if [[ "${PGSMP_FORCE_RECOMPUTE:-0}" == "1" ]]; then
  FORCE_ARGS+=(--pgsmp-force-recompute)
fi

run_one() {
  local seed="$1"
  local epochs="$2"
  local eval_every="$3"
  local log_every="$4"
  shift 4

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
      --pgsmp-cache-dir "$CACHE_DIR" \
      --pgsmp-chunk-size "$PGSMP_CHUNK_SIZE" \
      --result-root "$RESULT_ROOT" \
      "${CPU_ARGS[@]}" \
      "$@"
  )
}

first_seed() {
  read -r seed _ <<< "$SEEDS"
  printf '%s\n' "${seed:-0}"
}

pgsmp_args() {
  printf '%s\n' \
    --use-pgsmp \
    --pgsmp-alpha "$PGSMP_ALPHA" \
    --pgsmp-depth "$PGSMP_DEPTH" \
    --pgsmp-norm "$PGSMP_NORM" \
    --pgsmp-self-mode "$PGSMP_SELF_MODE" \
    "${FORCE_ARGS[@]}"
}

case "$MODE" in
  smoke)
    SEED="$(first_seed)"
    EPOCHS="${EPOCHS:-3}"
    EVAL_EVERY="${EVAL_EVERY:-1}"
    LOG_EVERY="${LOG_EVERY:-1}"
    echo "[RUN] smoke baseline seed=$SEED epochs=$EPOCHS"
    run_one "$SEED" "$EPOCHS" "$EVAL_EVERY" "$LOG_EVERY"
    echo "[RUN] smoke P-GSMP seed=$SEED epochs=$EPOCHS alpha=$PGSMP_ALPHA depth=$PGSMP_DEPTH norm=$PGSMP_NORM self=$PGSMP_SELF_MODE"
    mapfile -t EXTRA_ARGS < <(pgsmp_args)
    run_one "$SEED" "$EPOCHS" "$EVAL_EVERY" "$LOG_EVERY" "${EXTRA_ARGS[@]}"
    ;;
  baseline)
    EPOCHS="${EPOCHS:-2000}"
    EVAL_EVERY="${EVAL_EVERY:-10}"
    LOG_EVERY="${LOG_EVERY:-10}"
    for seed in $SEEDS; do
      echo "[RUN] tunedGNN GCN baseline seed=$seed epochs=$EPOCHS"
      run_one "$seed" "$EPOCHS" "$EVAL_EVERY" "$LOG_EVERY"
    done
    ;;
  pgsmp)
    EPOCHS="${EPOCHS:-2000}"
    EVAL_EVERY="${EVAL_EVERY:-10}"
    LOG_EVERY="${LOG_EVERY:-10}"
    mapfile -t EXTRA_ARGS < <(pgsmp_args)
    for seed in $SEEDS; do
      echo "[RUN] tunedGNN GCN+P-GSMP seed=$seed epochs=$EPOCHS alpha=$PGSMP_ALPHA depth=$PGSMP_DEPTH norm=$PGSMP_NORM self=$PGSMP_SELF_MODE"
      run_one "$seed" "$EPOCHS" "$EVAL_EVERY" "$LOG_EVERY" "${EXTRA_ARGS[@]}"
    done
    ;;
  all)
    if command -v sbatch >/dev/null 2>&1; then
      (cd "$EXP_ROOT" && sbatch slurm/run_tunedgcn_pgsmp_all_array.sbatch)
    else
      if [[ "${LOCAL_ALL:-0}" != "1" ]]; then
        echo "[ERROR] sbatch not found. To avoid accidental long local runs, set LOCAL_ALL=1 to run all locally." >&2
        exit 2
      fi
      echo "[WARN] sbatch not found; LOCAL_ALL=1 set, running baseline then P-GSMP sequentially"
      "$0" baseline
      "$0" pgsmp
    fi
    ;;
  *)
    echo "Usage: $0 {smoke|baseline|pgsmp|all}" >&2
    exit 2
    ;;
esac
