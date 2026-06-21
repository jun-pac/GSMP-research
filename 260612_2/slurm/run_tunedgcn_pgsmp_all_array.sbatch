#!/usr/bin/env bash
#SBATCH --job-name=tunedgcn-pgsmp-all
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-1%1

set -euo pipefail

# EDIT THIS SECTION FOR YOUR CLUSTER
set +u
source ~/.bashrc || true
set -u
if [[ -n "${CONDA_ENV:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    conda activate "$CONDA_ENV"
  else
    echo "[WARN] CONDA_ENV=$CONDA_ENV was set, but conda is not available; using PYTHON_BIN/current python."
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXP_ROOT="${EXP_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$EXP_ROOT/.." && pwd)"
LARGE_GRAPH="$EXP_ROOT/upstream/tunedGNN/large_graph"

mkdir -p "$EXP_ROOT/logs" "$EXP_ROOT/results/tunedgcn_pgsmp" "$EXP_ROOT/cache/pgsmp"

cd "$LARGE_GRAPH"
hostname
date
pwd
git rev-parse HEAD || true
nvidia-smi || true
"$PYTHON_BIN" --version
"$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
"$PYTHON_BIN" -c "import torch_geometric; print('pyg', torch_geometric.__version__)" || true
"$PYTHON_BIN" -c "import dgl; print('dgl', dgl.__version__)" || true

read -r -a SEED_ARRAY <<< "${SEEDS:-0}"
NUM_SEEDS="${#SEED_ARRAY[@]}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
METHOD_ID=$(( TASK_ID / NUM_SEEDS ))
SEED_ID=$(( TASK_ID % NUM_SEEDS ))

if (( METHOD_ID > 1 )); then
  echo "[SKIP] task_id=$TASK_ID exceeds baseline+P-GSMP grid for $NUM_SEEDS seeds"
  exit 0
fi

SEED="${SEED_ARRAY[$SEED_ID]}"
METHOD_ARGS=()
METHOD_NAME="baseline"
if (( METHOD_ID == 1 )); then
  METHOD_NAME="pgsmp"
  METHOD_ARGS+=(
    --use-pgsmp
    --pgsmp-alpha "${PGSMP_ALPHA:-0.5}"
    --pgsmp-depth "${PGSMP_DEPTH:-1}"
    --pgsmp-norm "${PGSMP_NORM:-strict_observed}"
    --pgsmp-self-mode "${PGSMP_SELF_MODE:-neighbor_only}"
  )
  if [[ "${PGSMP_FORCE_RECOMPUTE:-0}" == "1" ]]; then
    METHOD_ARGS+=(--pgsmp-force-recompute)
  fi
fi

EPOCHS="${EPOCHS:-2000}"
EVAL_EVERY="${EVAL_EVERY:-10}"
LOG_EVERY="${LOG_EVERY:-10}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
PGSMP_CHUNK_SIZE="${PGSMP_CHUNK_SIZE:-200000}"

echo "[RUN] method=$METHOD_NAME seed=$SEED epochs=$EPOCHS"

"$PYTHON_BIN" main-arxiv.py \
  --dataset ogbn-arxiv \
  --data_dir "$DATA_DIR" \
  --hidden_channels 512 \
  --epochs "$EPOCHS" \
  --lr 0.0005 \
  --runs 1 \
  --local_layers 5 \
  --bn \
  --res \
  --device "${DEVICE:-0}" \
  --seed "$SEED" \
  --eval-every "$EVAL_EVERY" \
  --log-every "$LOG_EVERY" \
  --pgsmp-cache-dir "$EXP_ROOT/cache/pgsmp" \
  --pgsmp-chunk-size "$PGSMP_CHUNK_SIZE" \
  --result-root "$EXP_ROOT/results/tunedgcn_pgsmp" \
  "${METHOD_ARGS[@]}"
