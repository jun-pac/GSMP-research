#!/usr/bin/env bash
#SBATCH --job-name=tunedgcn-gsmp1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-0%1

set -euo pipefail

# EDIT THIS SECTION FOR YOUR CLUSTER
set +u
source ~/.bashrc || true
set -u
if [[ -n "${CONDA_ENV:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "CONDA_ENV=$CONDA_ENV was set, but conda is not available." >&2
    exit 2
  fi
  conda activate "$CONDA_ENV"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXP_ROOT="${EXP_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$EXP_ROOT/.." && pwd)"
LARGE_GRAPH="$EXP_ROOT/upstream/tunedGNN/large_graph"

mkdir -p "$EXP_ROOT/logs" "$EXP_ROOT/results/tunedgcn_gsmp" "$EXP_ROOT/cache/gsmp"

cd "$LARGE_GRAPH"
hostname
date
pwd
git rev-parse HEAD || true
nvidia-smi || true
"$PYTHON_BIN" --version
"$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
"$PYTHON_BIN" -c "import torch_geometric; print('pyg', torch_geometric.__version__)" || true
"$PYTHON_BIN" -c "import importlib.util; spec=importlib.util.find_spec('dgl'); print('dgl', 'not installed' if spec is None else 'installed')" || true

read -r -a SEED_ARRAY <<< "${SEEDS:-0 1 2 3 4}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID >= ${#SEED_ARRAY[@]} )); then
  echo "[SKIP] task_id=$TASK_ID has no seed in SEEDS=${SEEDS:-0 1 2 3 4}"
  exit 0
fi
SEED="${SEED_ARRAY[$TASK_ID]}"

EPOCHS="${EPOCHS:-2000}"
EVAL_EVERY="${EVAL_EVERY:-1}"
LOG_EVERY="${LOG_EVERY:-50}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
GSMP_NORM="${GSMP_NORM:-scale_preserve}"
GCN_GSMP_MODE="${GCN_GSMP_MODE:-weighted_gcn_norm}"

echo "[RUN] method=GCN+GSMP seed=$SEED epochs=$EPOCHS apply=first_layer norm=$GSMP_NORM mode=$GCN_GSMP_MODE"

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
  --gsmp-cache-dir "$EXP_ROOT/cache/gsmp" \
  --result-root "$EXP_ROOT/results/tunedgcn_gsmp" \
  --use-gsmp \
  --gsmp-norm "$GSMP_NORM" \
  --gcn-gsmp-mode "$GCN_GSMP_MODE" \
  --gsmp-apply first_layer
