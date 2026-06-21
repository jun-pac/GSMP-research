#!/usr/bin/env bash
#SBATCH --job-name=tunedgcn-base
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

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

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-2000}"
EVAL_EVERY="${EVAL_EVERY:-10}"
LOG_EVERY="${LOG_EVERY:-10}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"

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
  --result-root "$EXP_ROOT/results/tunedgcn_pgsmp"
