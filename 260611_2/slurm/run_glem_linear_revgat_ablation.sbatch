#!/usr/bin/env bash
#SBATCH --job-name=glem-linear
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --chdir=/users/PAS1289/jyp531/GSMP-research/260611_2
#SBATCH --account=PAS1289
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
##SBATCH --partition=EDIT_ME
##SBATCH --constraint=EDIT_ME_V100_32GB_IF_NEEDED

set -euo pipefail

hostname
date
pwd
git -C upstream/GLEM rev-parse HEAD || true
nvidia-smi || true
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())" || true

# EDIT THIS SECTION FOR YOUR CLUSTER
export VIRTUAL_ENV="${GLEM_VENV:-/users/PAS1289/jyp531/GSMP-research/.venv}"
export PATH="$VIRTUAL_ENV/bin:$PATH"
echo "[ENV] python=$(command -v python)"
python -c "import torch, dgl, transformers, datasets, ogb; print('[ENV] deps ok')"

SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
PROFILE="${PROFILE:-full}"
if [[ "$PROFILE" == "smoke" ]]; then
  export BUDGET_GUARD_MAX_SECONDS="${BUDGET_GUARD_MAX_SECONDS:-600}"
  export BUDGET_GUARD_MAX_GPU_MEM_MB="${BUDGET_GUARD_MAX_GPU_MEM_MB:-14500}"
  export BUDGET_GUARD_MAX_RSS_MB="${BUDGET_GUARD_MAX_RSS_MB:-22000}"
else
  export BUDGET_GUARD_MAX_SECONDS="${BUDGET_GUARD_MAX_SECONDS:-7200}"
  export BUDGET_GUARD_MAX_GPU_MEM_MB="${BUDGET_GUARD_MAX_GPU_MEM_MB:-20000}"
  export BUDGET_GUARD_MAX_RSS_MB="${BUDGET_GUARD_MAX_RSS_MB:-32000}"
fi
GUARD_LOG="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
bash scripts/budget_guard_watch.sh "$GUARD_LOG" "$SLURM_JOB_ID" &
GUARD_PID=$!
trap 'kill "$GUARD_PID" 2>/dev/null || true' EXIT
PROFILE="$PROFILE" SEED="$SEED" bash scripts/run_one_glem_gnn.sh linear
