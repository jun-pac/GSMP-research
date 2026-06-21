#!/bin/bash
# Hyperparameter sweep for the fixed E5 mini-batch ogbn-arxiv setup.
#
# Default grid is 36 jobs:
#   modes: smp gsmp
#   label_smoothing: 0.2 0.3 0.4
#   dropout: 0.3 0.4
#   weight_decay: 1e-6 2e-6 4e-6
#   lr: 0.01
#   batch_size: 4096
#   num_neighbors: 15,10
#   seed: 1
#
# Submit from:
#   cd /users/PAS1289/jyp531/GSMP-research/ogbn_arxiv_temporal
#   mkdir -p logs
#   export FEATURES_PATH=/users/PAS1289/jyp531/GSMP-research/SimTeG/out/ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt
#   sbatch slurm/sweep_arxiv_sage_smp_gsmp.sh
#
# Optional overrides before sbatch:
#   export LABEL_SMOOTHING_GRID="0.1 0.2 0.3 0.4"
#   export LR_GRID="0.005 0.01"
#   export SEED_GRID="1 2 3"
# If you expand the grid past 36 jobs, submit with a matching --array.

#SBATCH --job-name=arxiv_sage_sweep
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-35%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/train_arxiv_sage_temporal.py" ]]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  PROJECT_DIR="/users/PAS1289/jyp531/GSMP-research/ogbn_arxiv_temporal"
fi
cd "${PROJECT_DIR}"
mkdir -p logs
echo "project_dir=${PROJECT_DIR}"
echo "starting environment setup"

# --- Environment: edit this block for your cluster ---
VENV_DIR="${PROJECT_DIR}/../.venv"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
  echo "Activated venv: ${VENV_DIR}"
else
  if command -v module >/dev/null 2>&1; then
    module load miniconda3/24.1.2-py310 2>/dev/null || true
  fi
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
      source "${CONDA_BASE}/etc/profile.d/conda.sh"
    fi
    ENV_NAME="${CONDA_ENV_NAME:-graphenv}"
    conda activate "${ENV_NAME}"
    echo "Activated conda env: ${ENV_NAME}"
  else
    echo "ERROR: neither ${VENV_DIR} nor conda is available." >&2
    exit 1
  fi
fi
# --- End environment block ---

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "hostname=$(hostname)"
echo "date=$(date)"
echo "project_dir=${PROJECT_DIR}"
echo "slurm_job_id=${SLURM_JOB_ID:-none}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-0}"
nvidia-smi || true

python - <<'PY'
import sys
print("python", sys.version.replace("\n", " "))
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_check_failed", repr(exc))
    raise
PY

read -r -a MODES <<< "${MODES_GRID:-smp gsmp}"
read -r -a LABEL_SMOOTHINGS <<< "${LABEL_SMOOTHING_GRID:-0.2 0.3 0.4}"
read -r -a DROPOUTS <<< "${DROPOUT_GRID:-0.3 0.4}"
read -r -a WEIGHT_DECAYS <<< "${WEIGHT_DECAY_GRID:-1e-6 2e-6 4e-6}"
read -r -a LRS <<< "${LR_GRID:-0.01}"
read -r -a BATCH_SIZES <<< "${BATCH_SIZE_GRID:-4096}"
read -r -a FANOUTS <<< "${NUM_NEIGHBORS_GRID:-15,10}"
read -r -a SEEDS <<< "${SEED_GRID:-1}"

TOTAL=$(( ${#MODES[@]} * ${#LABEL_SMOOTHINGS[@]} * ${#DROPOUTS[@]} * ${#WEIGHT_DECAYS[@]} * ${#LRS[@]} * ${#BATCH_SIZES[@]} * ${#FANOUTS[@]} * ${#SEEDS[@]} ))
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID >= TOTAL )); then
  echo "Task id ${TASK_ID} is outside grid size ${TOTAL}; exiting cleanly."
  exit 0
fi

IDX="${TASK_ID}"
SEED="${SEEDS[$(( IDX % ${#SEEDS[@]} ))]}"
IDX=$(( IDX / ${#SEEDS[@]} ))
NUM_NEIGHBORS="${FANOUTS[$(( IDX % ${#FANOUTS[@]} ))]}"
IDX=$(( IDX / ${#FANOUTS[@]} ))
BATCH_SIZE="${BATCH_SIZES[$(( IDX % ${#BATCH_SIZES[@]} ))]}"
IDX=$(( IDX / ${#BATCH_SIZES[@]} ))
LR="${LRS[$(( IDX % ${#LRS[@]} ))]}"
IDX=$(( IDX / ${#LRS[@]} ))
WEIGHT_DECAY="${WEIGHT_DECAYS[$(( IDX % ${#WEIGHT_DECAYS[@]} ))]}"
IDX=$(( IDX / ${#WEIGHT_DECAYS[@]} ))
DROPOUT="${DROPOUTS[$(( IDX % ${#DROPOUTS[@]} ))]}"
IDX=$(( IDX / ${#DROPOUTS[@]} ))
LABEL_SMOOTHING="${LABEL_SMOOTHINGS[$(( IDX % ${#LABEL_SMOOTHINGS[@]} ))]}"
IDX=$(( IDX / ${#LABEL_SMOOTHINGS[@]} ))
MODE="${MODES[$(( IDX % ${#MODES[@]} ))]}"

SWEEP_SAVE_ROOT="${SWEEP_SAVE_ROOT:-outputs/e5_smp_gsmp_sweep}"
RUN_NAME="ls${LABEL_SMOOTHING}_drop${DROPOUT}_wd${WEIGHT_DECAY}_lr${LR}_bs${BATCH_SIZE}_nn${NUM_NEIGHBORS}_seed${SEED}"
RUN_SAVE_DIR="${SWEEP_SAVE_ROOT}/${MODE}/${RUN_NAME}"
mkdir -p "${RUN_SAVE_DIR}"

FEATURE_ARGS=()
if [[ -n "${FEATURES_PATH:-}" ]]; then
  FEATURE_ARGS=(--features_path "${FEATURES_PATH}")
fi

echo "grid_total=${TOTAL}"
echo "mode=${MODE}"
echo "seed=${SEED}"
echo "label_smoothing=${LABEL_SMOOTHING}"
echo "dropout=${DROPOUT}"
echo "weight_decay=${WEIGHT_DECAY}"
echo "lr=${LR}"
echo "batch_size=${BATCH_SIZE}"
echo "num_neighbors=${NUM_NEIGHBORS}"
echo "save_dir=${RUN_SAVE_DIR}"

python -u train_arxiv_sage_temporal.py \
  --mode "${MODE}" \
  "${FEATURE_ARGS[@]}" \
  --seed "${SEED}" \
  --runs 1 \
  --training_mode "${TRAINING_MODE:-mini}" \
  --batch_size "${BATCH_SIZE}" \
  --num_neighbors "${NUM_NEIGHBORS}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --epochs "${EPOCHS:-100}" \
  --patience "${PATIENCE:-50}" \
  --eval_every "${EVAL_EVERY:-10}" \
  --log_every "${LOG_EVERY:-10}" \
  --hidden_channels "${HIDDEN_CHANNELS:-256}" \
  --num_layers "${NUM_LAYERS:-2}" \
  --aggregation_chunk_size "${AGGREGATION_CHUNK_SIZE:-200000}" \
  --dropout "${DROPOUT}" \
  --label_smoothing "${LABEL_SMOOTHING}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --device "${DEVICE:-cuda:0}" \
  --save_dir "${RUN_SAVE_DIR}"
