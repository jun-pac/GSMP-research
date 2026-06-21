#!/bin/bash
# Optional and more expensive. Submit only after the single-seed run looks good:
#   cd /users/PAS1289/jyp531/GSMP-research/ogbn_arxiv_temporal
#   mkdir -p logs results checkpoints
#   sbatch slurm/run_arxiv_sage_multiseed_optional.sh
#
# Monitor with:
#   tail -f logs/<job-name>_<job-id>_<array-id>.out

#SBATCH --job-name=arxiv_sage_multiseed
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-11
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
mkdir -p logs results checkpoints
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

MODES=(baseline smp ump gsmp)
SEEDS=(1 2 3)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODE_INDEX=$(( TASK_ID / ${#SEEDS[@]} ))
SEED_INDEX=$(( TASK_ID % ${#SEEDS[@]} ))
MODE="${MODES[${MODE_INDEX}]}"
SEED="${SEEDS[${SEED_INDEX}]}"

FEATURE_ARGS=()
if [[ -n "${FEATURES_PATH:-}" ]]; then
  FEATURE_ARGS=(--features_path "${FEATURES_PATH}")
fi

echo "Running optional multi-seed mode=${MODE} seed=${SEED}"
python -u train_arxiv_sage_temporal.py \
  --mode "${MODE}" \
  "${FEATURE_ARGS[@]}" \
  --seed "${SEED}" \
  --runs 1 \
  --training_mode "${TRAINING_MODE:-full}" \
  --batch_size "${BATCH_SIZE:-8192}" \
  --num_neighbors "${NUM_NEIGHBORS:-15,10,5}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --epochs "${EPOCHS:-300}" \
  --patience "${PATIENCE:-50}" \
  --eval_every "${EVAL_EVERY:-10}" \
  --log_every "${LOG_EVERY:-10}" \
  --hidden_channels "${HIDDEN_CHANNELS:-256}" \
  --num_layers "${NUM_LAYERS:-3}" \
  --aggregation_chunk_size "${AGGREGATION_CHUNK_SIZE:-200000}" \
  --dropout "${DROPOUT:-0.4}" \
  --label_smoothing "${LABEL_SMOOTHING:-0.0}" \
  --lr "${LR:-0.01}" \
  --weight_decay "${WEIGHT_DECAY:-2e-6}" \
  --device "${DEVICE:-cuda:0}" \
  --save_dir "${SAVE_DIR:-.}"
