#!/bin/bash
#SBATCH --job-name=pokec_preprocess
#SBATCH --account=PAS1289
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail
mkdir -p logs results data
JOB_ID="${SLURM_JOB_ID:-local}"
LOG_FILE="logs/pokec_preprocess_${JOB_ID}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "job=${SLURM_JOB_NAME:-pokec_preprocess} id=${JOB_ID}"
echo "host=$(hostname)"
date
nvidia-smi || true
which python || true
python --version || true

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-base}" || true
  conda info --envs || true
fi

EXTRA_ARGS=()
if [[ -n "${PROCESSED_PATH:-}" ]]; then
  EXTRA_ARGS+=(--processed-path "${PROCESSED_PATH}")
fi
if [[ "${ALLOW_RAW_PROFILE_FEATURES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow-raw-profile-features)
fi

python scripts/preprocess_pokec_temporal.py \
  --snap-dir "${SNAP_DIR:-../data/pokec}" \
  --out-dir data/pokec_temporal \
  "${EXTRA_ARGS[@]}"

date
echo "log=${LOG_FILE}"
