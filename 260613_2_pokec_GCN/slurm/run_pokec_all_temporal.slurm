#!/bin/bash
#SBATCH --job-name=pokec_all
#SBATCH --account=PAS1289
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p logs results
JOB_ID="${SLURM_JOB_ID:-local}"
LOG_FILE="logs/pokec_all_temporal_${JOB_ID}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "job=${SLURM_JOB_NAME:-pokec_all} id=${JOB_ID}"
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

if [[ ! -f data/pokec_temporal/metadata.json ]]; then
  echo "Missing data/pokec_temporal/metadata.json. Run sbatch slurm/preprocess_pokec_temporal.slurm first."
  exit 1
fi

bash slurm/run_pokec_gcn_temporal.slurm
bash slurm/run_pokec_gsmp_temporal.slurm
python scripts/summarize_results.py || true
date
echo "log=${LOG_FILE}"
