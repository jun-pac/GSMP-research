#!/usr/bin/env bash
#SBATCH --job-name=ld-linear-gsmp
##SBATCH --account=EDIT_ME
##SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4%1
#SBATCH --chdir=/users/PAS1289/jyp531/GSMP-research/260611
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

ROOT="${LD_GSMP_ROOT:-/users/PAS1289/jyp531/GSMP-research/260611}"
cd "${ROOT}"

# EDIT THIS SECTION FOR YOUR CLUSTER
if [[ "${LD_SOURCE_BASHRC:-0}" == "1" ]]; then
  set +u
  source ~/.bashrc || true
  set -u
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${LD_CONDA_ENV:-ld}" || true
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${FULL:-0}" != "1" && "${TASK_ID}" != "0" ]]; then
  echo "Smoke/default GSMP runs only seed 0; skipping array task ${TASK_ID}."
  exit 0
fi

export RUN_KIND=gsmp
export LINEAR=true
export USE_GSMP=true
export GSMP_NORM="${GSMP_NORM:-scale_preserve}"
export SEED="${SEED:-${TASK_ID}}"

bash scripts/run_one_ld_gnn.sh
