#!/usr/bin/env bash
#SBATCH --job-name=tape-linear-ablate
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/users/PAS1289/jyp531/GSMP-research/260610_2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

ROOT="${TAPE_GSMP_ROOT:-/users/PAS1289/jyp531/GSMP-research/260610_2}"
cd "${ROOT}"

export RUN_KIND=linear
export MODEL=LinearRevGAT
export USE_GSMP=False
export GSMP_NORM=scale_preserve
if [[ "${FULL:-0}" == "1" ]]; then
  export FEATURE_TYPE="${FEATURE_TYPE:-TA_P_E}"
else
  export FEATURE_TYPE="${FEATURE_TYPE:-ogb}"
fi

bash scripts/run_one_tape_gnn.sh
