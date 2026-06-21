#!/usr/bin/env bash
#SBATCH --job-name=tape-gsmp-array
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%1
#SBATCH --chdir=/users/PAS1289/jyp531/GSMP-research/260610_2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

ROOT="${TAPE_GSMP_ROOT:-/users/PAS1289/jyp531/GSMP-research/260610_2}"
cd "${ROOT}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    export RUN_KIND=baseline
    export MODEL=RevGAT
    export USE_GSMP=False
    export GSMP_NORM=scale_preserve
    ;;
  1)
    export RUN_KIND=linear
    export MODEL=LinearRevGAT
    export USE_GSMP=False
    export GSMP_NORM=scale_preserve
    ;;
  2)
    if [[ "${FULL:-0}" == "1" && "${SKIP_BASELINE_GATE:-0}" != "1" ]]; then
      (cd upstream/TAPE && python "${ROOT}/tools/check_baseline_gate.py" --results-root "${ROOT}/results/tape_revgat_gsmp")
    fi
    export RUN_KIND=gsmp
    export MODEL=LinearRevGAT
    export USE_GSMP=True
    export GSMP_NORM="${GSMP_NORM:-scale_preserve}"
    ;;
  *)
    echo "Unknown array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

if [[ "${FULL:-0}" == "1" ]]; then
  export FEATURE_TYPE="${FEATURE_TYPE:-TA_P_E}"
else
  export FEATURE_TYPE="${FEATURE_TYPE:-ogb}"
fi

bash scripts/run_one_tape_gnn.sh
