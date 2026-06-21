#!/usr/bin/env bash
#SBATCH --job-name=ld-gsmp-all
##SBATCH --account=EDIT_ME
##SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-14%1
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

if [[ "${FULL:-0}" != "1" ]]; then
  if (( TASK_ID > 2 )); then
    echo "Smoke/default all-run uses tasks 0..2 only; skipping task ${TASK_ID}."
    exit 0
  fi
  METHOD_ID="${TASK_ID}"
  SEED_ID=0
else
  METHOD_ID=$(( TASK_ID / 5 ))
  SEED_ID=$(( TASK_ID % 5 ))
fi

case "${METHOD_ID}" in
  0)
    export RUN_KIND=baseline
    export LINEAR=false
    export USE_GSMP=false
    ;;
  1)
    export RUN_KIND=linear
    export LINEAR=true
    export USE_GSMP=false
    ;;
  2)
    if [[ "${FULL:-0}" == "1" && "${SKIP_BASELINE_GATE:-0}" != "1" ]]; then
      python tools/check_baseline_gate.py --results-root results/ld_revgat_gsmp
    fi
    export RUN_KIND=gsmp
    export LINEAR=true
    export USE_GSMP=true
    ;;
  *)
    echo "Unknown method id ${METHOD_ID} from task ${TASK_ID}" >&2
    exit 2
    ;;
esac

export GSMP_NORM="${GSMP_NORM:-scale_preserve}"
export SEED="${SEED:-${SEED_ID}}"

bash scripts/run_one_ld_gnn.sh
