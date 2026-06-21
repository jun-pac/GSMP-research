#!/bin/bash
#SBATCH --job-name=hope-mag-budget
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=52G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-6%1
#SBATCH --output=logs/budget_%A_%a.out
#SBATCH --error=logs/budget_%A_%a.err

set -euo pipefail

# Cost-controlled full run:
#   baseline: seed 1 only, because the official leaderboard is the main baseline reference
#   impact methods: seeds 1,2,3
#
# Override JOB_SPECS if needed, e.g.
#   JOB_SPECS="smp:1 smp:2 gsmp:1 gsmp:2" sbatch --array=0-3%1 slurm/run_budget_full.sbatch
export JOB_SPECS="${JOB_SPECS:-none:1 smp:1 smp:2 smp:3 gsmp:1 gsmp:2 gsmp:3}"
export STAGES="${STAGES:-300 300 300 300}"
export EVAL_EVERY="${EVAL_EVERY:-5}"
export BATCH_SIZE="${BATCH_SIZE:-10000}"
export PATIENCE="${PATIENCE:-100}"
export IMPACT_APPLY_TO="${IMPACT_APPLY_TO:-both}"

if [[ -z "${EXTRA_ARGS:-}" ]]; then
    export EXTRA_ARGS="--gc-every 10 --no-empty-cache-every-epoch"
fi

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/run_one_common.sh"
