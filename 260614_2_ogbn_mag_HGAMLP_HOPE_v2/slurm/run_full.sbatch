#!/bin/bash
#SBATCH --job-name=hope-mag-full
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --array=0-29%1
#SBATCH --output=logs/full_%A_%a.out
#SBATCH --error=logs/full_%A_%a.err

set -euo pipefail

# Full mode keeps official HOPE leaderboard hyperparameters unless overridden
# intentionally by the submitter.
export METHODS="${METHODS:-none smp gsmp}"
export SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
export STAGES="${STAGES:-300 300 300 300}"
export EVAL_EVERY="${EVAL_EVERY:-1}"
export BATCH_SIZE="${BATCH_SIZE:-10000}"
export PATIENCE="${PATIENCE:-100}"
export IMPACT_APPLY_TO="${IMPACT_APPLY_TO:-both}"

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/run_one_common.sh"
