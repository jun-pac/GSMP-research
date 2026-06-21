#!/bin/bash
#SBATCH --job-name=hope-mag-pilot
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-8%1
#SBATCH --output=logs/pilot_%A_%a.out
#SBATCH --error=logs/pilot_%A_%a.err

set -euo pipefail

export METHODS="${METHODS:-none smp gsmp}"
export SEEDS="${SEEDS:-1 2 3}"
export STAGES="${STAGES:-60 60 60 60}"
export EVAL_EVERY="${EVAL_EVERY:-5}"
export BATCH_SIZE="${BATCH_SIZE:-10000}"
export PATIENCE="${PATIENCE:-50}"
export IMPACT_APPLY_TO="${IMPACT_APPLY_TO:-both}"

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/run_one_common.sh"
