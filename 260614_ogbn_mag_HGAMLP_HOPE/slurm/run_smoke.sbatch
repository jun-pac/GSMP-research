#!/bin/bash
#SBATCH --job-name=hope-mag-smoke
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/smoke_%A_%a.out
#SBATCH --error=logs/smoke_%A_%a.err

set -euo pipefail

export METHODS="${METHODS:-none smp gsmp}"
export SEEDS="${SEEDS:-1}"
export STAGES="${STAGES:-2 2 2 2}"
export EVAL_EVERY="${EVAL_EVERY:-1}"
export BATCH_SIZE="${BATCH_SIZE:-10000}"
export PATIENCE="${PATIENCE:-20}"
export IMPACT_APPLY_TO="${IMPACT_APPLY_TO:-both}"

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/run_one_common.sh"
