#!/bin/bash
#SBATCH --job-name=mag-shope-full
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --array=0-19%1
#SBATCH --output=logs/full_%A_%a.out
#SBATCH --error=logs/full_%A_%a.err

set -euo pipefail

export RUN_NAME="${RUN_NAME:-sehgnn_hope_full}"
export METHODS="${METHODS:-baseline gsmp}"
export SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
export STAGES="${STAGES:-300 300 300 300}"
export PATIENCE="${PATIENCE:-100}"
export EVAL_EVERY="${EVAL_EVERY:-1}"

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/run_one_common.sh"
