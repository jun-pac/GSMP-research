#!/bin/bash
#SBATCH --job-name=mag-shope-pilot
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/pilot_%A_%a.out
#SBATCH --error=logs/pilot_%A_%a.err

set -euo pipefail

export RUN_NAME="${RUN_NAME:-pilot_30epoch}"
export METHODS="${METHODS:-baseline gsmp}"
export SEEDS="${SEEDS:-1 2}"
export STAGES="${STAGES:-30}"
export PATIENCE="${PATIENCE:-10}"

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/run_one_common.sh"
