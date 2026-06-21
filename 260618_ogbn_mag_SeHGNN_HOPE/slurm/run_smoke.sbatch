#!/bin/bash
#SBATCH --job-name=mag-shope-smoke
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-1%1
#SBATCH --output=logs/smoke_%A_%a.out
#SBATCH --error=logs/smoke_%A_%a.err

set -euo pipefail

export RUN_NAME="${RUN_NAME:-smoke}"
export METHODS="${METHODS:-baseline gsmp}"
export SEEDS="${SEEDS:-1}"
export STAGES="${STAGES:-2}"
export PATIENCE="${PATIENCE:-2}"

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/run_one_common.sh"
