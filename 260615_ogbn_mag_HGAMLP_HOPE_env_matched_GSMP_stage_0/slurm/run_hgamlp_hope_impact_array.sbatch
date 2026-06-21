#!/bin/bash
#SBATCH --job-name=hope-mag-impact
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-29%1
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# Default grid:
#   methods = none smp gsmp
#   seeds   = 1..10
#
# Override cheaply, for example:
#   sbatch --array=0-8%1 --export=ALL,SEEDS="1 2 3",STAGES="60 60 60 60",EVAL_EVERY=5 slurm/run_hgamlp_hope_impact_array.sbatch

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/run_one_common.sh"
