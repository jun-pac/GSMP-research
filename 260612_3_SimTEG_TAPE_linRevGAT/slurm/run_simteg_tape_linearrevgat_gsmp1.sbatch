#!/bin/bash
#SBATCH --job-name=linrevgat-gsmp1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j_%a.out
#SBATCH --error=logs/%x_%j_%a.err

export EXPERIMENT_MODES="gsmp_first_layer"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
exec bash "${PROJECT_DIR}/slurm/run_simteg_tape_linearrevgat_all_array.sbatch"
