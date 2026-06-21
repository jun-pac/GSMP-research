#!/bin/bash
#SBATCH --job-name=mag-shope-data
#SBATCH --account=PAS1289
#SBATCH --partition=debug-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/check_data_%j.out
#SBATCH --error=logs/check_data_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

if command -v module >/dev/null 2>&1; then
    module load miniconda3/24.1.2-py310 2>/dev/null || true
    module load cuda/11.8.0 2>/dev/null || true
fi
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-hope_official}"
fi

export PYTHONPATH="${PROJECT_DIR}/sparse_tools:${PROJECT_DIR}/HOPE:${PYTHONPATH:-}"
"${PROJECT_DIR}/scripts/prepare_data_links.sh"
python scripts/check_data.py
