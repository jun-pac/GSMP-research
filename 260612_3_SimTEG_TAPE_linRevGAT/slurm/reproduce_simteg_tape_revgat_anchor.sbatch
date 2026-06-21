#!/bin/bash
#SBATCH --job-name=revgat-anchor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_DIR}" ]]; then
  PROJECT_DIR="$(pwd)"
fi
cd "${PROJECT_DIR}"
mkdir -p logs

# EDIT THIS SECTION FOR YOUR CLUSTER
set +u
source ~/.bashrc || true
set -u
if command -v conda >/dev/null 2>&1; then
  set +u
  conda activate "${CONDA_ENV:-simteg}" || true
  set -u
fi

hostname
date
pwd
git rev-parse HEAD || true
nvidia-smi || true
python --version || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())" || true
python -c "import torch_geometric; print('pyg', torch_geometric.__version__)" || true
python -c "import dgl; print('dgl', dgl.__version__)" || true

PYTHON_BIN="${PYTHON:-../.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
"${PYTHON_BIN}" -u verify_revgat_anchor.py --seeds "${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
