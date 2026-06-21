#!/bin/bash
#SBATCH --job-name=hgamlp-mag-lb
#SBATCH --account=PAS1289
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail

PROJECT_DIR=/users/PAS1289/jyp531/GSMP-research
CODE_DIR=${PROJECT_DIR}/HGAMLP_MAG
VENV_DIR=${PROJECT_DIR}/.venv
OUTPUT_ROOT=${PROJECT_DIR}/outputs/hgamlp_mag_leaderboard

GPU=0
SEEDS="${SEEDS:-0 1 2}"
STAGES="${STAGES:-400 400 400 500}"
BATCH_SIZE="${BATCH_SIZE:-10000}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/sparse_tools:${PYTHONPATH:-}"

mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_ROOT}"

echo "============================================================"
echo "HGAMLP-MAG leaderboard-style baseline"
echo "============================================================"
date
hostname
echo "Project: ${PROJECT_DIR}"
echo "Code: ${CODE_DIR}"
echo "Output: ${OUTPUT_ROOT}"
echo "Seeds: ${SEEDS}"
echo "Stages: ${STAGES}"
echo "============================================================"

cd "${PROJECT_DIR}"

if command -v module >/dev/null 2>&1; then
    module load miniconda3/24.1.2-py310 2>/dev/null || true
fi

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: venv not found at ${VENV_DIR}" >&2
    exit 2
fi

python --version
python - <<'PY'
import importlib.util
missing = [pkg for pkg in ("torch", "dgl", "ogb", "numpy", "sklearn") if importlib.util.find_spec(pkg) is None]
missing += [pkg for pkg in ("torch_scatter", "torch_sparse", "sparse_tools") if importlib.util.find_spec(pkg) is None]
if missing:
    raise SystemExit(
        "Missing packages for the leaderboard HGAMLP_MAG code: "
        + ", ".join(missing)
        + "\nInstall these before submitting this Slurm. "
        + "The older FGAMLP job does not use the exact leaderboard implementation."
    )
import torch_scatter
from torch_sparse import SparseTensor
import sparse_tools
print("Dependency import check OK")
PY

if [[ ! -f "${CODE_DIR}/mag.p" ]]; then
    cat >&2 <<'EOF'
ERROR: HGAMLP_MAG/mag.p is missing.

The leaderboard reproduction uses LINE embeddings from the HGAMLP_MAG README:
  https://github.com/GooLiang/HGAMLP_MAG

Download mag.p from the README's Google Drive link and place it at:
  /users/PAS1289/jyp531/GSMP-research/HGAMLP_MAG/mag.p
EOF
    exit 3
fi

nvidia-smi || true

cd "${CODE_DIR}"

python -u main.py \
    --stages ${STAGES} \
    --extra-embedding Line \
    --num-hops 2 \
    --label-feats \
    --num-label-hops 2 \
    --n-layers-1 2 \
    --n-layers-2 2 \
    --n-layers-3 3 \
    --act leaky_relu \
    --bns \
    --label-bns \
    --lr 0.002 \
    --weight-decay 0 \
    --threshold 0.75 \
    --patience 100 \
    --gama 10 \
    --amp \
    --seeds ${SEEDS} \
    --gpu "${GPU}" \
    --batch-size "${BATCH_SIZE}" \
    2>&1 | tee "${OUTPUT_ROOT}/hgamlp_mag_leaderboard_${SLURM_JOB_ID}.log"

echo "============================================================"
echo "Finished at $(date)"
echo "Log: ${OUTPUT_ROOT}/hgamlp_mag_leaderboard_${SLURM_JOB_ID}.log"
echo "============================================================"
