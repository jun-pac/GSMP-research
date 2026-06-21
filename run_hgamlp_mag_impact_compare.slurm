#!/bin/bash
#SBATCH --job-name=hgamlp-mag-impact
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
OUTPUT_ROOT=${PROJECT_DIR}/outputs/hgamlp_mag_impact_compare

GPU=0
SEEDS="${SEEDS:-0 1 2}"
METHODS="${METHODS:-baseline smp ump gsmp}"
STAGES="${STAGES:-400 400 400 500}"
BATCH_SIZE="${BATCH_SIZE:-10000}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/sparse_tools:${PYTHONPATH:-}"

mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_ROOT}/logs"

echo "============================================================"
echo "HGAMLP-MAG impact comparison: HH vs HH+SMP/UMP/GSMP"
echo "============================================================"
date
hostname
echo "Project: ${PROJECT_DIR}"
echo "Code: ${CODE_DIR}"
echo "Output: ${OUTPUT_ROOT}"
echo "Methods: ${METHODS}"
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
import torch_scatter
from torch_sparse import SparseTensor
import sparse_tools
print("Dependency import check OK")
PY

if [[ ! -f "${CODE_DIR}/mag.p" ]]; then
    echo "ERROR: ${CODE_DIR}/mag.p is missing." >&2
    exit 3
fi

nvidia-smi || true

cd "${CODE_DIR}"

for METHOD in ${METHODS}; do
    case "${METHOD}" in
        baseline) LABEL="HH" ;;
        smp) LABEL="HH+SMP" ;;
        ump) LABEL="HH+UMP" ;;
        gsmp) LABEL="HH+GSMP" ;;
        *) echo "Unknown METHOD=${METHOD}" >&2; exit 4 ;;
    esac

    RUN_LOG="${OUTPUT_ROOT}/logs/${METHOD}_${SLURM_JOB_ID}.log"
    echo "============================================================" | tee "${RUN_LOG}"
    echo "Run method: ${LABEL} (${METHOD})" | tee -a "${RUN_LOG}"
    echo "Start: $(date)" | tee -a "${RUN_LOG}"
    echo "Seeds: ${SEEDS}" | tee -a "${RUN_LOG}"
    echo "Stages: ${STAGES}" | tee -a "${RUN_LOG}"
    echo "============================================================" | tee -a "${RUN_LOG}"

    python -u main.py \
        --impact-method "${METHOD}" \
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
        2>&1 | tee -a "${RUN_LOG}"

    echo "End: $(date)" | tee -a "${RUN_LOG}"
done

echo "============================================================"
echo "All impact comparison runs finished at $(date)"
echo "Logs: ${OUTPUT_ROOT}/logs"
echo "============================================================"
