#!/bin/bash
#SBATCH --job-name=hgamlp-hope-mag
#SBATCH --account=PAS1289
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail

if [[ "${ALLOW_PLACEHOLDER_HGAMLP:-false}" != "true" ]]; then
    echo "ERROR: This script used the placeholder ogbn_mag_temporal model and is disabled." >&2
    echo "Use run_fgamlp_hgamlp_hope_impact_ogbn_mag.slurm for the real FGAMLP/HGAMLP-style pipeline." >&2
    echo "Set ALLOW_PLACEHOLDER_HGAMLP=true only if you intentionally want the old toy run." >&2
    exit 2
fi

###############################################################################
# Edit these settings for your cluster and experiment.
###############################################################################

PROJECT_DIR=/users/PAS1289/jyp531/GSMP-research
DATA_ROOT=/users/PAS1289/jyp531/GSMP-research/data
OUTPUT_ROOT=${PROJECT_DIR}/outputs/ogbn_mag_hgamlp_hope_impact_10000
CONDA_ENV=hgamlp
VENV_DIR=${PROJECT_DIR}/.venv
DATASET=ogbn-mag
EPOCHS=10000
LOG_EVERY=10
EVAL_EVERY=10
HIDDEN_DIM=64
NUM_LAYERS=1
DROPOUT=0.1
PATIENCE=10000
SEEDS="1 2 3"
METHODS="baseline smp ump gsmp"
RESUME=false

TRAIN_SCRIPT=train_hgamlp_hope.py

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

###############################################################################
# Setup and diagnostics.
###############################################################################

mkdir -p "${PROJECT_DIR}/logs"

echo "============================================================"
echo "HGAMLP-HOPE impact experiments on ogbn-mag"
echo "============================================================"
date
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

cd "${PROJECT_DIR}"

if command -v module >/dev/null 2>&1; then
    module load miniconda3/24.1.2-py310 2>/dev/null || true
fi

USING_ENV=none
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda activate "${CONDA_ENV}"
    USING_ENV="conda:${CONDA_ENV}"
elif [[ -f "${VENV_DIR}/bin/activate" ]]; then
    source "${VENV_DIR}/bin/activate"
    USING_ENV="venv:${VENV_DIR}"
else
    echo "ERROR: Could not activate conda env '${CONDA_ENV}' and no venv found at ${VENV_DIR}." >&2
    echo "Edit CONDA_ENV or VENV_DIR in ${BASH_SOURCE[0]}." >&2
    exit 2
fi

echo "Activated environment: ${USING_ENV}"

python --version
if command -v conda >/dev/null 2>&1; then
    conda info --envs
fi
pip freeze | grep -E "torch|dgl|ogb|numpy|scipy" || true

mkdir -p \
    "${OUTPUT_ROOT}/logs" \
    "${OUTPUT_ROOT}/checkpoints" \
    "${OUTPUT_ROOT}/results"

echo "Project dir: ${PROJECT_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Dataset: ${DATASET}"
echo "Methods: ${METHODS}"
echo "Seeds: ${SEEDS}"
echo "Epochs: ${EPOCHS}"
echo "Hidden dim: ${HIDDEN_DIM}"
echo "Num layers: ${NUM_LAYERS}"
echo "Dropout: ${DROPOUT}"
echo "Patience: ${PATIENCE}"
echo "Evaluation every ${EVAL_EVERY} epochs"
echo "Logging every ${LOG_EVERY} epochs"
echo "============================================================"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: ${TRAIN_SCRIPT} was not found in ${PROJECT_DIR}." >&2
    echo "The training script must implement --impact-method with choices: baseline, smp, ump, gsmp." >&2
    exit 2
fi

###############################################################################
# Method notes:
# - baseline: original HGAMLP-HOPE without SMP, UMP, or GSMP.
# - smp: temporally symmetrized weighted propagation.
# - ump: only past/self-time neighbors are used.
# - gsmp: timestamp-balanced inverse-frequency propagation.
#
# The actual training script must implement:
#   --impact-method {baseline,smp,ump,gsmp}
###############################################################################

for METHOD in ${METHODS}; do
    case "${METHOD}" in
        baseline|smp|ump|gsmp) ;;
        *) echo "ERROR: invalid method '${METHOD}'." >&2; exit 2 ;;
    esac

    for SEED in ${SEEDS}; do
        RUN_NAME=hgamlp_hope_${METHOD}_seed_${SEED}
        LOG_FILE=${OUTPUT_ROOT}/logs/${RUN_NAME}.log
        CKPT_DIR=${OUTPUT_ROOT}/checkpoints/${RUN_NAME}
        RESULT_FILE=${OUTPUT_ROOT}/results/${RUN_NAME}.json

        mkdir -p "${CKPT_DIR}" "$(dirname "${LOG_FILE}")" "$(dirname "${RESULT_FILE}")"

        echo "============================================================" | tee "${LOG_FILE}"
        echo "Run name: ${RUN_NAME}" | tee -a "${LOG_FILE}"
        echo "Method: ${METHOD}" | tee -a "${LOG_FILE}"
        echo "Seed: ${SEED}" | tee -a "${LOG_FILE}"
        echo "Start time: $(date)" | tee -a "${LOG_FILE}"
        echo "Log path: ${LOG_FILE}" | tee -a "${LOG_FILE}"
        echo "Checkpoint dir: ${CKPT_DIR}" | tee -a "${LOG_FILE}"
        echo "Result file: ${RESULT_FILE}" | tee -a "${LOG_FILE}"
        echo "============================================================" | tee -a "${LOG_FILE}"

        python -u "${TRAIN_SCRIPT}" \
            --dataset "${DATASET}" \
            --data-root "${DATA_ROOT}" \
            --model hgamlp_hope \
            --impact-method "${METHOD}" \
            --epochs "${EPOCHS}" \
            --eval-every "${EVAL_EVERY}" \
            --log-every "${LOG_EVERY}" \
            --hidden-dim "${HIDDEN_DIM}" \
            --num-layers "${NUM_LAYERS}" \
            --dropout "${DROPOUT}" \
            --patience "${PATIENCE}" \
            --seed "${SEED}" \
            --output-root "${OUTPUT_ROOT}" \
            --checkpoint-dir "${CKPT_DIR}" \
            --result-file "${RESULT_FILE}" \
            --resume "${RESUME}" \
            2>&1 | tee -a "${LOG_FILE}"

        echo "End time: $(date)" | tee -a "${LOG_FILE}"
    done
done

echo "============================================================"
echo "All runs finished at $(date)"
echo "Logs: ${OUTPUT_ROOT}/logs"
echo "Checkpoints: ${OUTPUT_ROOT}/checkpoints"
echo "Result JSON files: ${OUTPUT_ROOT}/results"
echo "============================================================"

if [[ -f summarize_results.py ]]; then
    python summarize_results.py \
        --result-dir "${OUTPUT_ROOT}/results" \
        --out "${OUTPUT_ROOT}/summary.csv"
else
    echo "No summarize_results.py found; skipping summary."
fi
