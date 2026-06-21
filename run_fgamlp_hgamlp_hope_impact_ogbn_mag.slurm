#!/bin/bash
#SBATCH --job-name=fgamlp-hh-mag
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
FGAMLP_DIR=${PROJECT_DIR}/FGAMLP
DATA_ROOT=${PROJECT_DIR}/data
EMB_PATH=${PROJECT_DIR}/data
OUTPUT_ROOT=${PROJECT_DIR}/outputs/fgamlp_hgamlp_hope_impact_ogbn_mag
REL_SUBSETS=${PROJECT_DIR}/sample_relation_subsets/mag/default.txt
VENV_DIR=${PROJECT_DIR}/.venv

EPOCHS=300
EVAL_EVERY=10
SEEDS="1 2 3"
METHODS="baseline smp ump gsmp"

GPU=0
NUM_HOPS=5
LABEL_NUM_HOPS=9
HIDDEN=512
BATCH_SIZE=10000
PATIENCE=300

export PYTHONUNBUFFERED=1

mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/results"

echo "============================================================"
echo "Real FGAMLP/HGAMLP-style ogbn-mag impact experiment"
echo "============================================================"
date
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

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
missing = [pkg for pkg in ("torch", "torch_geometric", "dgl", "ogb", "numpy", "pandas") if importlib.util.find_spec(pkg) is None]
if missing:
    raise SystemExit(f"Missing required Python packages: {missing}")
PY

if [[ ! -f "${EMB_PATH}/TransE_mag/author.pt" || \
      ! -f "${EMB_PATH}/TransE_mag/field_of_study.pt" || \
      ! -f "${EMB_PATH}/TransE_mag/institution.pt" ]]; then
    echo "TransE embeddings missing; preparing them with PyG OGB_MAG(preprocess=transe)."
    python prepare_mag_transe_embeddings.py \
        --pyg-root "${DATA_ROOT}/pyg_ogb_mag_transe" \
        --out-dir "${EMB_PATH}/TransE_mag"
fi

echo "Data root: ${DATA_ROOT}"
echo "Embedding path: ${EMB_PATH}"
echo "Relation subsets: ${REL_SUBSETS}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Methods: ${METHODS}"
echo "Seeds: ${SEEDS}"
echo "Epochs: ${EPOCHS}"
echo "Eval every: ${EVAL_EVERY}"
echo "============================================================"

cd "${FGAMLP_DIR}"

for METHOD in ${METHODS}; do
    for SEED in ${SEEDS}; do
        RUN_NAME=fgamlp_hh_${METHOD}_seed_${SEED}
        RUN_OUTPUT=${OUTPUT_ROOT}/results/${RUN_NAME}
        LOG_FILE=${OUTPUT_ROOT}/logs/${RUN_NAME}.log
        mkdir -p "${RUN_OUTPUT}"

        echo "============================================================" | tee "${LOG_FILE}"
        echo "Run: ${RUN_NAME}" | tee -a "${LOG_FILE}"
        echo "Method: ${METHOD}" | tee -a "${LOG_FILE}"
        echo "Seed: ${SEED}" | tee -a "${LOG_FILE}"
        echo "Start: $(date)" | tee -a "${LOG_FILE}"
        echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
        echo "============================================================" | tee -a "${LOG_FILE}"

        python -u main.py \
            --gpu "${GPU}" \
            --num-runs 1 \
            --seed "${SEED}" \
            --root "${DATA_ROOT}" \
            --dataset ogbn-mag \
            --emb-path "${EMB_PATH}" \
            --use-relation-subsets "${REL_SUBSETS}" \
            --impact-method "${METHOD}" \
            --output-dir "${RUN_OUTPUT}" \
            --method R_GAMLP \
            --stages "${EPOCHS}" \
            --train-num-epochs 0 \
            --eval-every "${EVAL_EVERY}" \
            --input-drop 0 \
            --att-drop 0.5 \
            --label-drop 0.5 \
            --dropout 0.5 \
            --pre-process \
            --act relu \
            --batch-size "${BATCH_SIZE}" \
            --patience "${PATIENCE}" \
            --n-layers-1 4 \
            --n-layers-2 4 \
            --n-layers-3 4 \
            --label-num-hops "${LABEL_NUM_HOPS}" \
            --num-hops "${NUM_HOPS}" \
            --hidden "${HIDDEN}" \
            --bns \
            2>&1 | tee -a "${LOG_FILE}"

        echo "End: $(date)" | tee -a "${LOG_FILE}"
    done
done

echo "============================================================"
echo "All real FGAMLP/HGAMLP-style runs finished at $(date)"
echo "Logs: ${OUTPUT_ROOT}/logs"
echo "Results: ${OUTPUT_ROOT}/results"
echo "============================================================"
