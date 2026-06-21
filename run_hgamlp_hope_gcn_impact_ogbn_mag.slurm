#!/bin/bash
#SBATCH --job-name=hgamlp-hope-gcn-mag
#SBATCH --account=PAS1289
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

###############################################################################
# Edit these settings for your cluster and experiment.
###############################################################################

DATASET="${DATASET:-ogbn-mag}"
MODEL="${MODEL:-hgamlp_hope_gcn}"
EPOCHS="${EPOCHS:-200}"
LOG_EVERY="${LOG_EVERY:-10}"
EVAL_EVERY="${EVAL_EVERY:-10}"
SEEDS=(${SEEDS:-1 2 3})
METHODS=(${METHODS:-gcn smp ump gsmp})

DATA_ROOT="${DATA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}/hgamlp_hope_gcn_ogbn_mag_runs}"

# Preferred training entry point requested by this experiment.
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_hgamlp_hope_gcn.py}"

# Compatibility fallback for this repository's current ogbn_mag_temporal code.
COMPAT_DIR="${COMPAT_DIR:-ogbn_mag_temporal}"
COMPAT_SCRIPT="${COMPAT_SCRIPT:-main.py}"

# Environment setup. Edit ENV_NAME or replace this block with your cluster modules.
ENV_NAME="${ENV_NAME:-gsmp}"
USE_CONDA="${USE_CONDA:-true}"
CONDA_INIT="${CONDA_INIT:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

RESUME="${RESUME:-false}"
OVERWRITE="${OVERWRITE:-false}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

###############################################################################
# Setup and diagnostics.
###############################################################################

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs checkpoints results "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/results"

echo "============================================================"
echo "HGAMLP-HOPE + GCN temporal message passing on ogbn-mag"
echo "============================================================"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Methods: ${METHODS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Epochs: ${EPOCHS}"
echo "Log every: ${LOG_EVERY}"
echo "Eval every: ${EVAL_EVERY}"
echo "Data root: ${DATA_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

if [[ "${USE_CONDA}" == "true" ]]; then
    if [[ -f "${CONDA_INIT}" ]]; then
        # Edit ENV_NAME above if your conda environment has a different name.
        source "${CONDA_INIT}"
        conda activate "${ENV_NAME}"
    elif [[ -d ".venv" ]]; then
        source .venv/bin/activate
    else
        echo "ERROR: Could not find conda init at ${CONDA_INIT} or local .venv." >&2
        echo "Edit CONDA_INIT/ENV_NAME, set USE_CONDA=false, or activate an environment before submit." >&2
        exit 2
    fi
fi

echo "Python executable: $(command -v python)"
python --version
python - <<'PY'
import importlib.util
for pkg in ("torch", "torch_geometric", "ogb"):
    spec = importlib.util.find_spec(pkg)
    print(f"{pkg}: {'ok' if spec else 'missing'}")
try:
    import torch
    print(f"torch cuda available: {torch.cuda.is_available()}")
    print(f"torch cuda device count: {torch.cuda.device_count()}")
except Exception as exc:
    print(f"torch cuda check failed: {exc}")
PY
nvidia-smi || true

###############################################################################
# Choose the training entry point.
###############################################################################

USE_COMPAT=false
if [[ -f "${TRAIN_SCRIPT}" ]]; then
    echo "Using requested training script: ${TRAIN_SCRIPT}"
elif [[ -f "${COMPAT_DIR}/${COMPAT_SCRIPT}" ]]; then
    echo "WARNING: ${TRAIN_SCRIPT} not found; using compatibility entry point ${COMPAT_DIR}/${COMPAT_SCRIPT}."
    echo "         The compatibility path maps method gcn -> baseline and writes repo-native result JSON/CSV files."
    USE_COMPAT=true
else
    echo "ERROR: Could not find ${TRAIN_SCRIPT} or ${COMPAT_DIR}/${COMPAT_SCRIPT}." >&2
    exit 2
fi

###############################################################################
# Run all experiments sequentially.
###############################################################################

for METHOD in "${METHODS[@]}"; do
    case "${METHOD}" in
        gcn|smp|ump|gsmp) ;;
        *) echo "ERROR: Invalid method '${METHOD}'. Expected exactly: gcn smp ump gsmp." >&2; exit 2 ;;
    esac

    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="${METHOD}_seed_${SEED}"
        SAVE_DIR="${OUTPUT_ROOT}/checkpoints/${METHOD}/seed_${SEED}"
        LOG_FILE="${OUTPUT_ROOT}/logs/${RUN_NAME}.log"
        RESULT_FILE="${OUTPUT_ROOT}/results/${RUN_NAME}.json"
        mkdir -p "${SAVE_DIR}" "$(dirname "${LOG_FILE}")" "$(dirname "${RESULT_FILE}")"

        echo "============================================================" | tee "${LOG_FILE}"
        echo "Method: ${METHOD}" | tee -a "${LOG_FILE}"
        echo "Seed: ${SEED}" | tee -a "${LOG_FILE}"
        echo "Start time: $(date)" | tee -a "${LOG_FILE}"
        echo "Save dir: ${SAVE_DIR}" | tee -a "${LOG_FILE}"
        echo "Result file: ${RESULT_FILE}" | tee -a "${LOG_FILE}"
        echo "============================================================" | tee -a "${LOG_FILE}"

        if [[ "${USE_COMPAT}" == "true" ]]; then
            COMPAT_METHOD="${METHOD}"
            if [[ "${METHOD}" == "gcn" ]]; then
                COMPAT_METHOD="baseline"
            fi

            (
                cd "${COMPAT_DIR}"
                python -u "${COMPAT_SCRIPT}" \
                    --dataset "${DATASET}" \
                    --model hgamLP_hope \
                    --method "${COMPAT_METHOD}" \
                    --epochs "${EPOCHS}" \
                    --eval-steps "${EVAL_EVERY}" \
                    --seed "${SEED}" \
                    --num-runs 1 \
                    --device "${DEVICE}" \
                    --root "${DATA_ROOT}" \
                    --save-dir "${OUTPUT_ROOT}/results/${METHOD}/seed_${SEED}" \
                    ${EXTRA_ARGS}
            ) 2>&1 | tee -a "${LOG_FILE}"
        else
            # The requested interface should print validation/test accuracy at
            # --eval-every/--log-every intervals and should implement the four
            # message-passing modes exactly as: gcn, smp, ump, gsmp.
            CMD=(
                python -u "${TRAIN_SCRIPT}"
                --dataset "${DATASET}"
                --data-root "${DATA_ROOT}"
                --model "${MODEL}"
                --message-passing "${METHOD}"
                --epochs "${EPOCHS}"
                --log-every "${LOG_EVERY}"
                --eval-every "${EVAL_EVERY}"
                --seed "${SEED}"
                --save-dir "${SAVE_DIR}"
                --log-file "${LOG_FILE}"
                --result-file "${RESULT_FILE}"
            )
            if [[ "${RESUME}" == "true" ]]; then
                CMD+=(--resume)
            fi
            if [[ "${OVERWRITE}" == "true" ]]; then
                CMD+=(--overwrite)
            fi

            "${CMD[@]}" ${EXTRA_ARGS} 2>&1 | tee -a "${LOG_FILE}"
        fi

        echo "End time: $(date)" | tee -a "${LOG_FILE}"
    done
done

###############################################################################
# Optional result summary.
###############################################################################

echo "============================================================"
echo "All runs finished at $(date)"
echo "Logs: ${OUTPUT_ROOT}/logs"
echo "Checkpoints: ${OUTPUT_ROOT}/checkpoints"
echo "Results: ${OUTPUT_ROOT}/results"
echo "============================================================"

if [[ -f summarize_results.py ]]; then
    python summarize_results.py --result-dir "${OUTPUT_ROOT}/results" || true
elif [[ -f "${COMPAT_DIR}/summarize_results.py" ]]; then
    python "${COMPAT_DIR}/summarize_results.py" --result-dir "${OUTPUT_ROOT}/results" || true
else
    echo "No summarize_results.py found; inspect per-run logs and result files under ${OUTPUT_ROOT}."
fi
