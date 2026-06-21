#!/bin/bash
# Three-epoch mini test for the HH ogbn-mag ablation pipeline.
# Order: hh_gsmp -> hh_smp -> hh_ump -> hh

#SBATCH --job-name=hh_mag_smoke
##SBATCH --account=PAS1289
#SBATCH --output=logs/slurm_smoke_%j.out
#SBATCH --error=logs/slurm_smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

mkdir -p logs mini_runs_3epoch/results mini_runs_3epoch/checkpoints mini_runs_3epoch/precomputed
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

###############################################################################
# Environment setup
###############################################################################

if [ -f /users/PAS1289/jyp531/GSMP-research/.venv/bin/activate ]; then
    source /users/PAS1289/jyp531/GSMP-research/.venv/bin/activate
else
    source ~/.bashrc || true
    if command -v conda >/dev/null 2>&1; then
        CONDA_ENV_NAME=${CONDA_ENV_NAME:-graphenv}
        conda activate "${CONDA_ENV_NAME}" || echo "conda env ${CONDA_ENV_NAME} not found; continuing with current Python"
    fi
fi

###############################################################################
# Debugging info
###############################################################################

echo "hostname: $(hostname)"
echo "date: $(date)"
nvidia-smi || true
which python
python --version
echo "job id: ${SLURM_JOB_ID:-local}"

###############################################################################
# Small experiment settings
###############################################################################

METHODS=("hh_gsmp" "hh_smp" "hh_ump" "hh")
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-3}
EVAL_EVERY=${EVAL_EVERY:-1}
LOG_EVERY=${LOG_EVERY:-1}
HIDDEN_DIM=${HIDDEN_DIM:-64}
DROPOUT=${DROPOUT:-0.5}
LR=${LR:-0.001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
NUM_HOPS=${NUM_HOPS:-1}
BATCH_SIZE=${BATCH_SIZE:-4096}
MAX_TRAIN_NODES=${MAX_TRAIN_NODES:-12000}
MAX_VALID_NODES=${MAX_VALID_NODES:-4000}
MAX_TEST_NODES=${MAX_TEST_NODES:-4000}
DEVICE=${DEVICE:-cuda}
ROOT=${ROOT:-./data}
OUTPUT_DIR=${OUTPUT_DIR:-./mini_runs_3epoch}

for METHOD in "${METHODS[@]}"; do
    METHOD_LOG="logs/${METHOD}_mini3_seed${SEED}.out"
    echo "========== Running 3-epoch mini method: ${METHOD} =========="
    echo "start: $(date)"
    echo "method log: ${METHOD_LOG}"
    python -u train_hh_mag.py \
        --method "${METHOD}" \
        --dataset ogbn-mag \
        --root "${ROOT}" \
        --epochs "${EPOCHS}" \
        --eval-every "${EVAL_EVERY}" \
        --log-every "${LOG_EVERY}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --output-dir "${OUTPUT_DIR}" \
        --hidden-dim "${HIDDEN_DIM}" \
        --dropout "${DROPOUT}" \
        --lr "${LR}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --num-hops "${NUM_HOPS}" \
        --batch-size "${BATCH_SIZE}" \
        --max-train-nodes "${MAX_TRAIN_NODES}" \
        --max-valid-nodes "${MAX_VALID_NODES}" \
        --max-test-nodes "${MAX_TEST_NODES}" \
        --use-precomputed \
        2>&1 | tee "${METHOD_LOG}"
    echo "end: $(date)"
done

echo "========== HH 3-epoch mini experiment complete =========="
cat mini_runs_3epoch/results/summary.csv
