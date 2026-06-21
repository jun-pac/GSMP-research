#!/bin/bash
# Local CPU smoke test for the HH ogbn-mag ablation pipeline.
# Order: hh_gsmp -> hh_smp -> hh_ump -> hh
#
# This is intentionally small for login-node debugging. It uses one-hop
# propagation plus limited official splits so SMP/UMP/GSMP are actually
# exercised without turning the check into a full experiment.

set -euo pipefail

mkdir -p logs mini_runs_3epoch/results mini_runs_3epoch/checkpoints mini_runs_3epoch/precomputed
export PYTHONUNBUFFERED=1

if [ -f /users/PAS1289/jyp531/GSMP-research/.venv/bin/activate ]; then
    source /users/PAS1289/jyp531/GSMP-research/.venv/bin/activate
fi

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
DEVICE=${DEVICE:-cpu}
ROOT=${ROOT:-./data}
OUTPUT_DIR=${OUTPUT_DIR:-./mini_runs_3epoch}

echo "hostname: $(hostname)"
echo "date: $(date)"
which python
python --version

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
        --force-recompute \
        2>&1 | tee "${METHOD_LOG}"
    echo "end: $(date)"
done

echo "========== Local HH 3-epoch mini experiment complete =========="
cat mini_runs_3epoch/results/summary.csv
