#!/bin/bash
#SBATCH --job-name=pokec_one
#SBATCH --account=PAS1289
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p logs results
JOB_ID="${SLURM_JOB_ID:-local}"
METHOD="${METHOD:-gcn_gsmp_first}"
LOG_FILE="logs/pokec_${METHOD}_one_${JOB_ID}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "job=${SLURM_JOB_NAME:-pokec_one} id=${JOB_ID} method=${METHOD}"
echo "host=$(hostname)"
date
nvidia-smi || true
which python || true
python --version || true

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-base}" || true
  conda info --envs || true
fi

if [[ ! -f data/pokec_temporal/metadata.json ]]; then
  echo "Missing data/pokec_temporal/metadata.json. Run preprocess first."
  exit 1
fi

SEEDS_ARGS=(${SEEDS:-0})

extra_args=()
if [[ "${METHOD}" == "gcn_gsmp_first" && "${GSMP_PRECOMPUTE_GLOBAL:-0}" == "1" ]]; then
  extra_args+=(--gsmp-precompute-global)
fi
if [[ "${SAVE_CHECKPOINTS:-0}" == "1" ]]; then
  extra_args+=(--save-checkpoints)
fi
if [[ "${RESUME_FROM_CHECKPOINT:-0}" == "1" ]]; then
  extra_args+=(--resume-from-checkpoint)
fi
if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
  extra_args+=(--checkpoint-dir "${CHECKPOINT_DIR}")
fi

python scripts/train_pokec_temporal.py \
  --data-dir data/pokec_temporal \
  --method "${METHOD}" \
  --hidden-channels "${HIDDEN_CHANNELS:-256}" \
  --num-layers "${NUM_LAYERS:-7}" \
  --dropout "${DROPOUT:-0.2}" \
  --in-dropout "${IN_DROPOUT:-0.0}" \
  --lr "${LR:-0.0005}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --batch-size "${BATCH_SIZE:-250000}" \
  --epochs "${EPOCHS:-50}" \
  --bn \
  --res \
  --eval-step "${EVAL_STEP:-9}" \
  --eval-start-epoch "${EVAL_START_EPOCH:-1000}" \
  --min-epochs "${MIN_EPOCHS:-0}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE:-0}" \
  --early-stop-min-delta "${EARLY_STOP_MIN_DELTA:-0.0}" \
  --runs "${RUNS:-1}" \
  --seeds "${SEEDS_ARGS[@]}" \
  --device "${DEVICE:-cuda:0}" \
  --eval-device "${EVAL_DEVICE:-cpu}" \
  "${extra_args[@]}"

python scripts/summarize_results.py || true
date
echo "log=${LOG_FILE}"
