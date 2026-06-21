#!/bin/bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "Usage: $0 DATASET_TAG FEATURE_NAME MODEL_VARIANT SEEDS EPOCHS FEATURE_PATH RUN_NAME" >&2
  echo "MODEL_VARIANT: linear or gsmp. FEATURE_PATH may be __gpt_preds__." >&2
  exit 2
fi

DATASET_TAG="$1"
FEATURE_NAME="$2"
MODEL_VARIANT="$3"
SEEDS="$4"
EPOCHS="$5"
FEATURE_PATH="$6"
RUN_NAME="$7"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-runs/ogbn_arxiv_simteg_tape_revgat_gsmp}"
LOG_DIR="${PROJECT_DIR}/${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

VENV_PY="${PYTHON:-${REPO_DIR}/.venv/bin/python}"
if [[ ! -x "${VENV_PY}" ]]; then
  VENV_PY="$(command -v python)"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA_FEATURE_ARGS=()
if [[ "${FEATURE_PATH}" == "__gpt_preds__" ]]; then
  GPT_PREDS_PATH="${GPT_PREDS_PATH:-${PROJECT_DIR}/resources/ogbn-arxiv-gpt-preds.csv}"
  if [[ ! -f "${GPT_PREDS_PATH}" ]]; then
    echo "ERROR: missing GPT prediction CSV: ${GPT_PREDS_PATH}" >&2
    echo "Run: bash scripts/download_gpt_preds.sh" >&2
    exit 1
  fi
  EXTRA_FEATURE_ARGS=(--use_gpt_preds --gpt_preds_path "${GPT_PREDS_PATH}")
else
  if [[ ! -f "${FEATURE_PATH}" ]]; then
    echo "ERROR: missing cached embedding: ${FEATURE_PATH}" >&2
    echo "Run: ALL_COMPONENTS=1 bash scripts/download_embeddings.sh" >&2
    exit 1
  fi
  EXTRA_FEATURE_ARGS=(--use_bert_x --bert_x_dir "${FEATURE_PATH}")
fi

for seed in ${SEEDS}; do
  LOG="${LOG_DIR}/exp_${RUN_NAME}_seed${seed}.log"
  echo "RUN component=${RUN_NAME} variant=${MODEL_VARIANT} seed=${seed} epochs=${EPOCHS}"
  OPTIONAL_ARGS=()
  if [[ "${USE_LABELS:-1}" == "1" ]]; then
    OPTIONAL_ARGS+=(--use-labels)
  fi
  if [[ "${USE_NORM:-0}" == "1" ]]; then
    OPTIONAL_ARGS+=(--use-norm)
  fi
  if [[ "${NO_SELF_LOOPS:-0}" == "1" ]]; then
    OPTIONAL_ARGS+=(--no_self_loops)
  fi
  if [[ "${SAVE_PRED:-1}" == "1" ]]; then
    OPTIONAL_ARGS+=(--save_pred)
  fi
  if [[ "${CPU:-0}" == "1" ]]; then
    OPTIONAL_ARGS+=(--cpu)
  fi
  (
    cd "${PROJECT_DIR}"
    "${VENV_PY}" -u linear_revgat_gsmp.py \
      --experiment_name "${RUN_NAME}" \
      --run_name "${RUN_NAME}" \
      --model_variant "${MODEL_VARIANT}" \
      --dataset ogbn-arxiv \
      --feature_source "${DATASET_TAG}_${FEATURE_NAME}" \
      --seeds "${seed}" \
      --n_epochs "${EPOCHS}" \
      --eval_every "${EVAL_EVERY:-1}" \
      --gpu "${GPU:-0}" \
      --dgl_data_root "${DGL_DATA_ROOT:-../dgl_data}" \
      --output_root "${RUN_ROOT}" \
      --n-layers "${N_LAYERS:-2}" \
      --n-heads "${N_HEADS:-2}" \
      --n-hidden "${N_HIDDEN:-256}" \
      --early_stop_patience "${EARLY_STOP_PATIENCE:-0}" \
      --early_stop_min_epochs "${EARLY_STOP_MIN_EPOCHS:-0}" \
      --dropout "${DROPOUT:-0.58}" \
      --input-drop "${INPUT_DROP:-0.37}" \
      --edge-drop "${EDGE_DROP:-0.0}" \
      --lr "${LR:-0.002}" \
      --wd "${WD:-0}" \
      --label_smoothing_factor "${LABEL_SMOOTHING:-0.02}" \
      --group "${GROUP:-1}" \
      --mask-rate "${MASK_RATE:-0.5}" \
      --n-label-iters "${N_LABEL_ITERS:-0}" \
      --gsmp_norm "${GSMP_NORM:-active_years}" \
      --year_universe "${YEAR_UNIVERSE:-unique}" \
      --edge_direction "${EDGE_DIRECTION:-bidirected}" \
      "${OPTIONAL_ARGS[@]}" \
      "${EXTRA_FEATURE_ARGS[@]}"
  ) 2>&1 | tee "${LOG}"
done
