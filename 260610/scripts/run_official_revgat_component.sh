#!/bin/bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 DATASET_TAG FEATURE_NAME SEED N_RUNS N_EPOCHS OUTPUT_SUFFIX" >&2
  echo "DATASET_TAG is ogbn-arxiv, ogbn-arxiv-tape, or gpt-preds." >&2
  exit 2
fi

DATASET_TAG="$1"
FEATURE_NAME="$2"
SEED="$3"
N_RUNS="$4"
N_EPOCHS="$5"
OUTPUT_SUFFIX="$6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
SIMTEG_DIR="${SIMTEG_DIR:-${REPO_DIR}/SimTeG}"
VENV_PY="${PYTHON:-${REPO_DIR}/.venv/bin/python}"
LOG_DIR="${PROJECT_DIR}/runs/ogbn_arxiv_simteg_tape_revgat_gsmp/logs"
mkdir -p "${LOG_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_DIR="${SIMTEG_DIR}/out/${DATASET_TAG}/revgat/${OUTPUT_SUFFIX}"
CKPT_DIR="${OUTPUT_DIR}/ckpt"
mkdir -p "${CKPT_DIR}"

FEATURE_ARGS=()
if [[ "${DATASET_TAG}" == "gpt-preds" || "${FEATURE_NAME}" == "gpt-preds" ]]; then
  GPT_PREDS_PATH="${GPT_PREDS_PATH:-${PROJECT_DIR}/resources/ogbn-arxiv-gpt-preds.csv}"
  if [[ ! -f "${GPT_PREDS_PATH}" ]]; then
    echo "ERROR: missing GPT prediction CSV: ${GPT_PREDS_PATH}" >&2
    echo "Run: bash ${PROJECT_DIR}/scripts/download_gpt_preds.sh" >&2
    exit 1
  fi
  mkdir -p "${SIMTEG_DIR}/src/misc/gpt_preds"
  ln -sf "${GPT_PREDS_PATH}" "${SIMTEG_DIR}/src/misc/gpt_preds/ogbn-arxiv.csv"
  FEATURE_ARGS=(--use_gpt_preds)
  OUTPUT_DIR="${SIMTEG_DIR}/out/ogbn-arxiv/revgat/${OUTPUT_SUFFIX}"
  CKPT_DIR="${OUTPUT_DIR}/ckpt"
  mkdir -p "${CKPT_DIR}"
else
  BERT_X_DIR="${SIMTEG_DIR}/out/${DATASET_TAG}/${FEATURE_NAME}/main/cached_embs/x_embs.pt"
  if [[ ! -f "${BERT_X_DIR}" ]]; then
    echo "ERROR: missing cached embedding: ${BERT_X_DIR}" >&2
    echo "Run: ALL_COMPONENTS=1 bash ${PROJECT_DIR}/scripts/download_embeddings.sh" >&2
    exit 1
  fi
  FEATURE_ARGS=(--use_bert_x --bert_x_dir "${BERT_X_DIR}")
fi

LOG="${LOG_DIR}/official_revgat_${OUTPUT_SUFFIX}_seed${SEED}.log"
(
  cd "${SIMTEG_DIR}"
  "${VENV_PY}" -u -m src.misc.revgat.main \
    --use-norm \
    --no-attn-dst \
    --mode teacher \
    --gpu "${GPU:-0}" \
    --dropout "${DROPOUT:-0.58}" \
    --edge-drop "${EDGE_DROP:-0.46}" \
    --group "${GROUP:-1}" \
    --input-drop "${INPUT_DROP:-0.37}" \
    --label_smoothing_factor "${LABEL_SMOOTHING:-0.02}" \
    --n-heads "${N_HEADS:-2}" \
    --n-hidden "${N_HIDDEN:-256}" \
    --n-label-iters "${N_LABEL_ITERS:-2}" \
    --n-layers "${N_LAYERS:-2}" \
    --use-labels \
    --suffix "${OUTPUT_SUFFIX}" \
    --ckpt_dir "${CKPT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --save_pred \
    --seed "${SEED}" \
    --n-runs "${N_RUNS}" \
    --n-epochs "${N_EPOCHS}" \
    "${FEATURE_ARGS[@]}"
) 2>&1 | tee "${LOG}"
