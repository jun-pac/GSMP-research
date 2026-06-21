#!/bin/bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 EXPERIMENT_MODE COMPONENT SEED EPOCHS [RUN_ID_PREFIX]" >&2
  echo "COMPONENT: arxiv_e5 arxiv_roberta tape_e5 tape_roberta gpt_preds" >&2
  exit 2
fi

EXPERIMENT_MODE="$1"
COMPONENT="$2"
SEED="$3"
EPOCHS="$4"
RUN_ID_PREFIX="${5:-local}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-${REPO_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

FEATURE_ARGS=()
FEATURE_SOURCE=""
case "${COMPONENT}" in
  arxiv_e5)
    FEATURE_SOURCE="ogbn-arxiv_e5-large"
    FEATURE_PATH="${SIMTEG_ROOT:-${REPO_DIR}/SimTeG}/out/ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt"
    ;;
  arxiv_roberta)
    FEATURE_SOURCE="ogbn-arxiv_all-roberta-large-v1"
    FEATURE_PATH="${SIMTEG_ROOT:-${REPO_DIR}/SimTeG}/out/ogbn-arxiv/all-roberta-large-v1/main/cached_embs/x_embs.pt"
    ;;
  tape_e5)
    FEATURE_SOURCE="ogbn-arxiv-tape_e5-large"
    FEATURE_PATH="${SIMTEG_ROOT:-${REPO_DIR}/SimTeG}/out/ogbn-arxiv-tape/e5-large/main/cached_embs/x_embs.pt"
    ;;
  tape_roberta)
    FEATURE_SOURCE="ogbn-arxiv-tape_all-roberta-large-v1"
    FEATURE_PATH="${SIMTEG_ROOT:-${REPO_DIR}/SimTeG}/out/ogbn-arxiv-tape/all-roberta-large-v1/main/cached_embs/x_embs.pt"
    ;;
  gpt_preds)
    FEATURE_SOURCE="ogbn-arxiv_gpt-preds"
    FEATURE_PATH="${GPT_PREDS_PATH:-${PROJECT_DIR}/../260609/resources/ogbn-arxiv-gpt-preds.csv}"
    ;;
  *)
    echo "Unknown COMPONENT=${COMPONENT}" >&2
    exit 2
    ;;
esac

if [[ "${COMPONENT}" == "gpt_preds" ]]; then
  if [[ "${EXPERIMENT_MODE}" == "pgsmp" || "${EXPERIMENT_MODE}" == "pgsmp_plus_gsmp_first_layer" ]]; then
    echo "[SKIP] P-GSMP is disabled for GPT-pred label features component=${COMPONENT}" >&2
    exit 0
  fi
  if [[ ! -f "${FEATURE_PATH}" ]]; then
    echo "ERROR: missing GPT prediction CSV: ${FEATURE_PATH}" >&2
    exit 1
  fi
  FEATURE_ARGS=(--use-gpt-preds --gpt-preds-path "${FEATURE_PATH}")
else
  if [[ ! -f "${FEATURE_PATH}" ]]; then
    echo "ERROR: missing cached embedding: ${FEATURE_PATH}" >&2
    echo "Run from this folder: ALL_COMPONENTS=1 bash ../260609/scripts/download_embeddings.sh" >&2
    exit 1
  fi
  FEATURE_ARGS=(--use-bert-x --bert-x-dir "${FEATURE_PATH}")
fi

RUN_ID="${RUN_ID_PREFIX}_${EXPERIMENT_MODE}_${COMPONENT}_seed${SEED}"
OPTIONAL_ARGS=()
if [[ "${USE_LABELS:-1}" == "1" ]]; then
  OPTIONAL_ARGS+=(--use-labels)
fi
if [[ "${USE_NORM:-1}" == "0" ]]; then
  OPTIONAL_ARGS+=(--no-use-norm)
fi
if [[ "${SAVE_PRED:-1}" == "1" ]]; then
  OPTIONAL_ARGS+=(--save-pred)
fi
if [[ "${CPU:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--cpu)
fi
if [[ "${NO_SELF_LOOPS:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--no-self-loops)
fi
if [[ "${GSMP_FORCE_RECOMPUTE:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--gsmp-force-recompute)
fi
if [[ "${PGSMP_FORCE_RECOMPUTE:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--pgsmp-force-recompute)
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${PROJECT_DIR}"
echo "[RUN_ONE] mode=${EXPERIMENT_MODE} component=${COMPONENT} seed=${SEED} epochs=${EPOCHS} run_id=${RUN_ID}"
"${PYTHON_BIN}" -u linear_revgat_gsmp_experiment.py \
  --experiment-mode "${EXPERIMENT_MODE}" \
  --run-id "${RUN_ID}" \
  --dataset ogbn-arxiv \
  --feature-source "${FEATURE_SOURCE}" \
  --seeds "${SEED}" \
  --n-epochs "${EPOCHS}" \
  --eval-every "${EVAL_EVERY:-1}" \
  --log-every "${LOG_EVERY:-10}" \
  --gpu "${GPU:-0}" \
  --dgl-data-root "${DGL_DATA_ROOT:-../dgl_data}" \
  --output-root "${OUTPUT_ROOT:-results/simteg_tape_linearrevgat_gsmp}" \
  --cache-root "${CACHE_ROOT:-cache}" \
  --n-layers "${N_LAYERS:-2}" \
  --n-heads "${N_HEADS:-2}" \
  --n-hidden "${N_HIDDEN:-256}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE:-0}" \
  --early-stop-min-epochs "${EARLY_STOP_MIN_EPOCHS:-0}" \
  --dropout "${DROPOUT:-0.58}" \
  --input-drop "${INPUT_DROP:-0.37}" \
  --edge-drop "${EDGE_DROP:-0.0}" \
  --lr "${LR:-0.002}" \
  --wd "${WD:-0}" \
  --label-smoothing-factor "${LABEL_SMOOTHING:-0.02}" \
  --group "${GROUP:-1}" \
  --mask-rate "${MASK_RATE:-0.5}" \
  --n-label-iters "${N_LABEL_ITERS:-2}" \
  --gsmp-layer "${GSMP_LAYER:-0}" \
  --gsmp-norm "${GSMP_NORM:-scale_preserve}" \
  --pgsmp-alpha "${PGSMP_ALPHA:-0.5}" \
  --pgsmp-depth "${PGSMP_DEPTH:-1}" \
  --pgsmp-norm "${PGSMP_NORM:-strict_observed}" \
  --pgsmp-self-mode "${PGSMP_SELF_MODE:-neighbor_only}" \
  --pgsmp-chunk-size "${PGSMP_CHUNK_SIZE:-1000000}" \
  --edge-direction "${EDGE_DIRECTION:-bidirected}" \
  "${OPTIONAL_ARGS[@]}" \
  "${FEATURE_ARGS[@]}"
