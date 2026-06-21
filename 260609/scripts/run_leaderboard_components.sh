#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

VARIANTS="${VARIANTS:-baseline gsmp}"
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-50}"
PREFIX="${PREFIX:-leaderboard}"
INCLUDE_GPT_PREDS="${INCLUDE_GPT_PREDS:-1}"
GPT_PREDS_PATH="${GPT_PREDS_PATH:-${PROJECT_DIR}/resources/ogbn-arxiv-gpt-preds.csv}"

COMPONENT_NAMES=(
  arxiv_e5
  arxiv_tape_e5
  arxiv_roberta
  arxiv_tape_roberta
)
COMPONENT_DATASETS=(
  ogbn-arxiv
  ogbn-arxiv-tape
  ogbn-arxiv
  ogbn-arxiv-tape
)
COMPONENT_EMBEDDINGS=(
  e5-large
  e5-large
  all-roberta-large-v1
  all-roberta-large-v1
)
COMPONENT_PATHS=(
  "${REPO_DIR}/SimTeG/out/ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt"
  "${REPO_DIR}/SimTeG/out/ogbn-arxiv-tape/e5-large/main/cached_embs/x_embs.pt"
  "${REPO_DIR}/SimTeG/out/ogbn-arxiv/all-roberta-large-v1/main/cached_embs/x_embs.pt"
  "${REPO_DIR}/SimTeG/out/ogbn-arxiv-tape/all-roberta-large-v1/main/cached_embs/x_embs.pt"
)

for idx in "${!COMPONENT_NAMES[@]}"; do
  name="${COMPONENT_NAMES[$idx]}"
  dataset="${COMPONENT_DATASETS[$idx]}"
  embedding="${COMPONENT_EMBEDDINGS[$idx]}"
  bert_x_dir="${COMPONENT_PATHS[$idx]}"
  if [[ ! -f "${bert_x_dir}" ]]; then
    echo "ERROR: missing ${name} embeddings: ${bert_x_dir}" >&2
    echo "Run: ALL_COMPONENTS=1 bash scripts/download_embeddings.sh" >&2
    exit 1
  fi
  for variant in ${VARIANTS}; do
    run_name="${PREFIX}_${name}_${variant}"
    echo "COMPONENT name=${name} dataset=${dataset} embedding=${embedding} variant=${variant} run_name=${run_name}"
    EMBEDDING_NAME="${embedding}" "${SCRIPT_DIR}/run_all_variants.sh" \
      "${dataset}" \
      "${variant}" \
      "${SEEDS}" \
      "${EPOCHS}" \
      "${bert_x_dir}" \
      "${run_name}"
  done
done

if [[ "${INCLUDE_GPT_PREDS}" == "1" ]]; then
  if [[ ! -f "${GPT_PREDS_PATH}" ]]; then
    echo "ERROR: missing GPT prediction CSV: ${GPT_PREDS_PATH}" >&2
    echo "Run: bash scripts/download_gpt_preds.sh" >&2
    exit 1
  fi
  for variant in ${VARIANTS}; do
    run_name="${PREFIX}_arxiv_gpt_preds_${variant}"
    echo "COMPONENT name=arxiv_gpt_preds dataset=ogbn-arxiv embedding=gpt-preds variant=${variant} run_name=${run_name}"
    USE_GPT_PREDS=1 EMBEDDING_NAME="gpt-preds" GPT_PREDS_PATH="${GPT_PREDS_PATH}" \
      "${SCRIPT_DIR}/run_all_variants.sh" \
        "ogbn-arxiv" \
        "${variant}" \
        "${SEEDS}" \
        "${EPOCHS}" \
        "__gpt_preds__" \
        "${run_name}"
  done
fi
