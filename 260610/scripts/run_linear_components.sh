#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

VARIANTS="${VARIANTS:-linear gsmp}"
SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-3}"
PREFIX="${PREFIX:-smoke}"
COMPONENT_LIMIT="${COMPONENT_LIMIT:-5}"
INCLUDE_GPT_PREDS="${INCLUDE_GPT_PREDS:-1}"

COMPONENT_DATASET_TAGS=(arxiv arxiv_tape arxiv arxiv_tape)
COMPONENT_FEATURES=(e5-large e5-large all-roberta-large-v1 all-roberta-large-v1)
COMPONENT_NAMES=(arxiv_e5 arxiv_tape_e5 arxiv_roberta arxiv_tape_roberta)
COMPONENT_PATHS=(
  "${REPO_DIR}/SimTeG/out/ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt"
  "${REPO_DIR}/SimTeG/out/ogbn-arxiv-tape/e5-large/main/cached_embs/x_embs.pt"
  "${REPO_DIR}/SimTeG/out/ogbn-arxiv/all-roberta-large-v1/main/cached_embs/x_embs.pt"
  "${REPO_DIR}/SimTeG/out/ogbn-arxiv-tape/all-roberta-large-v1/main/cached_embs/x_embs.pt"
)

run_count=0
for idx in "${!COMPONENT_NAMES[@]}"; do
  if (( run_count >= COMPONENT_LIMIT )); then
    break
  fi
  for variant in ${VARIANTS}; do
    run_name="${PREFIX}_${COMPONENT_NAMES[$idx]}_${variant}"
    USE_LABELS="${USE_LABELS:-1}" SAVE_PRED="${SAVE_PRED:-1}" \
      "${SCRIPT_DIR}/run_linear_component.sh" \
        "${COMPONENT_DATASET_TAGS[$idx]}" \
        "${COMPONENT_FEATURES[$idx]}" \
        "${variant}" \
        "${SEEDS}" \
        "${EPOCHS}" \
        "${COMPONENT_PATHS[$idx]}" \
        "${run_name}"
  done
  run_count=$((run_count + 1))
done

if [[ "${INCLUDE_GPT_PREDS}" == "1" && "${COMPONENT_LIMIT}" -ge 5 ]]; then
  for variant in ${VARIANTS}; do
    run_name="${PREFIX}_arxiv_gpt_preds_${variant}"
    USE_LABELS="${USE_LABELS:-1}" SAVE_PRED="${SAVE_PRED:-1}" \
      "${SCRIPT_DIR}/run_linear_component.sh" \
        "arxiv" \
        "gpt-preds" \
        "${variant}" \
        "${SEEDS}" \
        "${EPOCHS}" \
        "__gpt_preds__" \
        "${run_name}"
  done
fi
