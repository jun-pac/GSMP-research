#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SEED="${SEED:-42}"
N_RUNS="${N_RUNS:-1}"
EPOCHS="${EPOCHS:-3}"
PREFIX="${PREFIX:-official_smoke}"
COMPONENT_LIMIT="${COMPONENT_LIMIT:-1}"
INCLUDE_GPT_PREDS="${INCLUDE_GPT_PREDS:-1}"

COMPONENT_DATASETS=(ogbn-arxiv ogbn-arxiv-tape ogbn-arxiv ogbn-arxiv-tape)
COMPONENT_FEATURES=(e5-large e5-large all-roberta-large-v1 all-roberta-large-v1)
COMPONENT_NAMES=(arxiv_e5 arxiv_tape_e5 arxiv_roberta arxiv_tape_roberta)

run_count=0
for idx in "${!COMPONENT_NAMES[@]}"; do
  if (( run_count >= COMPONENT_LIMIT )); then
    break
  fi
  suffix="${PREFIX}_${COMPONENT_NAMES[$idx]}"
  "${SCRIPT_DIR}/run_official_revgat_component.sh" \
    "${COMPONENT_DATASETS[$idx]}" \
    "${COMPONENT_FEATURES[$idx]}" \
    "${SEED}" \
    "${N_RUNS}" \
    "${EPOCHS}" \
    "${suffix}"
  run_count=$((run_count + 1))
done

if [[ "${INCLUDE_GPT_PREDS}" == "1" && "${COMPONENT_LIMIT}" -ge 5 ]]; then
  "${SCRIPT_DIR}/run_official_revgat_component.sh" \
    "gpt-preds" \
    "gpt-preds" \
    "${SEED}" \
    "${N_RUNS}" \
    "${EPOCHS}" \
    "${PREFIX}_arxiv_gpt_preds"
fi
