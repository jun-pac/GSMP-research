#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

is_placeholder_path() {
  [[ "$1" == /path/to/* || "$1" == PATH_TO_* ]]
}

if [[ -z "${LD_LM_PATH:-}" ]]; then
  LD_LM_PATH="${HOME}/model/deberta-base"
elif is_placeholder_path "${LD_LM_PATH}"; then
  LD_LM_PATH="${HOME}/model/deberta-base"
fi
DEFAULT_LOCAL_TOKEN_FOLDER="${ROOT}/downloads/ld_tokens/extracted/ogbn_arxiv/token/microsoft/deberta-base"
if [[ -n "${LD_TOKEN_FOLDER:-}" ]] && is_placeholder_path "${LD_TOKEN_FOLDER}"; then
  unset LD_TOKEN_FOLDER
fi
if [[ -z "${LD_TOKEN_FOLDER:-}" && -f "${DEFAULT_LOCAL_TOKEN_FOLDER}/input_ids.npy" ]]; then
  LD_TOKEN_FOLDER="${DEFAULT_LOCAL_TOKEN_FOLDER}"
else
  LD_TOKEN_FOLDER="${LD_TOKEN_FOLDER:-/OGB/ogbn_arxiv/token/microsoft/deberta-base}"
fi
DATASET_NAME="${DATASET_NAME:-arxiv}"
SEEDS="${SEEDS:-0 1 2 3 4}"

echo "Checking LD ogbn-arxiv assets"
echo "ROOT=${ROOT}"
echo "LD_LM_PATH=${LD_LM_PATH}"
echo "LD_TOKEN_FOLDER=${LD_TOKEN_FOLDER}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "SEEDS=${SEEDS}"
echo

missing=0
for token_file in input_ids.npy attention_mask.npy token_type_ids.npy; do
  path="${LD_TOKEN_FOLDER}/${token_file}"
  if [[ -f "${path}" ]]; then
    echo "OK   ${path}"
  else
    echo "MISS ${path}"
    missing=1
  fi
done

echo
for seed in ${SEEDS}; do
  path="${LD_LM_PATH}/${DATASET_NAME}_seed${seed}/hidden_state.pt"
  if [[ -f "${path}" ]]; then
    echo "OK   ${path}"
  else
    echo "MISS ${path}"
    missing=1
  fi
done

echo
if [[ "${missing}" == "0" ]]; then
  echo "All expected token files and seed hidden_state.pt files are present."
else
  echo "Some assets are missing."
  echo "Smoke/full runs may regenerate missing hidden_state.pt with the LM, which costs extra GPU time."
  echo "Set LD_REQUIRE_HIDDEN_STATE=1 in your own workflow if you want to refuse that."
fi
