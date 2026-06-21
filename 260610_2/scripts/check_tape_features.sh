#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAPE_DIR="${ROOT}/upstream/TAPE"
DATASET="${DATASET:-ogbn-arxiv}"
MODEL_NAME="${LM_MODEL_NAME:-microsoft/deberta-base}"
RUNS="${RUNS:-3}"

missing=0
expected_bytes=$((169343 * 768 * 2))

echo "Checking official TAPE feature files"
echo "TAPE_DIR=${TAPE_DIR}"
echo "DATASET=${DATASET}"
echo "LM_MODEL_NAME=${MODEL_NAME}"
echo "RUNS=${RUNS}"
echo "Expected ogbn-arxiv .emb size: ${expected_bytes} bytes"
echo

for ((seed=0; seed<RUNS; seed++)); do
  for root in "${DATASET}" "${DATASET}2"; do
    path="${TAPE_DIR}/prt_lm/${root}/${MODEL_NAME}-seed${seed}.emb"
    if [[ -s "${path}" ]]; then
      bytes="$(stat -c '%s' "${path}")"
      if [[ "${DATASET}" == "ogbn-arxiv" && "${bytes}" -ne "${expected_bytes}" ]]; then
        echo "WARN size mismatch: ${path} (${bytes} bytes)"
      else
        echo "OK   ${path}"
      fi
    else
      echo "MISS ${path}"
      missing=1
    fi
  done
done

if [[ -s "${TAPE_DIR}/gpt_preds/${DATASET}.csv" ]]; then
  echo "OK   ${TAPE_DIR}/gpt_preds/${DATASET}.csv"
else
  echo "MISS ${TAPE_DIR}/gpt_preds/${DATASET}.csv"
  missing=1
fi

if [[ "${missing}" -ne 0 ]]; then
  echo
  echo "Missing feature files. Download/place them before full TA_P_E runs."
  exit 1
fi

echo
echo "All required TAPE feature files are present."
