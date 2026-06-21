#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

URL="${GPT_PREDS_URL:-https://raw.githubusercontent.com/vermouthdky/SimTeG/main/src/misc/gpt_preds/ogbn-arxiv.csv}"
OUT="${GPT_PREDS_PATH:-${PROJECT_DIR}/resources/ogbn-arxiv-gpt-preds.csv}"

mkdir -p "$(dirname "${OUT}")"
echo "Downloading GPT prediction labels:"
echo "  ${URL}"
echo "  -> ${OUT}"
curl -L --fail --show-error --silent "${URL}" -o "${OUT}"

rows="$(wc -l < "${OUT}")"
if [[ "${rows}" -ne 169343 ]]; then
  echo "ERROR: expected 169343 rows for ogbn-arxiv, got ${rows}: ${OUT}" >&2
  exit 1
fi
echo "${OUT}"
