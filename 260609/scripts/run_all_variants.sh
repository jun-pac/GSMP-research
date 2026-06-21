#!/bin/bash
set -euo pipefail

DATASET="${1:-${DATASET:-ogbn-arxiv}}"
VARIANTS="${2:-${VARIANTS:-baseline smp ump gsmp}}"
SEEDS="${3:-${SEEDS:-1 2 3}}"
EPOCHS="${4:-${EPOCHS:-100}}"
BERT_X_DIR="${5:-${BERT_X_DIR:-../SimTeG/out/ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt}}"
RUN_NAME="${6:-${RUN_NAME:-main_$(date +%Y%m%d_%H%M%S)}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p "results/${RUN_NAME}"

echo "run_name=${RUN_NAME}"
echo "dataset=${DATASET}"
echo "variants=${VARIANTS}"
echo "seeds=${SEEDS}"
echo "epochs=${EPOCHS}"
echo "bert_x_dir=${BERT_X_DIR}"

for variant in ${VARIANTS}; do
  for seed in ${SEEDS}; do
    echo "RUN variant=${variant} seed=${seed} epochs=${EPOCHS}"
    "${SCRIPT_DIR}/run_one_variant.sh" "${DATASET}" "${variant}" "${seed}" "${EPOCHS}" "${BERT_X_DIR}" "${RUN_NAME}"
  done
done

python summarize_results.py --run_dir "results/${RUN_NAME}"
