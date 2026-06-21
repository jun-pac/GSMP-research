#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
SIMTEG_DIR="${SIMTEG_DIR:-${REPO_DIR}/SimTeG}"
PREFIX="${PREFIX:-official_smoke}"
SEEDS="${SEEDS:-1}"
VENV_PY="${PYTHON:-${REPO_DIR}/.venv/bin/python}"

RUN_DIRS=(
  "${SIMTEG_DIR}/out/ogbn-arxiv/revgat/${PREFIX}_arxiv_e5"
  "${SIMTEG_DIR}/out/ogbn-arxiv/revgat/${PREFIX}_arxiv_roberta"
  "${SIMTEG_DIR}/out/ogbn-arxiv-tape/revgat/${PREFIX}_arxiv_tape_e5"
  "${SIMTEG_DIR}/out/ogbn-arxiv-tape/revgat/${PREFIX}_arxiv_tape_roberta"
  "${SIMTEG_DIR}/out/ogbn-arxiv/revgat/${PREFIX}_arxiv_gpt_preds"
)

cd "${PROJECT_DIR}"
"${VENV_PY}" ensemble_logits.py \
  --name "${PREFIX}_official_revgat_ensemble" \
  --seeds "${SEEDS}" \
  --run_dirs "${RUN_DIRS[*]}" \
  --weights "${WEIGHTS:-2 2 1 1 1}"
