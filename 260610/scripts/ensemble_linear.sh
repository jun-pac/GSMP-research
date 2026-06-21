#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

PREFIX="${PREFIX:-smoke}"
VARIANT="${VARIANT:-linear}"
SEEDS="${SEEDS:-42}"
RUN_ROOT="${RUN_ROOT:-runs/ogbn_arxiv_simteg_tape_revgat_gsmp}"
VENV_PY="${PYTHON:-${REPO_DIR}/.venv/bin/python}"

RUN_DIRS=(
  "${RUN_ROOT}/components/${PREFIX}_arxiv_e5_${VARIANT}"
  "${RUN_ROOT}/components/${PREFIX}_arxiv_roberta_${VARIANT}"
  "${RUN_ROOT}/components/${PREFIX}_arxiv_tape_e5_${VARIANT}"
  "${RUN_ROOT}/components/${PREFIX}_arxiv_tape_roberta_${VARIANT}"
  "${RUN_ROOT}/components/${PREFIX}_arxiv_gpt_preds_${VARIANT}"
)

cd "${PROJECT_DIR}"
"${VENV_PY}" ensemble_logits.py \
  --name "${PREFIX}_${VARIANT}_ensemble" \
  --seeds "${SEEDS}" \
  --run_dirs "${RUN_DIRS[*]}" \
  --weights "${WEIGHTS:-2 2 1 1 1}"
