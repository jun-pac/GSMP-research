#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-runs/ogbn_arxiv_simteg_tape_revgat_gsmp}"
PREFIX="${PREFIX:-full3_seed123}"
CSV_DIR="${PROJECT_DIR}/${RUN_ROOT}/csv"
LOG_DIR="${PROJECT_DIR}/${RUN_ROOT}/logs"

echo "== Queue =="
squeue -u "${USER}" || true

echo
echo "== Completed seed summaries for PREFIX=${PREFIX} =="
if [[ -f "${CSV_DIR}/per_seed_summary.csv" ]]; then
  awk -F, -v prefix="${PREFIX}" 'NR == 1 || index($2, prefix) == 1 { print }' "${CSV_DIR}/per_seed_summary.csv" \
    | column -s, -t
else
  echo "No per_seed_summary.csv yet."
fi

echo
echo "== Latest matching component log =="
latest="$(ls -t "${LOG_DIR}"/exp_"${PREFIX}"_*.log 2>/dev/null | head -1 || true)"
if [[ -n "${latest}" ]]; then
  echo "${latest}"
  tail -40 "${latest}"
else
  echo "No matching component logs yet under ${LOG_DIR}."
fi
