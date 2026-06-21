#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/runs/ogbn_arxiv_simteg_tape_revgat_gsmp/logs"

latest="$(ls -t "${LOG_DIR}"/*.log 2>/dev/null | head -1 || true)"
if [[ -z "${latest}" ]]; then
  echo "No logs yet under ${LOG_DIR}" >&2
  exit 1
fi
echo "tail -f ${latest}"
tail -f "${latest}"
