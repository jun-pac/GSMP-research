#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
latest="$(ls -t "${ROOT}"/logs/*.out 2>/dev/null | head -1 || true)"
if [[ -z "${latest}" ]]; then
  echo "No Slurm .out logs found under ${ROOT}/logs" >&2
  exit 1
fi
echo "tail -f ${latest}"
tail -f "${latest}"
