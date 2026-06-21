#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
latest="$(find "${ROOT}/logs" -maxdepth 1 -type f -name '*.out' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"

if [[ -z "${latest}" ]]; then
  echo "No log files found under ${ROOT}/logs" >&2
  exit 1
fi

echo "tail -f ${latest}"
tail -f "${latest}"
