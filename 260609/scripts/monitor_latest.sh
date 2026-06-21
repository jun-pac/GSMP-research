#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

STAGE="${1:-all}"
case "${STAGE}" in
  smoke)
    SEARCH_DIRS=(logs/smoke)
    ;;
  main)
    SEARCH_DIRS=(logs/main)
    ;;
  all)
    SEARCH_DIRS=(logs/smoke logs/main)
    ;;
  *)
    echo "Usage: $0 [smoke|main|all]" >&2
    exit 2
    ;;
esac

latest="$(
  find "${SEARCH_DIRS[@]}" -type f -name '*.out' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
)"

if [[ -z "${latest}" ]]; then
  echo "No log files found in ${SEARCH_DIRS[*]}" >&2
  exit 1
fi

echo "tailing ${latest}"
tail -f "${latest}"
