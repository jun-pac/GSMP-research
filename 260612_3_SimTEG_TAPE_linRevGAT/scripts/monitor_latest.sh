#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

JOB_ID="${1:-}"
echo "[MONITOR] cwd=${PROJECT_DIR}"
echo "[MONITOR] time=$(date)"

if [[ -n "${JOB_ID}" ]]; then
  echo "[MONITOR] squeue job=${JOB_ID}"
  squeue -j "${JOB_ID}" -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R" || true
  echo "[MONITOR] sacct job=${JOB_ID}"
  sacct -j "${JOB_ID}" --format=JobID,JobName%30,State,ExitCode,Elapsed,AllocTRES%60 -P 2>/dev/null || true
else
  echo "[MONITOR] squeue user=${USER}"
  squeue -u "${USER}" -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R" || true
fi

echo
echo "[MONITOR] latest log files"
find logs -maxdepth 1 -type f -printf "%T@ %p %s bytes\n" 2>/dev/null | sort -nr | head -12 | cut -d' ' -f2-

latest_out="$(find logs -maxdepth 1 -type f -name '*.out' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
latest_err="$(find logs -maxdepth 1 -type f -name '*.err' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"

if [[ -n "${latest_out}" ]]; then
  echo
  echo "[MONITOR] tail ${latest_out}"
  tail -n 80 "${latest_out}" || true
fi
if [[ -n "${latest_err}" ]]; then
  echo
  echo "[MONITOR] tail ${latest_err}"
  tail -n 80 "${latest_err}" || true
fi

echo
echo "[MONITOR] follow latest stdout:"
echo "tail -f ${latest_out:-logs/<jobname>_<jobid>.out}"

