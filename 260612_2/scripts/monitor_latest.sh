#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EXP_ROOT"

echo "[QUEUE]"
squeue -u "${USER:-$(whoami)}" || true
echo

latest_out="$(find logs -maxdepth 1 -type f -name '*.out' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}')"
latest_err="$(find logs -maxdepth 1 -type f -name '*.err' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}')"

if [[ -n "${latest_out:-}" ]]; then
  echo "[LATEST_OUT] $latest_out"
  tail -80 "$latest_out" || true
else
  echo "[LATEST_OUT] none"
fi

echo
if [[ -n "${latest_err:-}" ]]; then
  echo "[LATEST_ERR] $latest_err"
  tail -80 "$latest_err" || true
else
  echo "[LATEST_ERR] none"
fi

echo
job_id=""
if [[ "${latest_out:-}" =~ _([0-9]+)(_([0-9]+))?\.out$ ]]; then
  job_id="${BASH_REMATCH[1]}"
elif [[ "${latest_err:-}" =~ _([0-9]+)(_([0-9]+))?\.err$ ]]; then
  job_id="${BASH_REMATCH[1]}"
fi

if [[ -n "$job_id" ]]; then
  echo "[SACCT] job=$job_id"
  sacct -j "$job_id" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS 2>/dev/null || true
fi

echo
echo "[RESULT_DIRS]"
find results/tunedgcn_pgsmp -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -5 | cut -d' ' -f2-
