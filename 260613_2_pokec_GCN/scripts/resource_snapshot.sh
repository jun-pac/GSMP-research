#!/usr/bin/env bash
set -euo pipefail

USER_FILTER="${SLURM_MONITOR_USER:-$(whoami)}"
DAYS="${DAYS:-2}"
START_DATE="$(date -d "${DAYS} days ago" +%F)"

echo "[active jobs]"
squeue -u "${USER_FILTER}" -o "%.18i %.9P %.32j %.8u %.2t %.10M %.10l %.6D %R" || true

echo
echo "[recent accounting since ${START_DATE}]"
sacct -u "${USER_FILTER}" \
  --starttime "${START_DATE}" \
  --format JobID,JobName%28,State,ExitCode,Elapsed,TotalCPU,AllocCPUS,ReqMem,MaxRSS,AllocTRES%80 \
  -P || true

echo
echo "[dashboard monitor snapshot]"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -x "${REPO_ROOT}/slurm_monitor/run_monitor.sh" ]]; then
  (cd "${REPO_ROOT}" && ./slurm_monitor/run_monitor.sh once)
  echo "state: ${REPO_ROOT}/slurm_monitor/state/jobs.json"
else
  echo "slurm_monitor not found at ${REPO_ROOT}/slurm_monitor"
fi
