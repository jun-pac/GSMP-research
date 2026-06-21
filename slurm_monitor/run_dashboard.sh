#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${ROOT}/state"
PID_FILE="${STATE_DIR}/dashboard.pid"
LOG_FILE="${STATE_DIR}/dashboard.log"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${SLURM_DASHBOARD_HOST:-127.0.0.1}"
PORT="${SLURM_DASHBOARD_PORT:-8765}"
STATE_FILE="${SLURM_MONITOR_STATE:-${STATE_DIR}/jobs.json}"
HISTORY_FILE="${SLURM_MONITOR_HISTORY:-${STATE_DIR}/snapshots.jsonl}"

mkdir -p "${STATE_DIR}"

is_running() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}")"
  [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1
}

dashboard_args=(
  "${ROOT}/dashboard.py"
  --host "${HOST}"
  --port "${PORT}"
  --state "${STATE_FILE}"
  --history "${HISTORY_FILE}"
)

start_detached() {
  if command -v setsid >/dev/null 2>&1; then
    setsid bash -c 'pid_file="$1"; shift; echo "$$" >"${pid_file}"; exec "$@"' \
      _ "${PID_FILE}" "${PYTHON_BIN}" "${dashboard_args[@]}" >>"${LOG_FILE}" 2>&1 < /dev/null &
  else
    nohup "${PYTHON_BIN}" "${dashboard_args[@]}" >>"${LOG_FILE}" 2>&1 &
    echo "$!" >"${PID_FILE}"
  fi
}

case "${1:-start}" in
  start)
    if is_running; then
      echo "Slurm dashboard already running: $(cat "${PID_FILE}")"
      exit 0
    fi
    start_detached
    sleep 0.5
    echo "Started Slurm dashboard: $(cat "${PID_FILE}")"
    echo "URL: http://${HOST}:${PORT}/"
    echo "Log: ${LOG_FILE}"
    ;;
  stop)
    if is_running; then
      kill "$(cat "${PID_FILE}")"
      rm -f "${PID_FILE}"
      echo "Stopped Slurm dashboard"
    else
      rm -f "${PID_FILE}"
      echo "Slurm dashboard is not running"
    fi
    ;;
  restart)
    "$0" stop
    "$0" start
    ;;
  status)
    if is_running; then
      echo "Slurm dashboard running: $(cat "${PID_FILE}")"
      echo "URL: http://${HOST}:${PORT}/"
      echo "Log: ${LOG_FILE}"
    else
      echo "Slurm dashboard is not running"
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}" >&2
    exit 2
    ;;
esac
