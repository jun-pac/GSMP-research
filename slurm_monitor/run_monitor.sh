#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${ROOT}/state"
PID_FILE="${STATE_DIR}/monitor.pid"
LOG_FILE="${STATE_DIR}/monitor.log"

PYTHON_BIN="${PYTHON_BIN:-python3}"
INTERVAL="${SLURM_MONITOR_INTERVAL:-60}"
DAYS="${SLURM_MONITOR_DAYS:-14}"
USER_FILTER="${SLURM_MONITOR_USER:-$(whoami)}"
ACCOUNT_FILTER="${SLURM_MONITOR_ACCOUNT:-}"
CONFIG_FILE="${SLURM_MONITOR_CONFIG:-${ROOT}/config.json}"
STATE_FILE="${SLURM_MONITOR_STATE:-${STATE_DIR}/jobs.json}"
HISTORY_FILE="${SLURM_MONITOR_HISTORY:-${STATE_DIR}/snapshots.jsonl}"

mkdir -p "${STATE_DIR}"

is_running() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}")"
  [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1
}

monitor_args=(
  "${ROOT}/monitor.py"
  --interval "${INTERVAL}"
  --days "${DAYS}"
  --user "${USER_FILTER}"
  --state "${STATE_FILE}"
  --history "${HISTORY_FILE}"
  --config "${CONFIG_FILE}"
)

if [[ -n "${ACCOUNT_FILTER}" ]]; then
  monitor_args+=(--account "${ACCOUNT_FILTER}")
fi

if [[ "${SLURM_MONITOR_ALL_USERS:-0}" == "1" ]]; then
  monitor_args+=(--all-users)
fi

start_detached() {
  if command -v setsid >/dev/null 2>&1; then
    setsid bash -c 'pid_file="$1"; shift; echo "$$" >"${pid_file}"; exec "$@"' \
      _ "${PID_FILE}" "${PYTHON_BIN}" "${monitor_args[@]}" >>"${LOG_FILE}" 2>&1 < /dev/null &
  else
    nohup "${PYTHON_BIN}" "${monitor_args[@]}" >>"${LOG_FILE}" 2>&1 &
    echo "$!" >"${PID_FILE}"
  fi
}

case "${1:-start}" in
  start)
    if is_running; then
      echo "Slurm monitor already running: $(cat "${PID_FILE}")"
      exit 0
    fi
    start_detached
    sleep 0.5
    echo "Started Slurm monitor: $(cat "${PID_FILE}")"
    echo "Log: ${LOG_FILE}"
    ;;
  stop)
    if is_running; then
      kill "$(cat "${PID_FILE}")"
      rm -f "${PID_FILE}"
      echo "Stopped Slurm monitor"
    else
      rm -f "${PID_FILE}"
      echo "Slurm monitor is not running"
    fi
    ;;
  restart)
    "$0" stop
    "$0" start
    ;;
  status)
    if is_running; then
      echo "Slurm monitor running: $(cat "${PID_FILE}")"
      echo "Log: ${LOG_FILE}"
    else
      echo "Slurm monitor is not running"
      exit 1
    fi
    ;;
  once)
    "${PYTHON_BIN}" "${monitor_args[@]}" --once
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|once}" >&2
    exit 2
    ;;
esac
