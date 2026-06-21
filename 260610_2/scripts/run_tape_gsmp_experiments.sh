#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-smoke}"
FULL=0
if [[ "${2:-}" == "--full" ]]; then
  FULL=1
fi

preflight_env() {
  if [[ "${SKIP_PREFLIGHT:-0}" == "1" ]]; then
    return 0
  fi
  local activate=""
  if [[ -n "${VENV_DIR:-}" && -f "${VENV_DIR}/bin/activate" ]]; then
    activate="${VENV_DIR}/bin/activate"
  elif [[ -f "${ROOT}/../.venv/bin/activate" ]]; then
    activate="${ROOT}/../.venv/bin/activate"
  fi
  if [[ -n "${activate}" ]]; then
    # shellcheck disable=SC1090
    source "${activate}"
  fi
  python - <<'PY'
missing = []
for mod in ["yacs", "torch", "dgl", "ogb", "torch_geometric"]:
    try:
        __import__(mod)
    except Exception as exc:
        missing.append(f"{mod}: {exc!r}")
if missing:
    raise SystemExit("TAPE smoke preflight failed before sbatch:\n" + "\n".join(missing))
print("TAPE smoke preflight OK")
PY
}

submit() {
  local script="$1"
  local extra_export="${2:-}"
  local export_vars="ALL,FULL=${FULL},TAPE_GSMP_ROOT=${ROOT}"
  if [[ -n "${extra_export}" ]]; then
    export_vars="${export_vars},${extra_export}"
  fi
  if command -v sbatch >/dev/null 2>&1; then
    sbatch --export="${export_vars}" "${script}"
  else
    echo "sbatch not found; running directly with bash"
    if [[ -n "${extra_export}" ]]; then
      env FULL="${FULL}" "${extra_export}" bash "${script}"
    else
      env FULL="${FULL}" bash "${script}"
    fi
  fi
}

case "${MODE}" in
  smoke)
    FULL=0
    FEATURE_TYPE="${FEATURE_TYPE:-ogb}"
    preflight_env
    submit "${ROOT}/slurm/run_all_tape_gsmp_array.sbatch" "FEATURE_TYPE=${FEATURE_TYPE}"
    ;;
  baseline)
    preflight_env
    submit "${ROOT}/slurm/reproduce_tape_revgat_baseline.sbatch"
    ;;
  linear)
    preflight_env
    submit "${ROOT}/slurm/run_linear_revgat_ablation.sbatch"
    ;;
  gsmp)
    preflight_env
    if [[ "${FULL}" == "1" && "${SKIP_BASELINE_GATE:-0}" != "1" ]]; then
      (cd "${ROOT}/upstream/TAPE" && python "${ROOT}/tools/check_baseline_gate.py" --results-root "${ROOT}/results/tape_revgat_gsmp")
    fi
    submit "${ROOT}/slurm/run_linear_revgat_gsmp.sbatch"
    ;;
  all)
    if [[ "${FULL}" == "1" && "${CONFIRM_FULL:-0}" != "1" ]]; then
      echo "Refusing full all-run without CONFIRM_FULL=1" >&2
      exit 2
    fi
    preflight_env
    submit "${ROOT}/slurm/run_all_tape_gsmp_array.sbatch"
    ;;
  *)
    echo "Usage: $0 {smoke|baseline|linear|gsmp|all} [--full]" >&2
    exit 2
    ;;
esac
