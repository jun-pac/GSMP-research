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
  else
    source ~/.bashrc || true
    if command -v conda >/dev/null 2>&1; then
      conda activate "${LD_CONDA_ENV:-ld}" || true
    fi
  fi
  python - <<'PY'
missing = []
for mod in ["torch", "dgl", "ogb", "transformers", "hydra", "omegaconf"]:
    try:
        __import__(mod)
    except Exception as exc:
        missing.append(f"{mod}: {exc!r}")
if missing:
    raise SystemExit("LD smoke preflight failed before sbatch:\n" + "\n".join(missing))
print("LD smoke preflight OK")
PY
}

submit() {
  local script="$1"
  local extra_export="${2:-}"
  local extra_sbatch="${3:-}"
  local export_vars="ALL,FULL=${FULL},LD_GSMP_ROOT=${ROOT}"
  if [[ -n "${extra_export}" ]]; then
    export_vars="${export_vars},${extra_export}"
  fi
  if command -v sbatch >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    sbatch \
      --account="${SLURM_ACCOUNT:-PAS1289}" \
      --partition="${SLURM_PARTITION:-gpu}" \
      --export="${export_vars}" \
      ${extra_sbatch} \
      "${script}"
  else
    echo "sbatch not found; running directly with bash"
    env FULL="${FULL}" ${extra_export:+${extra_export}} bash "${script}"
  fi
}

case "${MODE}" in
  smoke)
    FULL=0
    preflight_env
    submit "${ROOT}/slurm/run_ld_gsmp_all_array.sbatch" "SMOKE=1" "--array=0-2%1"
    ;;
  baseline)
    preflight_env
    if [[ "${FULL}" == "1" ]]; then
      submit "${ROOT}/slurm/reproduce_ld_revgat_baseline.sbatch"
    else
      submit "${ROOT}/slurm/reproduce_ld_revgat_baseline.sbatch" "" "--array=0"
    fi
    ;;
  linear)
    preflight_env
    if [[ "${FULL}" == "1" ]]; then
      submit "${ROOT}/slurm/run_ld_linear_revgat_ablation.sbatch"
    else
      submit "${ROOT}/slurm/run_ld_linear_revgat_ablation.sbatch" "" "--array=0"
    fi
    ;;
  gsmp)
    preflight_env
    if [[ "${FULL}" == "1" && "${SKIP_BASELINE_GATE:-0}" != "1" ]]; then
      python "${ROOT}/tools/check_baseline_gate.py" --results-root "${ROOT}/results/ld_revgat_gsmp"
    fi
    if [[ "${FULL}" == "1" ]]; then
      submit "${ROOT}/slurm/run_ld_linear_revgat_gsmp.sbatch"
    else
      submit "${ROOT}/slurm/run_ld_linear_revgat_gsmp.sbatch" "" "--array=0"
    fi
    ;;
  all)
    if [[ "${FULL}" == "1" && "${CONFIRM_FULL:-0}" != "1" ]]; then
      echo "Refusing full all-run without CONFIRM_FULL=1" >&2
      exit 2
    fi
    preflight_env
    submit "${ROOT}/slurm/run_ld_gsmp_all_array.sbatch"
    ;;
  *)
    echo "Usage: $0 {smoke|baseline|linear|gsmp|all} [--full]" >&2
    exit 2
    ;;
esac
