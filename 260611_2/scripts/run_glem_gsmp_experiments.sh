#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
shift || true

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_SLURM_ACCOUNT="${DEFAULT_SLURM_ACCOUNT:-PAS1289}"
FULL_DEFAULT_TIME="${FULL_DEFAULT_TIME:-02:00:00}"
FULL_DEFAULT_MEM="${FULL_DEFAULT_MEM:-32G}"
FULL_DEFAULT_MAX_SECONDS="${FULL_DEFAULT_MAX_SECONDS:-7200}"
FULL_DEFAULT_MAX_GPU_MEM_MB="${FULL_DEFAULT_MAX_GPU_MEM_MB:-20000}"
FULL_DEFAULT_MAX_RSS_MB="${FULL_DEFAULT_MAX_RSS_MB:-32000}"

run_static() {
  cd "$WORKDIR"
  "$PYTHON_BIN" tests/test_gsmp_weights.py
  "$PYTHON_BIN" -m py_compile upstream/GLEM/src/models/GNNs/LinearRevGAT/gsmp.py
  "$PYTHON_BIN" -m py_compile upstream/GLEM/src/models/GNNs/LinearRevGAT/model.py
  "$PYTHON_BIN" -m py_compile upstream/GLEM/src/models/GNNs/LinearRevGAT/config.py
  "$PYTHON_BIN" -m py_compile upstream/GLEM/src/models/GNNs/gnn_trainer.py
  "$PYTHON_BIN" -m py_compile upstream/GLEM/src/models/GLEM/config.py
}

preflight_lm_assets() {
  local profile="$1"
  if [[ "$profile" != "full" || "${BUDGET_GUARD_NO_LM_WORK:-T}" != "T" ]]; then
    return 0
  fi

  local lm_dir="$WORKDIR/upstream/GLEM/temp/prt_lm/arxiv_TA/Deberta/FtV1"
  local missing=()
  for asset in Deberta.ckpt Deberta.emb Deberta.pred; do
    if [[ ! -e "$lm_dir/$asset" ]]; then
      missing+=("$lm_dir/$asset")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    printf 'Refusing full GLEM job because cached LM assets are missing and BUDGET_GUARD_NO_LM_WORK=T:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
    printf 'Provide the cached assets or deliberately set BUDGET_GUARD_NO_LM_WORK=F.\n' >&2
    return 3
  fi
}

sbatch_account_args() {
  if [[ -n "${SBATCH_ACCOUNT:-${SLURM_ACCOUNT:-}}" ]]; then
    printf '%s\n' --account="${SBATCH_ACCOUNT:-${SLURM_ACCOUNT:-}}"
  elif [[ -n "$DEFAULT_SLURM_ACCOUNT" ]]; then
    printf '%s\n' --account="$DEFAULT_SLURM_ACCOUNT"
  fi
}

run_sbatch() {
  printf '[SBATCH]'
  for arg in sbatch "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  if [[ "${DRY_RUN_SUBMIT:-0}" == "1" ]]; then
    return 0
  fi
  sbatch "$@"
}

submit_or_run() {
  local method="$1"
  local profile="$2"
  preflight_lm_assets "$profile"
  if [[ "${USE_SLURM:-0}" == "1" ]]; then
    local sbatch_args=()
    if [[ "$profile" == "smoke" ]]; then
      sbatch_args+=(--time="${SBATCH_TIME:-00:15:00}" --mem="${SBATCH_MEM:-24G}")
      export BUDGET_GUARD_MAX_SECONDS="${BUDGET_GUARD_MAX_SECONDS:-600}"
      export BUDGET_GUARD_MAX_GPU_MEM_MB="${BUDGET_GUARD_MAX_GPU_MEM_MB:-14500}"
      export BUDGET_GUARD_MAX_RSS_MB="${BUDGET_GUARD_MAX_RSS_MB:-22000}"
    else
      sbatch_args+=(--time="${SBATCH_TIME:-$FULL_DEFAULT_TIME}" --mem="${SBATCH_MEM:-$FULL_DEFAULT_MEM}")
      export BUDGET_GUARD_MAX_SECONDS="${BUDGET_GUARD_MAX_SECONDS:-$FULL_DEFAULT_MAX_SECONDS}"
      export BUDGET_GUARD_MAX_GPU_MEM_MB="${BUDGET_GUARD_MAX_GPU_MEM_MB:-$FULL_DEFAULT_MAX_GPU_MEM_MB}"
      export BUDGET_GUARD_MAX_RSS_MB="${BUDGET_GUARD_MAX_RSS_MB:-$FULL_DEFAULT_MAX_RSS_MB}"
    fi
    mapfile -t account_args < <(sbatch_account_args)
    sbatch_args+=("${account_args[@]}")
    if [[ -n "${SBATCH_PARTITION:-${SLURM_PARTITION:-}}" ]]; then
      sbatch_args+=(--partition="${SBATCH_PARTITION:-${SLURM_PARTITION:-}}")
    fi
    if [[ -n "${SBATCH_CONSTRAINT:-${SLURM_CONSTRAINT:-}}" ]]; then
      sbatch_args+=(--constraint="${SBATCH_CONSTRAINT:-${SLURM_CONSTRAINT:-}}")
    fi
    case "$method" in
      baseline) PROFILE="$profile" run_sbatch "${sbatch_args[@]}" "$WORKDIR/slurm/reproduce_glem_revgat_baseline.sbatch" ;;
      linear) PROFILE="$profile" run_sbatch "${sbatch_args[@]}" "$WORKDIR/slurm/run_glem_linear_revgat_ablation.sbatch" ;;
      gsmp) PROFILE="$profile" run_sbatch "${sbatch_args[@]}" "$WORKDIR/slurm/run_glem_linear_revgat_gsmp.sbatch" ;;
    esac
  else
    if [[ "$profile" == "full" && "${RUN_LOCAL_FULL:-0}" != "1" ]]; then
      echo "Refusing to run a full $method job locally. Use USE_SLURM=1 or set RUN_LOCAL_FULL=1 deliberately." >&2
      exit 2
    fi
    PROFILE="$profile" bash "$WORKDIR/scripts/run_one_glem_gnn.sh" "$method"
  fi
}

case "$MODE" in
  static)
    run_static
    ;;
  smoke)
    run_static
    PROFILE=smoke SEED="${SEED:-0}" submit_or_run baseline smoke
    PROFILE=smoke SEED="${SEED:-0}" submit_or_run linear smoke
    PROFILE=smoke SEED="${SEED:-0}" submit_or_run gsmp smoke
    ;;
  baseline)
    PROFILE="${PROFILE:-full}" submit_or_run baseline "$PROFILE"
    ;;
  linear)
    PROFILE="${PROFILE:-full}" submit_or_run linear "$PROFILE"
    ;;
  gsmp)
    if [[ "${PROFILE:-full}" == "full" && "${SKIP_BASELINE_GATE:-0}" != "1" ]]; then
      "$PYTHON_BIN" "$WORKDIR/tools/check_baseline_gate.py" \
        --results-root "$WORKDIR/results/glem_revgat_gsmp"
    fi
    PROFILE="${PROFILE:-full}" submit_or_run gsmp "$PROFILE"
    ;;
  all)
    if [[ "${USE_SLURM:-1}" == "1" ]]; then
      sbatch_args=()
      if [[ "${PROFILE:-full}" == "smoke" ]]; then
        sbatch_args+=(--time="${SBATCH_TIME:-00:15:00}" --mem="${SBATCH_MEM:-24G}")
        export BUDGET_GUARD_MAX_SECONDS="${BUDGET_GUARD_MAX_SECONDS:-600}"
        export BUDGET_GUARD_MAX_GPU_MEM_MB="${BUDGET_GUARD_MAX_GPU_MEM_MB:-14500}"
        export BUDGET_GUARD_MAX_RSS_MB="${BUDGET_GUARD_MAX_RSS_MB:-22000}"
      else
        preflight_lm_assets "${PROFILE:-full}"
        sbatch_args+=(--time="${SBATCH_TIME:-$FULL_DEFAULT_TIME}" --mem="${SBATCH_MEM:-$FULL_DEFAULT_MEM}")
        export BUDGET_GUARD_MAX_SECONDS="${BUDGET_GUARD_MAX_SECONDS:-$FULL_DEFAULT_MAX_SECONDS}"
        export BUDGET_GUARD_MAX_GPU_MEM_MB="${BUDGET_GUARD_MAX_GPU_MEM_MB:-$FULL_DEFAULT_MAX_GPU_MEM_MB}"
        export BUDGET_GUARD_MAX_RSS_MB="${BUDGET_GUARD_MAX_RSS_MB:-$FULL_DEFAULT_MAX_RSS_MB}"
      fi
      mapfile -t account_args < <(sbatch_account_args)
      sbatch_args+=("${account_args[@]}")
      if [[ -n "${SBATCH_PARTITION:-${SLURM_PARTITION:-}}" ]]; then
        sbatch_args+=(--partition="${SBATCH_PARTITION:-${SLURM_PARTITION:-}}")
      fi
      if [[ -n "${SBATCH_CONSTRAINT:-${SLURM_CONSTRAINT:-}}" ]]; then
        sbatch_args+=(--constraint="${SBATCH_CONSTRAINT:-${SLURM_CONSTRAINT:-}}")
      fi
      PROFILE="${PROFILE:-full}" run_sbatch "${sbatch_args[@]}" "$WORKDIR/slurm/run_glem_gsmp_all_array.sbatch"
    else
      PROFILE="${PROFILE:-full}" submit_or_run baseline "$PROFILE"
      PROFILE="${PROFILE:-full}" submit_or_run linear "$PROFILE"
      if [[ "${PROFILE:-full}" == "full" && "${SKIP_BASELINE_GATE:-0}" != "1" ]]; then
        "$PYTHON_BIN" "$WORKDIR/tools/check_baseline_gate.py" \
          --results-root "$WORKDIR/results/glem_revgat_gsmp"
      fi
      PROFILE="${PROFILE:-full}" submit_or_run gsmp "$PROFILE"
    fi
    ;;
  *)
    echo "Usage: bash scripts/run_glem_gsmp_experiments.sh {static|smoke|baseline|linear|gsmp|all}" >&2
    exit 2
    ;;
esac
