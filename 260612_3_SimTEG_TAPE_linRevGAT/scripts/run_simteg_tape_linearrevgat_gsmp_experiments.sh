#!/bin/bash
set -euo pipefail

ACTION="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p logs

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh smoke
  bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh anchor
  bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh baseline
  bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh gsmp1
  bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh pgsmp
  bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh all

Useful overrides:
  SEEDS="42 43 44" EPOCHS=200 COMPONENTS="arxiv_e5 arxiv_roberta tape_e5 tape_roberta"
  INCLUDE_GPT_PREDS=1 for baseline/gsmp1 only
  ACCOUNT=PAS1289 PARTITION=... QOS=... for cluster-specific sbatch options
  LOCAL=1 to run sequentially without sbatch
USAGE
}

infer_account_from_path() {
  local path="$1"
  if [[ "${path}" == /users/* ]]; then
    local rest="${path#/users/}"
    echo "${rest%%/*}"
  fi
}

make_sbatch_args() {
  local inferred_account
  inferred_account="$(infer_account_from_path "${PROJECT_DIR}")"
  local account="${ACCOUNT:-${SLURM_ACCOUNT:-${inferred_account}}}"
  local args=()
  if [[ -n "${account}" ]]; then
    args+=(--account="${account}")
  fi
  if [[ -n "${PARTITION:-}" ]]; then
    args+=(--partition="${PARTITION}")
  fi
  if [[ -n "${QOS:-}" ]]; then
    args+=(--qos="${QOS}")
  fi
  if [[ -n "${SBATCH_TIME:-}" ]]; then
    args+=(--time="${SBATCH_TIME}")
  fi
  if [[ -n "${SBATCH_CPUS:-}" ]]; then
    args+=(--cpus-per-task="${SBATCH_CPUS}")
  fi
  if [[ -n "${SBATCH_MEM:-}" ]]; then
    args+=(--mem="${SBATCH_MEM}")
  fi
  if [[ -n "${SBATCH_GRES:-}" ]]; then
    args+=(--gres="${SBATCH_GRES}")
  fi
  printf '%s\n' "${args[@]}"
}

submit_sbatch() {
  local sbatch_args=()
  mapfile -t sbatch_args < <(make_sbatch_args)
  if ((${#sbatch_args[@]})); then
    echo "[SBATCH] extra_args=${sbatch_args[*]}"
  fi
  sbatch "${sbatch_args[@]}" "$@"
}

submit_array() {
  local modes="$1"
  local components="$2"
  local seeds="$3"
  local epochs="$4"
  local include_gpt="${5:-0}"
  local job_script="${6:-slurm/run_simteg_tape_linearrevgat_all_array.sbatch}"

  read -r -a mode_arr <<< "${modes}"
  read -r -a comp_arr <<< "${components}"
  read -r -a seed_arr <<< "${seeds}"
  local comp_count="${#comp_arr[@]}"
  if [[ "${include_gpt}" == "1" ]]; then
    comp_count=$((comp_count + 1))
  fi
  local total=$(( ${#mode_arr[@]} * comp_count * ${#seed_arr[@]} ))
  if (( total <= 0 )); then
    echo "Nothing to submit." >&2
    exit 1
  fi
  local last=$((total - 1))
  local run_prefix="${RUN_ID_PREFIX:-$(date +%Y%m%d_%H%M%S)}"
  echo "[LAUNCH] modes=${modes}"
  echo "[LAUNCH] components=${components} include_gpt=${include_gpt}"
  echo "[LAUNCH] seeds=${seeds} epochs=${epochs} tasks=${total}"
  if [[ "${LOCAL:-0}" == "1" ]]; then
    local task mode_idx rem comp_idx seed_idx mode component seed
    for ((task=0; task<total; task++)); do
      mode_idx=$(( task / (comp_count * ${#seed_arr[@]}) ))
      rem=$(( task % (comp_count * ${#seed_arr[@]}) ))
      comp_idx=$(( rem / ${#seed_arr[@]} ))
      seed_idx=$(( rem % ${#seed_arr[@]} ))
      mode="${mode_arr[$mode_idx]}"
      if (( comp_idx < ${#comp_arr[@]} )); then
        component="${comp_arr[$comp_idx]}"
      else
        component="gpt_preds"
      fi
      seed="${seed_arr[$seed_idx]}"
      bash scripts/run_one_linearrevgat_component.sh "${mode}" "${component}" "${seed}" "${epochs}" "${run_prefix}"
    done
  else
    PROJECT_DIR="${PROJECT_DIR}" EXPERIMENT_MODES="${modes}" COMPONENTS="${components}" SEEDS="${seeds}" EPOCHS="${epochs}" \
      INCLUDE_GPT_PREDS="${include_gpt}" RUN_ID_PREFIX="${run_prefix}" \
      submit_sbatch --array=0-"${last}" "${job_script}"
  fi
}

case "${ACTION}" in
  smoke)
    submit_array \
      "${EXPERIMENT_MODES:-baseline gsmp_first_layer pgsmp}" \
      "${COMPONENTS:-arxiv_e5}" \
      "${SEEDS:-42}" \
      "${EPOCHS:-3}" \
      "0"
    ;;
  anchor)
    if [[ "${LOCAL:-0}" == "1" ]]; then
      PYTHON_BIN="${PYTHON:-../.venv/bin/python}"
      [[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="$(command -v python)"
      "${PYTHON_BIN}" -u verify_revgat_anchor.py --seeds "${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
    else
      PROJECT_DIR="${PROJECT_DIR}" submit_sbatch slurm/reproduce_simteg_tape_revgat_anchor.sbatch
    fi
    ;;
  baseline)
    submit_array \
      "baseline" \
      "${COMPONENTS:-arxiv_e5 arxiv_roberta tape_e5 tape_roberta}" \
      "${SEEDS:-42}" \
      "${EPOCHS:-200}" \
      "${INCLUDE_GPT_PREDS:-0}" \
      "slurm/run_simteg_tape_linearrevgat_baseline.sbatch"
    ;;
  gsmp1)
    submit_array \
      "gsmp_first_layer" \
      "${COMPONENTS:-arxiv_e5 arxiv_roberta tape_e5 tape_roberta}" \
      "${SEEDS:-42}" \
      "${EPOCHS:-200}" \
      "${INCLUDE_GPT_PREDS:-0}" \
      "slurm/run_simteg_tape_linearrevgat_gsmp1.sbatch"
    ;;
  pgsmp)
    submit_array \
      "pgsmp" \
      "${COMPONENTS:-arxiv_e5 arxiv_roberta tape_e5 tape_roberta}" \
      "${SEEDS:-42}" \
      "${EPOCHS:-200}" \
      "0" \
      "slurm/run_simteg_tape_linearrevgat_pgsmp.sbatch"
    ;;
  all)
    submit_array \
      "${EXPERIMENT_MODES:-baseline gsmp_first_layer pgsmp}" \
      "${COMPONENTS:-arxiv_e5 arxiv_roberta tape_e5 tape_roberta}" \
      "${SEEDS:-42}" \
      "${EPOCHS:-200}" \
      "${INCLUDE_GPT_PREDS:-0}"
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    usage >&2
    exit 2
    ;;
esac
