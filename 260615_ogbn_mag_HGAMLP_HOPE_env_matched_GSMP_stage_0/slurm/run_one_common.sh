#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOPE_DIR="${PROJECT_DIR}/HOPE"
RESEARCH_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [[ -n "${JOB_SPECS:-}" ]]; then
    read -r -a JOB_SPECS_ARR <<< "${JOB_SPECS}"
    TOTAL=${#JOB_SPECS_ARR[@]}
    if (( TASK_ID >= TOTAL )); then
        echo "Task ${TASK_ID} is outside configured explicit job list size ${TOTAL}; exiting."
        exit 0
    fi
    JOB_SPEC=${JOB_SPECS_ARR[$TASK_ID]}
    if [[ "${JOB_SPEC}" != *:* ]]; then
        echo "Invalid JOB_SPECS entry '${JOB_SPEC}'. Expected method:seed, e.g. smp:2." >&2
        exit 2
    fi
    METHOD=${JOB_SPEC%%:*}
    SEED=${JOB_SPEC##*:}
else
    METHODS_STR="${METHODS:-none smp gsmp}"
    SEEDS_STR="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
    read -r -a METHODS_ARR <<< "${METHODS_STR}"
    read -r -a SEEDS_ARR <<< "${SEEDS_STR}"

    NUM_METHODS=${#METHODS_ARR[@]}
    NUM_SEEDS=${#SEEDS_ARR[@]}
    TOTAL=$((NUM_METHODS * NUM_SEEDS))

    if (( TASK_ID >= TOTAL )); then
        echo "Task ${TASK_ID} is outside configured method/seed grid size ${TOTAL}; exiting."
        exit 0
    fi

    METHOD_INDEX=$((TASK_ID / NUM_SEEDS))
    SEED_INDEX=$((TASK_ID % NUM_SEEDS))
    METHOD=${METHODS_ARR[$METHOD_INDEX]}
    SEED=${SEEDS_ARR[$SEED_INDEX]}
fi

JOB_GROUP=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}
LOG_DIR="${PROJECT_DIR}/logs/${JOB_GROUP}"
RUN_NAME="${RUN_NAME:-${METHOD}}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_${METHOD}_seed${SEED}.log"
mkdir -p "${LOG_DIR}" "${PROJECT_DIR}/results" "${PROJECT_DIR}/impact_cache"
ln -sfn "${LOG_DIR}" "${PROJECT_DIR}/logs/latest"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "HOPE ogbn-mag impact run"
echo "============================================================"
date
hostname
echo "project=${PROJECT_DIR}"
echo "hope_dir=${HOPE_DIR}"
echo "job_group=${JOB_GROUP}"
echo "task_id=${TASK_ID}"
echo "method=${METHOD}"
echo "seed=${SEED}"
echo "run_name=${RUN_NAME}"
echo "impact_apply_to=${IMPACT_APPLY_TO:-both}"
echo "impact_stages=${IMPACT_STAGES:-all}"
echo "gsmp_layer_mode=${GSMP_LAYER_MODE:-first}"
echo "job_specs=${JOB_SPECS:-grid:${METHODS_STR:-none smp gsmp} x ${SEEDS_STR:-1 2 3 4 5 6 7 8 9 10}}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "log_file=${LOG_FILE}"
echo "============================================================"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTHONPATH="${PROJECT_DIR}/sparse_tools:${HOPE_DIR}:${PYTHONPATH:-}"

if command -v module >/dev/null 2>&1; then
    module load miniconda3/24.1.2-py310 2>/dev/null || true
    module load cuda/11.8.0 2>/dev/null || true
fi

CONDA_ENV="${CONDA_ENV:-hope}"
VENV_DIR="${VENV_DIR:-${RESEARCH_DIR}/.venv}"
USING_ENV="system"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda activate "${CONDA_ENV}"
    USING_ENV="conda:${CONDA_ENV}"
elif [[ -f "${VENV_DIR}/bin/activate" ]]; then
    source "${VENV_DIR}/bin/activate"
    USING_ENV="venv:${VENV_DIR}"
else
    echo "WARNING: no conda env '${CONDA_ENV}' and no venv at ${VENV_DIR}; using current Python." >&2
fi

echo "env=${USING_ENV}"
python --version
python - <<'PY' || true
import importlib
for name in ["torch", "dgl", "ogb", "torch_sparse"]:
    try:
        mod = importlib.import_module(name)
        print(f"{name}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}=MISSING ({exc})")
PY
nvidia-smi || true

cd "${HOPE_DIR}"

STAGES_STR="${STAGES:-300 300 300 300}"
read -r -a STAGES_ARR <<< "${STAGES_STR}"
EVAL_EVERY="${EVAL_EVERY:-1}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
PATIENCE="${PATIENCE:-100}"
IMPACT_APPLY_TO="${IMPACT_APPLY_TO:-both}"
IMPACT_STAGES="${IMPACT_STAGES:-all}"
GSMP_LAYER_MODE="${GSMP_LAYER_MODE:-first}"
ROOT="${ROOT:-./dataset/}"
EMB_PATH="${EMB_PATH:-./dataset/ogbn_mag/}"
PROGRESS_FILE="${PROGRESS_FILE:-${PROJECT_DIR}/results/${RUN_NAME}_live_progress.tsv}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/impact_cache}"
EXTRA_ARGS_STR="${EXTRA_ARGS:-}"
if [[ ! -f "${PROGRESS_FILE}" ]]; then
    mkdir -p "$(dirname "${PROGRESS_FILE}")"
    printf 'timestamp\tjob_id\tmethod\tseed\tstage\tepoch\ttrain_acc\tval_acc\ttest_acc\tbest_epoch\tbest_val\tbest_test_at_best_val\telapsed_sec\n' > "${PROGRESS_FILE}"
fi

CMD=(
    python -u training.py
    --aggregation HGAMLP-HOPE
    --impact-method "${METHOD}"
    --impact-apply-to "${IMPACT_APPLY_TO}"
    --impact-stages "${IMPACT_STAGES}"
    --label-residual
    --alpha 0.5
    --similarity-threshold 0.6
    --lower-bound 0.5
    --upper-bound 3
    --seeds "${SEED}"
    --stages "${STAGES_ARR[@]}"
    --eval-every "${EVAL_EVERY}"
    --batch-size "${BATCH_SIZE}"
    --patience "${PATIENCE}"
    --root "${ROOT}"
    --emb_path "${EMB_PATH}"
    --impact-cache-dir "${CACHE_DIR}"
    --progress-file "${PROGRESS_FILE}"
    --use-sparse-tools
)

case "${GSMP_LAYER_MODE}" in
    first)
        CMD+=(--impact-gsmp-first-layer-only)
        ;;
    all)
        CMD+=(--no-impact-gsmp-first-layer-only)
        ;;
    *)
        echo "Invalid GSMP_LAYER_MODE='${GSMP_LAYER_MODE}'. Use 'first' or 'all'." >&2
        exit 2
        ;;
esac

if [[ -n "${EXTRA_ARGS_STR}" ]]; then
    read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS_STR}"
    CMD+=("${EXTRA_ARGS_ARR[@]}")
fi

echo "command=${CMD[*]}"
echo "============================================================"

if command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL "${CMD[@]}"
else
    "${CMD[@]}"
fi

echo "============================================================"
echo "finished method=${METHOD} seed=${SEED} at $(date)"
echo "log_file=${LOG_FILE}"
echo "progress_file=${PROGRESS_FILE}"
echo "============================================================"
