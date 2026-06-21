#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOPE_DIR="${PROJECT_DIR}/HOPE"
RESEARCH_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
METHODS_STR="${METHODS:-baseline gsmp}"
SEEDS_STR="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
read -r -a METHODS_ARR <<< "${METHODS_STR}"
read -r -a SEEDS_ARR <<< "${SEEDS_STR}"

NUM_METHODS=${#METHODS_ARR[@]}
NUM_SEEDS=${#SEEDS_ARR[@]}
TOTAL=$((NUM_METHODS * NUM_SEEDS))

if (( TASK_ID >= TOTAL )); then
    echo "Task ${TASK_ID} outside grid size ${TOTAL}; exiting."
    exit 0
fi

METHOD_INDEX=$((TASK_ID / NUM_SEEDS))
SEED_INDEX=$((TASK_ID % NUM_SEEDS))
METHOD=${METHODS_ARR[$METHOD_INDEX]}
SEED=${SEEDS_ARR[$SEED_INDEX]}

case "${METHOD}" in
    baseline|gsmp|smp) ;;
    *)
        echo "Invalid METHOD=${METHOD}; use baseline, gsmp, or smp." >&2
        exit 2
        ;;
esac

JOB_GROUP=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}
RUN_NAME="${RUN_NAME:-sehgnn_hope_compare}"
LOG_DIR="${PROJECT_DIR}/logs/${JOB_GROUP}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_${METHOD}_seed${SEED}.log"
mkdir -p "${LOG_DIR}" "${PROJECT_DIR}/results" "${PROJECT_DIR}/cache/gsmp" "${PROJECT_DIR}/cache/propagation"
ln -sfn "${LOG_DIR}" "${PROJECT_DIR}/logs/latest"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "SeHGNN-HOPE ogbn-mag run"
echo "============================================================"
date
hostname
echo "project=${PROJECT_DIR}"
echo "hope_dir=${HOPE_DIR}"
echo "job_group=${JOB_GROUP}"
echo "task_id=${TASK_ID}/${TOTAL}"
echo "method=${METHOD}"
echo "seed=${SEED}"
echo "run_name=${RUN_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "log_file=${LOG_FILE}"
echo "============================================================"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

if command -v module >/dev/null 2>&1; then
    module load miniconda3/24.1.2-py310 2>/dev/null || true
    module load cuda/11.8.0 2>/dev/null || true
fi

if [[ "${SKIP_ENV_ACTIVATE:-0}" != "1" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    fi

    CONDA_ENV="${CONDA_ENV:-hope_official}"
    FALLBACK_CONDA_ENV="${FALLBACK_CONDA_ENV:-hope}"
    if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
        conda activate "${CONDA_ENV}"
        echo "env=conda:${CONDA_ENV}"
    elif command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "${FALLBACK_CONDA_ENV}"; then
        conda activate "${FALLBACK_CONDA_ENV}"
        echo "env=conda:${FALLBACK_CONDA_ENV}"
    else
        echo "No conda env ${CONDA_ENV} or ${FALLBACK_CONDA_ENV}; using current Python." >&2
    fi
fi

export PYTHONPATH="${PROJECT_DIR}/sparse_tools:${HOPE_DIR}:${PYTHONPATH:-}"
python --version
python "${PROJECT_DIR}/env/check_env.py"
nvidia-smi || true

"${PROJECT_DIR}/scripts/prepare_data_links.sh"

cd "${HOPE_DIR}"

STAGES_STR="${STAGES:-300 300 300 300}"
read -r -a STAGES_ARR <<< "${STAGES_STR}"
ROOT="${ROOT:-${HOPE_DIR}/dataset}"
EMB_PATH="${EMB_PATH:-${HOPE_DIR}/dataset/ogbn_mag}"
PROGRESS_FILE="${PROGRESS_FILE:-${PROJECT_DIR}/results/${RUN_NAME}_live_progress.tsv}"
PROP_CACHE_DIR="${PROP_CACHE_DIR:-${PROJECT_DIR}/cache/propagation}"
GSMP_CACHE_DIR="${GSMP_CACHE_DIR:-${PROJECT_DIR}/cache/gsmp}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
PATIENCE="${PATIENCE:-100}"
EVAL_EVERY="${EVAL_EVERY:-1}"
EXTRA_ARGS_STR="${EXTRA_ARGS:-}"

CMD=(
    python -u training.py
    --dataset ogbn-mag
    --aggregation SeHGNN-HOPE
    --label-residual
    --similarity-threshold 0.6
    --lower-bound 0.5
    --upper-bound 3
    --lamb 0.5
    --hidden 512
    --n-layers-1 2
    --n-layers-2 2
    --lr 0.001
    --weight-decay 0
    --stages "${STAGES_ARR[@]}"
    --num-hops 2
    --label-feats
    --num-label-hops 2
    --extra-embedding Line
    --amp
    --use-sparse-tools
    --eval-every "${EVAL_EVERY}"
    --batch-size "${BATCH_SIZE}"
    --patience "${PATIENCE}"
    --seeds "${SEED}"
    --root "${ROOT}"
    --emb_path "${EMB_PATH}"
    --progress-file "${PROGRESS_FILE}"
    --propagation-cache-dir "${PROP_CACHE_DIR}"
    --gsmp-cache-dir "${GSMP_CACHE_DIR}"
    --method-name "${METHOD}"
    --gc-every "${GC_EVERY:-1}"
)

if [[ "${EMPTY_CACHE_EVERY_EPOCH:-0}" == "1" ]]; then
    CMD+=(--empty-cache-every-epoch)
fi

if [[ "${METHOD}" == "gsmp" ]]; then
    CMD+=(
        --gsmp-first-layer
        --gsmp-scope "${GSMP_SCOPE:-paper-stack}"
        --gsmp-normalizer "${GSMP_NORMALIZER:-nonempty}"
        --gsmp-time-source "${GSMP_TIME_SOURCE:-all}"
        --gsmp-derived-time "${GSMP_DERIVED_TIME:-mode}"
    )
elif [[ "${METHOD}" == "smp" ]]; then
    CMD+=(
        --smp-first-layer
        --gsmp-scope "${GSMP_SCOPE:-paper-stack}"
        --gsmp-time-source "${GSMP_TIME_SOURCE:-all}"
        --gsmp-derived-time "${GSMP_DERIVED_TIME:-mode}"
    )
fi

if [[ "${GSMP_APPLY_LABEL_PROP:-0}" == "1" ]]; then
    CMD+=(--gsmp-apply-label-prop)
fi

if [[ "${NO_CACHE_PROPAGATION:-0}" == "1" ]]; then
    CMD+=(--no-cache-propagation)
fi

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
echo "progress_file=${PROGRESS_FILE}"
echo "============================================================"
