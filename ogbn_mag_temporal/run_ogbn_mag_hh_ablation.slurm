#!/bin/bash
# SLURM job script for leaderboard-style HGAMLP-HOPE (HH) ablation on ogbn-mag
# 
# Submit with:
#     sbatch run_ogbn_mag_hh_ablation.slurm
#
# Monitor with:
#     tail -f logs/hh_${SLURM_JOB_ID}.out
#     tail -f logs/hh_smp_${SLURM_JOB_ID}.out
#     tail -f logs/hh_ump_${SLURM_JOB_ID}.out
#     tail -f logs/hh_gsmp_${SLURM_JOB_ID}.out
#
# Alternatively, run as array job (optional):
#     uncomment #SBATCH --array=0-3 below
#     task id 0 -> hh, 1 -> hh_smp, 2 -> hh_ump, 3 -> hh_gsmp

#SBATCH --job-name=hh_mag_ablation
#SBATCH --account=PAS1289
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
##SBATCH --array=0-3

set -euo pipefail

###############################################################################
# Paths and directories
###############################################################################

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${PROJECT_DIR}/HGAMLP_MAG"
VENV_DIR="${PROJECT_DIR}/.venv"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/hgamlp_mag_hh_ablation}"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}/logs" "${PROJECT_DIR}/logs"

###############################################################################
# Environment variables
###############################################################################

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="${PROJECT_DIR}/sparse_tools:${PYTHONPATH:-}"

###############################################################################
# Environment activation
###############################################################################

cd "${PROJECT_DIR}"

if command -v module >/dev/null 2>&1; then
    module load miniconda3/24.1.2-py310 2>/dev/null || true
fi

if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
    echo "Activated .venv"
else
    if [ -f ~/.bashrc ]; then
        source ~/.bashrc
        echo "Sourced ~/.bashrc"
    fi
    if command -v conda >/dev/null 2>&1; then
        CONDA_ENV_NAME=${CONDA_ENV_NAME:-graphenv}
        if conda env list | grep -q "${CONDA_ENV_NAME}"; then
            conda activate "${CONDA_ENV_NAME}"
            echo "Activated conda env: ${CONDA_ENV_NAME}"
        else
            echo "Warning: conda env ${CONDA_ENV_NAME} not found; proceeding with current Python"
        fi
    fi
fi

###############################################################################
# Print debug information
###############################################################################

echo "==============================================================================="
echo "Job Information"
echo "==============================================================================="
echo "hostname: $(hostname)"
echo "date: $(date)"
echo "job_id: ${SLURM_JOB_ID:-local}"
echo "array_task_id: ${SLURM_ARRAY_TASK_ID:-none}"
echo "project_dir: ${PROJECT_DIR}"
echo "code_dir: ${CODE_DIR}"
echo "output_root: ${OUTPUT_ROOT}"
echo "cpus_per_task: ${SLURM_CPUS_PER_TASK:-16}"
echo "mem_mb: ${SLURM_MEM_PER_NODE:-unknown}"

echo ""
echo "==============================================================================="
echo "Python & GPU Information"
echo "==============================================================================="
which python
python --version
echo ""
nvidia-smi || echo "Warning: nvidia-smi not available"

python - <<'PY'
import importlib.util
missing = [
    pkg
    for pkg in ("torch", "dgl", "ogb", "numpy", "sklearn", "torch_scatter", "torch_sparse", "sparse_tools")
    if importlib.util.find_spec(pkg) is None
]
if missing:
    raise SystemExit(
        "Missing packages for leaderboard HGAMLP_MAG: "
        + ", ".join(missing)
        + "\nUse the project .venv with DGL, OGB, torch_scatter, torch_sparse, and sparse_tools installed."
    )
print("Dependency import check OK")
PY

if [[ ! -f "${CODE_DIR}/mag.p" ]]; then
    cat >&2 <<EOF
ERROR: ${CODE_DIR}/mag.p is missing.

HH leaderboard reproduction requires the LINE embeddings described in:
  ${CODE_DIR}/README.md
Place mag.p at:
  ${CODE_DIR}/mag.p
EOF
    exit 3
fi

###############################################################################
# Experiment configuration
###############################################################################

METHODS=("hh" "hh_smp" "hh_ump" "hh_gsmp")
SEEDS="${SEEDS:-${SEED:-0 1 2}}"
if [[ -n "${EPOCHS:-}" && -z "${STAGES:-}" ]]; then
    echo "Warning: EPOCHS=${EPOCHS} is ignored by the leaderboard HGAMLP_MAG runner; use STAGES instead."
fi
STAGES="${STAGES:-400 400 400 500}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
GPU="${GPU:-0}"
NUM_HOPS="${NUM_HOPS:-2}"
NUM_LABEL_HOPS="${NUM_LABEL_HOPS:-2}"
LR="${LR:-0.002}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
PATIENCE="${PATIENCE:-100}"
THRESHOLD="${THRESHOLD:-0.75}"
GAMA="${GAMA:-10}"
METHODS_OVERRIDE="${METHODS_OVERRIDE:-}"

if [[ -n "${METHODS_OVERRIDE}" ]]; then
    read -r -a METHODS <<< "${METHODS_OVERRIDE}"
elif [[ -n "${METHOD:-}" ]]; then
    read -r -a METHODS <<< "${METHOD}"
fi

# Determine which methods to run
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    METHOD_INDEX=${SLURM_ARRAY_TASK_ID}
    if (( METHOD_INDEX < 0 || METHOD_INDEX >= ${#METHODS[@]} )); then
        echo "ERROR: SLURM_ARRAY_TASK_ID=${METHOD_INDEX} is outside 0..$((${#METHODS[@]} - 1))." >&2
        exit 4
    fi
    METHODS_TO_RUN=("${METHODS[${METHOD_INDEX}]}")
    echo "Array mode: running method ${METHODS[${METHOD_INDEX}]} (array task ${SLURM_ARRAY_TASK_ID})"
else
    METHODS_TO_RUN=("${METHODS[@]}")
    echo "Sequential mode: running all ${#METHODS[@]} methods"
fi

echo ""
echo "Methods: ${METHODS_TO_RUN[*]}"
echo "Seeds: ${SEEDS}"
echo "Stages: ${STAGES}"
echo "Batch size: ${BATCH_SIZE}"
echo "GPU: ${GPU}"
echo ""

impact_method_for() {
    case "$1" in
        hh) echo "baseline" ;;
        hh_smp) echo "smp" ;;
        hh_ump) echo "ump" ;;
        hh_gsmp) echo "gsmp" ;;
        baseline|smp|ump|gsmp) echo "$1" ;;
        *) echo "ERROR: unknown method '$1'. Expected hh, hh_smp, hh_ump, hh_gsmp." >&2; return 1 ;;
    esac
}

###############################################################################
# Run experiments
###############################################################################

OVERALL_START=$(date +%s.%N)
cd "${CODE_DIR}"

for METHOD in "${METHODS_TO_RUN[@]}"; do
    IMPACT_METHOD="$(impact_method_for "${METHOD}")"
    METHOD_LOG="${LOG_DIR}/${METHOD}_${SLURM_JOB_ID:-local}.out"

    echo "==============================================================================="
    echo "Running method: ${METHOD} (HGAMLP_MAG --impact-method ${IMPACT_METHOD})"
    echo "Log: ${METHOD_LOG}"
    echo "==============================================================================="
    METHOD_START=$(date +%s)
    
    python -u main.py \
        --impact-method "${IMPACT_METHOD}" \
        --stages ${STAGES} \
        --extra-embedding Line \
        --num-hops "${NUM_HOPS}" \
        --label-feats \
        --num-label-hops "${NUM_LABEL_HOPS}" \
        --n-layers-1 2 \
        --n-layers-2 2 \
        --n-layers-3 3 \
        --act leaky_relu \
        --bns \
        --label-bns \
        --lr "${LR}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --threshold "${THRESHOLD}" \
        --patience "${PATIENCE}" \
        --gama "${GAMA}" \
        --amp \
        --seeds ${SEEDS} \
        --gpu "${GPU}" \
        --batch-size "${BATCH_SIZE}" \
        2>&1 | tee "${METHOD_LOG}"
    
    METHOD_END=$(date +%s)
    METHOD_TIME=$((METHOD_END - METHOD_START))
    echo "Method ${METHOD} completed in ${METHOD_TIME}s at $(date)"
    echo ""
done

OVERALL_END=$(date +%s.%N)
OVERALL_TIME=$(echo "${OVERALL_END} - ${OVERALL_START}" | bc)

###############################################################################
# Summary
###############################################################################

echo "==============================================================================="
echo "Ablation Complete"
echo "==============================================================================="
echo "Total time: ${OVERALL_TIME}s"
echo "Start time: $(date -d @$(echo ${OVERALL_START} | cut -d. -f1) '+%Y-%m-%d %H:%M:%S')"
echo "End time: $(date)"
echo ""

echo ""
echo "Method logs: ${LOG_DIR}/"
echo "HGAMLP checkpoints/raw outputs: ${CODE_DIR}/output/ogbn-mag/"
echo "A previous verified HH leaderboard-style run in this repo reached Val 60.1227, Test 58.2250."
echo ""
echo "To monitor logs in real-time, use:"
echo "  tail -f ${LOG_DIR}/hh_${SLURM_JOB_ID:-local}.out"
echo "  tail -f ${LOG_DIR}/hh_smp_${SLURM_JOB_ID:-local}.out"
echo "  tail -f ${LOG_DIR}/hh_ump_${SLURM_JOB_ID:-local}.out"
echo "  tail -f ${LOG_DIR}/hh_gsmp_${SLURM_JOB_ID:-local}.out"
