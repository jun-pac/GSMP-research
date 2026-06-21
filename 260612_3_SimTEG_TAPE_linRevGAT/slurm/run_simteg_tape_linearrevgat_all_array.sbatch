#!/bin/bash
#SBATCH --job-name=linrevgat-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j_%a.out
#SBATCH --error=logs/%x_%j_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_DIR}" ]]; then
  PROJECT_DIR="$(pwd)"
fi
cd "${PROJECT_DIR}"
mkdir -p logs

# EDIT THIS SECTION FOR YOUR CLUSTER
set +u
source ~/.bashrc || true
set -u
if command -v conda >/dev/null 2>&1; then
  set +u
  conda activate "${CONDA_ENV:-simteg}" || true
  set -u
fi

hostname
date
pwd
git rev-parse HEAD || true
nvidia-smi || true
python --version || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())" || true
python -c "import torch_geometric; print('pyg', torch_geometric.__version__)" || true
python -c "import dgl; print('dgl', dgl.__version__)" || true

IFS=' ' read -r -a MODES <<< "${EXPERIMENT_MODES:-baseline gsmp_first_layer pgsmp}"
IFS=' ' read -r -a COMPONENT_ARRAY <<< "${COMPONENTS:-arxiv_e5 arxiv_roberta tape_e5 tape_roberta}"
if [[ "${INCLUDE_GPT_PREDS:-0}" == "1" ]]; then
  COMPONENT_ARRAY+=(gpt_preds)
fi
IFS=' ' read -r -a SEED_ARRAY <<< "${SEEDS:-42}"
EPOCHS="${EPOCHS:-3}"

TOTAL=$(( ${#MODES[@]} * ${#COMPONENT_ARRAY[@]} * ${#SEED_ARRAY[@]} ))
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID >= TOTAL )); then
  echo "[NOOP] task_id=${TASK_ID} total=${TOTAL}"
  exit 0
fi

MODE_INDEX=$(( TASK_ID / (${#COMPONENT_ARRAY[@]} * ${#SEED_ARRAY[@]}) ))
REM=$(( TASK_ID % (${#COMPONENT_ARRAY[@]} * ${#SEED_ARRAY[@]}) ))
COMP_INDEX=$(( REM / ${#SEED_ARRAY[@]} ))
SEED_INDEX=$(( REM % ${#SEED_ARRAY[@]} ))

MODE="${MODES[$MODE_INDEX]}"
COMPONENT="${COMPONENT_ARRAY[$COMP_INDEX]}"
SEED="${SEED_ARRAY[$SEED_INDEX]}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)}"

echo "[TASK] task_id=${TASK_ID}/${TOTAL} mode=${MODE} component=${COMPONENT} seed=${SEED} epochs=${EPOCHS}"
bash scripts/run_one_linearrevgat_component.sh "${MODE}" "${COMPONENT}" "${SEED}" "${EPOCHS}" "${RUN_ID_PREFIX}"
