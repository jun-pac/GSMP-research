#!/bin/bash
# Main controlled experiment: baseline, SMP, UMP, GSMP; seeds 1,2,3; 100 epochs.
#
# Submit:
#   cd /users/PAS1289/jyp531/GSMP-research/260609
#   mkdir -p logs/main
#   sbatch slurm/main_ogbn_arxiv_smpumpgsmp.sbatch
#
# Optional final reproduction after the 3-seed run looks reasonable:
#   SEEDS="1 2 3 4 5 6 7 8 9 10" sbatch slurm/main_ogbn_arxiv_smpumpgsmp.sbatch
#
# Monitor:
#   tail -f logs/main/main_arxiv_mp_<jobid>.out

#SBATCH --job-name=main_arxiv_mp
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --output=logs/main/%x_%j.out
#SBATCH --error=logs/main/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --exclude=p0240

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/users/PAS1289/jyp531/GSMP-research/260609}"
cd "${PROJECT_DIR}"
mkdir -p logs/main results checkpoints

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

DATASET="${DATASET:-ogbn-arxiv}"
VARIANTS="${VARIANTS:-baseline smp ump gsmp}"
BERT_X_DIR="${BERT_X_DIR:-/users/PAS1289/jyp531/GSMP-research/SimTeG/out/ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt}"
SEEDS="${SEEDS:-1 2 3}"
EPOCHS="${EPOCHS:-100}"
RUN_NAME="${RUN_NAME:-main_${SLURM_JOB_ID:-manual}}"

echo "hostname=$(hostname)"
echo "date=$(date)"
echo "project_dir=${PROJECT_DIR}"
echo "run_name=${RUN_NAME}"
echo "variants=${VARIANTS}"
echo "seeds=${SEEDS}"
echo "epochs=${EPOCHS}"
echo "bert_x_dir=${BERT_X_DIR}"
nvidia-smi || true

bash scripts/run_all_variants.sh \
  "${DATASET}" \
  "${VARIANTS}" \
  "${SEEDS}" \
  "${EPOCHS}" \
  "${BERT_X_DIR}" \
  "${RUN_NAME}"
