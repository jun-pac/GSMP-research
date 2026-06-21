#!/bin/bash
# Smoke test: baseline, SMP, UMP, GSMP; seed 1; 5 epochs.
#
# Submit:
#   cd /users/PAS1289/jyp531/GSMP-research/260609
#   mkdir -p logs/smoke
#   sbatch slurm/smoke_ogbn_arxiv_smpumpgsmp.sbatch
#
# Monitor:
#   tail -f logs/smoke/smoke_arxiv_mp_<jobid>.out

#SBATCH --job-name=smoke_arxiv_mp
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --output=logs/smoke/%x_%j.out
#SBATCH --error=logs/smoke/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --exclude=p0240

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/users/PAS1289/jyp531/GSMP-research/260609}"
cd "${PROJECT_DIR}"
mkdir -p logs/smoke results checkpoints

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export RUN_UNIT_TEST=1

DATASET="${DATASET:-ogbn-arxiv}"
VARIANTS="${VARIANTS:-baseline smp ump gsmp}"
BERT_X_DIR="${BERT_X_DIR:-/users/PAS1289/jyp531/GSMP-research/SimTeG/out/ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt}"
RUN_NAME="${RUN_NAME:-smoke_${SLURM_JOB_ID:-manual}}"

echo "hostname=$(hostname)"
echo "date=$(date)"
echo "project_dir=${PROJECT_DIR}"
echo "run_name=${RUN_NAME}"
echo "variants=${VARIANTS}"
echo "bert_x_dir=${BERT_X_DIR}"
nvidia-smi || true

bash scripts/run_all_variants.sh \
  "${DATASET}" \
  "${VARIANTS}" \
  "1" \
  "5" \
  "${BERT_X_DIR}" \
  "${RUN_NAME}"
