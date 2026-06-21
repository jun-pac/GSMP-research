#!/bin/bash
# Leaderboard-style component run for SimTeG+TAPE GraphSAGE.
#
# Submit a quick 3-seed baseline/GSMP comparison:
#   cd /users/PAS1289/jyp531/GSMP-research/260609
#   mkdir -p logs/main
#   bash scripts/download_gpt_preds.sh
#   VARIANTS="baseline gsmp" SEEDS="42 43 44" PREFIX=leaderboard3 \
#     sbatch slurm/leaderboard_components_ogbn_arxiv.sbatch
#
# Official-style 10-seed run:
#   VARIANTS="baseline gsmp" SEEDS="42 43 44 45 46 47 48 49 50 51" PREFIX=leaderboard10 \
#     sbatch slurm/leaderboard_components_ogbn_arxiv.sbatch

#SBATCH --job-name=leader_arxiv_mp
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --output=logs/main/%x_%j.out
#SBATCH --error=logs/main/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --exclude=p0240

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/users/PAS1289/jyp531/GSMP-research/260609}"
cd "${PROJECT_DIR}"
mkdir -p logs/main results checkpoints

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

VARIANTS="${VARIANTS:-baseline gsmp}"
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-50}"
PREFIX="${PREFIX:-leaderboard_${SLURM_JOB_ID:-manual}}"
INCLUDE_GPT_PREDS="${INCLUDE_GPT_PREDS:-1}"

echo "hostname=$(hostname)"
echo "date=$(date)"
echo "project_dir=${PROJECT_DIR}"
echo "prefix=${PREFIX}"
echo "variants=${VARIANTS}"
echo "seeds=${SEEDS}"
echo "epochs=${EPOCHS}"
echo "include_gpt_preds=${INCLUDE_GPT_PREDS}"
nvidia-smi || true

VARIANTS="${VARIANTS}" SEEDS="${SEEDS}" EPOCHS="${EPOCHS}" PREFIX="${PREFIX}" INCLUDE_GPT_PREDS="${INCLUDE_GPT_PREDS}" \
  bash scripts/run_leaderboard_components.sh
