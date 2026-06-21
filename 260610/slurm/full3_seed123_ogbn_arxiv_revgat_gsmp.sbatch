#!/bin/bash
#SBATCH --job-name=260610_rg3_full
#SBATCH --account=PAS1289
#SBATCH --output=/users/PAS1289/jyp531/GSMP-research/260610/runs/ogbn_arxiv_simteg_tape_revgat_gsmp/logs/slurm_full3_seed123_%j.out
#SBATCH --error=/users/PAS1289/jyp531/GSMP-research/260610/runs/ogbn_arxiv_simteg_tape_revgat_gsmp/logs/slurm_full3_seed123_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

set -euo pipefail

module load cuda/12.4.1 2>/dev/null || true
export DGLBACKEND=pytorch

if [[ "${CONFIRM_3SEED_FULL:-0}" != "1" ]]; then
  echo "Refusing to launch the 3-seed full run without CONFIRM_3SEED_FULL=1."
  echo "Example: CONFIRM_3SEED_FULL=1 sbatch slurm/full3_seed123_ogbn_arxiv_revgat_gsmp.sbatch"
  exit 2
fi

cd /users/PAS1289/jyp531/GSMP-research/260610

echo "3-seed full run: seeds=1 2 3, max_epochs=200, early_stop_min_epochs=80, patience=40."
echo "This uses LinearRevGAT and LinearRevGAT+GSMP over all five SimTeG/TAPE/GPT-preds components."

COMPONENT_LIMIT="${COMPONENT_LIMIT:-5}" \
PREFIX="${PREFIX:-full3_seed123}" \
VARIANTS="${VARIANTS:-linear gsmp}" \
SEEDS="${SEEDS:-1 2 3}" \
EPOCHS="${EPOCHS:-200}" \
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-40}" \
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-80}" \
N_LABEL_ITERS="${N_LABEL_ITERS:-2}" \
EDGE_DROP="${EDGE_DROP:-0.46}" \
USE_NORM="${USE_NORM:-0}" \
SAVE_PRED="${SAVE_PRED:-1}" \
bash scripts/run_linear_components.sh
