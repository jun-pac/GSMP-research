#!/bin/bash
#SBATCH --job-name=260610_revgat_10seed
#SBATCH --account=PAS1289
#SBATCH --output=/users/PAS1289/jyp531/GSMP-research/260610/runs/ogbn_arxiv_simteg_tape_revgat_gsmp/logs/slurm_10seed_%j.out
#SBATCH --error=/users/PAS1289/jyp531/GSMP-research/260610/runs/ogbn_arxiv_simteg_tape_revgat_gsmp/logs/slurm_10seed_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail

module load cuda/12.4.1 2>/dev/null || true
export DGLBACKEND=pytorch

if [[ "${CONFIRM_FINAL:-0}" != "1" ]]; then
  echo "Refusing to launch the 10-seed final run without CONFIRM_FINAL=1."
  echo "Example: CONFIRM_FINAL=1 sbatch slurm/ten_seed_ogbn_arxiv_revgat_gsmp.sbatch"
  exit 2
fi

cd /users/PAS1289/jyp531/GSMP-research/260610

echo "Final 10-seed run. This is intentionally guarded because it is the expensive path."

COMPONENT_LIMIT="${COMPONENT_LIMIT:-5}" \
VARIANTS="${VARIANTS:-linear gsmp}" \
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}" \
EPOCHS="${EPOCHS:-200}" \
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-40}" \
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-80}" \
N_LABEL_ITERS="${N_LABEL_ITERS:-2}" \
EDGE_DROP="${EDGE_DROP:-0.46}" \
USE_NORM="${USE_NORM:-0}" \
SAVE_PRED="${SAVE_PRED:-1}" \
bash scripts/run_linear_components.sh
