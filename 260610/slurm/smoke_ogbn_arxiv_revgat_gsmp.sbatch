#!/bin/bash
#SBATCH --job-name=260610_revgat_smoke
#SBATCH --account=PAS1289
#SBATCH --output=/users/PAS1289/jyp531/GSMP-research/260610/runs/ogbn_arxiv_simteg_tape_revgat_gsmp/logs/slurm_smoke_%j.out
#SBATCH --error=/users/PAS1289/jyp531/GSMP-research/260610/runs/ogbn_arxiv_simteg_tape_revgat_gsmp/logs/slurm_smoke_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00

set -euo pipefail

module load cuda/12.4.1 2>/dev/null || true
export DGLBACKEND=pytorch

cd /users/PAS1289/jyp531/GSMP-research/260610

echo "Smoke run: one cached-embedding component, linear and GSMP, short epochs."
echo "This validates code paths without trying to reproduce leaderboard numbers."

COMPONENT_LIMIT="${COMPONENT_LIMIT:-1}" \
VARIANTS="${VARIANTS:-linear gsmp}" \
SEEDS="${SEEDS:-42}" \
EPOCHS="${EPOCHS:-2}" \
N_LABEL_ITERS="${N_LABEL_ITERS:-0}" \
EDGE_DROP="${EDGE_DROP:-0.0}" \
SAVE_PRED="${SAVE_PRED:-1}" \
bash scripts/run_linear_components.sh

if [[ "${RUN_OFFICIAL_BASELINE:-0}" == "1" ]]; then
  echo "Optional official RevGAT baseline smoke."
  COMPONENT_LIMIT="${OFFICIAL_COMPONENT_LIMIT:-1}" \
  SEED="${OFFICIAL_SEED:-42}" \
  N_RUNS="${OFFICIAL_N_RUNS:-1}" \
  EPOCHS="${OFFICIAL_EPOCHS:-2}" \
  bash scripts/run_official_revgat_components.sh
fi
