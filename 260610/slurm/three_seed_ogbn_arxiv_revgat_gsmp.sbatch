#!/bin/bash
#SBATCH --job-name=260610_revgat_3seed
#SBATCH --account=PAS1289
#SBATCH --output=/users/PAS1289/jyp531/GSMP-research/260610/runs/ogbn_arxiv_simteg_tape_revgat_gsmp/logs/slurm_3seed_%j.out
#SBATCH --error=/users/PAS1289/jyp531/GSMP-research/260610/runs/ogbn_arxiv_simteg_tape_revgat_gsmp/logs/slurm_3seed_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

set -euo pipefail

module load cuda/12.4.1 2>/dev/null || true
export DGLBACKEND=pytorch

cd /users/PAS1289/jyp531/GSMP-research/260610

echo "Three-seed sanity run. Defaults are conservative; raise EPOCHS after smoke logs look healthy."

COMPONENT_LIMIT="${COMPONENT_LIMIT:-5}" \
PREFIX="${PREFIX:-three_seed}" \
VARIANTS="${VARIANTS:-linear gsmp}" \
SEEDS="${SEEDS:-42 43 44}" \
EPOCHS="${EPOCHS:-30}" \
N_LABEL_ITERS="${N_LABEL_ITERS:-0}" \
EDGE_DROP="${EDGE_DROP:-0.0}" \
SAVE_PRED="${SAVE_PRED:-1}" \
bash scripts/run_linear_components.sh
