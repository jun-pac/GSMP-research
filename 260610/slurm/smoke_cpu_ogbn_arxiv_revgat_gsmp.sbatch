#!/bin/bash
#SBATCH --job-name=260610_revgat_cpu_smoke
#SBATCH --account=PAS1289
#SBATCH --output=/users/PAS1289/jyp531/GSMP-research/260610/runs/ogbn_arxiv_simteg_tape_revgat_gsmp/logs/slurm_cpu_smoke_%j.out
#SBATCH --error=/users/PAS1289/jyp531/GSMP-research/260610/runs/ogbn_arxiv_simteg_tape_revgat_gsmp/logs/slurm_cpu_smoke_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail

cd /users/PAS1289/jyp531/GSMP-research/260610

echo "CPU smoke run: no GPU requested. Use this while DGL is CPU-only."

CPU=1 \
PREFIX="${PREFIX:-cpu_smoke}" \
COMPONENT_LIMIT="${COMPONENT_LIMIT:-1}" \
VARIANTS="${VARIANTS:-linear gsmp}" \
SEEDS="${SEEDS:-42}" \
EPOCHS="${EPOCHS:-1}" \
N_LABEL_ITERS="${N_LABEL_ITERS:-0}" \
EDGE_DROP="${EDGE_DROP:-0.0}" \
SAVE_PRED="${SAVE_PRED:-0}" \
bash scripts/run_linear_components.sh
