#!/bin/bash
#SBATCH --job-name=elliptic-smp-gcn
#SBATCH --output=logs/elliptic_smp_%j.out
#SBATCH --error=logs/elliptic_smp_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Set this before submitting, or edit the default below.
ELLIPTIC_ROOT="${ELLIPTIC_ROOT:-/path/to/elliptic_bitcoin_dataset}"

# Optional: uncomment and adjust for your cluster.
# module purge
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate pyg

echo "Running on $(hostname)"
echo "ELLIPTIC_ROOT=${ELLIPTIC_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"

python3 elliptic_bitcoin_smp.py \
  --root "${ELLIPTIC_ROOT}" \
  --device cuda:0 \
  --use-smp \
  --hidden-channels 128 \
  --num-layers 2 \
  --dropout 0.5 \
  --lr 0.01 \
  --weight-decay 5e-4 \
  --epochs 200 \
  --result-dir results \
  --tag smp_undirected_gcn_L2_H128_lr0.01_wd5e-4
