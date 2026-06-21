#!/bin/bash
#SBATCH --job-name=pokec-linkx-gsmp
#SBATCH --account=PAS1289
#SBATCH --output=logs/pokec_raw_linkx_gsmp_%j.out
#SBATCH --error=logs/pokec_raw_linkx_gsmp_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results data/pokec

module load python/3.10
source .venv/bin/activate

POKEC_ROOT="${POKEC_ROOT:-data/pokec}"
EPOCHS="${EPOCHS:-200}"
LOG_EVERY="${LOG_EVERY:-10}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-64}"
TRAIN_UNTIL_YEAR="${TRAIN_UNTIL_YEAR:-2009}"
VAL_YEAR="${VAL_YEAR:-2010}"
TEST_FROM_YEAR="${TEST_FROM_YEAR:-2011}"
TAG="${TAG:-pokec_raw_linkx_vs_linkx_gsmp_train${TRAIN_UNTIL_YEAR}_val${VAL_YEAR}_test${TEST_FROM_YEAR}}"

echo "Running on $(hostname)"
echo "Python=$(command -v python3)"
echo "POKEC_ROOT=${POKEC_ROOT}"
echo "EPOCHS=${EPOCHS}"
echo "LOG_EVERY=${LOG_EVERY}"
echo "HIDDEN_CHANNELS=${HIDDEN_CHANNELS}"
echo "TRAIN_UNTIL_YEAR=${TRAIN_UNTIL_YEAR}"
echo "VAL_YEAR=${VAL_YEAR}"
echo "TEST_FROM_YEAR=${TEST_FROM_YEAR}"
echo "TAG=${TAG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"

PYTHONUNBUFFERED=1 python3 -u pokec_raw_linkx_gsmp.py \
  --root "${POKEC_ROOT}" \
  --device cuda:0 \
  --epochs "${EPOCHS}" \
  --log-every "${LOG_EVERY}" \
  --train-until-year "${TRAIN_UNTIL_YEAR}" \
  --val-year "${VAL_YEAR}" \
  --test-from-year "${TEST_FROM_YEAR}" \
  --hidden-channels "${HIDDEN_CHANNELS}" \
  --num-layers 2 \
  --num-edge-layers 1 \
  --num-node-layers 1 \
  --dropout 0.5 \
  --lr 0.01 \
  --weight-decay 0 \
  --result-dir results \
  --tag "${TAG}"
