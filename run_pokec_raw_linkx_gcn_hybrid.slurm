#!/bin/bash
#SBATCH --job-name=pokec-linkx-gcn
#SBATCH --account=PAS1289
#SBATCH --output=logs/pokec_raw_linkx_gcn_hybrid_%j.out
#SBATCH --error=logs/pokec_raw_linkx_gcn_hybrid_%j.err
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
MODE="${MODE:-gsmp}"
EPOCHS="${EPOCHS:-200}"
LOG_EVERY="${LOG_EVERY:-10}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-64}"
LINKX_NUM_LAYERS="${LINKX_NUM_LAYERS:-2}"
GCN_NUM_LAYERS="${GCN_NUM_LAYERS:-2}"
TRAIN_UNTIL_YEAR="${TRAIN_UNTIL_YEAR:-2009}"
VAL_YEAR="${VAL_YEAR:-2010}"
TEST_FROM_YEAR="${TEST_FROM_YEAR:-2011}"
TAG="${TAG:-pokec_raw_linkx_gcn_${MODE}_train${TRAIN_UNTIL_YEAR}_val${VAL_YEAR}_test${TEST_FROM_YEAR}}"

echo "Running on $(hostname)"
echo "Python=$(command -v python3)"
echo "POKEC_ROOT=${POKEC_ROOT}"
echo "MODE=${MODE}"
echo "EPOCHS=${EPOCHS}"
echo "LOG_EVERY=${LOG_EVERY}"
echo "HIDDEN_CHANNELS=${HIDDEN_CHANNELS}"
echo "LINKX_NUM_LAYERS=${LINKX_NUM_LAYERS}"
echo "GCN_NUM_LAYERS=${GCN_NUM_LAYERS}"
echo "TRAIN_UNTIL_YEAR=${TRAIN_UNTIL_YEAR}"
echo "VAL_YEAR=${VAL_YEAR}"
echo "TEST_FROM_YEAR=${TEST_FROM_YEAR}"
echo "TAG=${TAG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"

PYTHONUNBUFFERED=1 python3 -u pokec_raw_linkx_gcn_hybrid.py \
  --root "${POKEC_ROOT}" \
  --mode "${MODE}" \
  --device cuda:0 \
  --epochs "${EPOCHS}" \
  --log-every "${LOG_EVERY}" \
  --train-until-year "${TRAIN_UNTIL_YEAR}" \
  --val-year "${VAL_YEAR}" \
  --test-from-year "${TEST_FROM_YEAR}" \
  --hidden-channels "${HIDDEN_CHANNELS}" \
  --linkx-num-layers "${LINKX_NUM_LAYERS}" \
  --linkx-num-edge-layers 1 \
  --linkx-num-node-layers 1 \
  --gcn-num-layers "${GCN_NUM_LAYERS}" \
  --dropout 0.5 \
  --lr 0.01 \
  --weight-decay 5e-4 \
  --result-dir results \
  --tag "${TAG}"
