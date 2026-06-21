#!/bin/bash
#SBATCH --job-name=pokec-linkx-smp
#SBATCH --account=PAS1289
#SBATCH --output=logs/pokec_linkx_smp_%j.out
#SBATCH --error=logs/pokec_linkx_smp_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results data/pokec

module load python/3.10
source .venv/bin/activate

POKEC_ROOT="${POKEC_ROOT:-data/pokec}"
EPOCHS="${EPOCHS:-500}"
TIME_FEATURE_COL="${TIME_FEATURE_COL:-}"

echo "Running on $(hostname)"
echo "Python=$(command -v python3)"
echo "POKEC_ROOT=${POKEC_ROOT}"
echo "EPOCHS=${EPOCHS}"
echo "TIME_FEATURE_COL=${TIME_FEATURE_COL}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"

if [[ -z "${TIME_FEATURE_COL}" ]]; then
  echo "ERROR: Set TIME_FEATURE_COL to the Pokec feature column used as SMP time." >&2
  echo "First run: python3 pokec_linkx_smp.py --root ${POKEC_ROOT} --inspect --epochs 1" >&2
  exit 2
fi

python3 pokec_linkx_smp.py \
  --root "${POKEC_ROOT}" \
  --use-fixed-split \
  --split-id 0 \
  --time-feature-col "${TIME_FEATURE_COL}" \
  --device cuda:0 \
  --epochs "${EPOCHS}" \
  --hidden-channels 256 \
  --num-layers 2 \
  --num-edge-layers 1 \
  --num-node-layers 1 \
  --dropout 0.5 \
  --lr 0.01 \
  --weight-decay 0 \
  --result-dir results \
  --tag pokec_linkx_vs_linkx_smp_split0
