#!/bin/bash
#
# Submit with:
#     sbatch run_ogbn_mag_hgamLP_hope_gcn_temporal_all_in_one.sbatch
#
# Monitor with:
#     squeue -u $USER
#
# Cancel with:
#     scancel JOB_ID

#SBATCH --job-name=mag_hgamLP_gcn_tmp
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs
mkdir -p results/ogbn_mag_hgamLP_hope_gcn_temporal
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

###############################################################################
# Environment setup
###############################################################################

# Option A: existing project virtualenv
source /users/PAS1289/jyp531/GSMP-research/.venv/bin/activate

# Option B: conda through the cluster module system, commented out
# module load miniconda3/24.1.2-py310
# conda activate YOUR_ENV_NAME

# Option C: module system, commented out
# module load cuda/12.4.1
# module load python/3.10

###############################################################################
# Debugging info
###############################################################################

echo "hostname: $(hostname)"
echo "date: $(date)"
nvidia-smi || true
which python
python --version
echo "job id: ${SLURM_JOB_ID}"

###############################################################################
# Run all experiments sequentially
###############################################################################

METHODS=(gcn gcn_smp gcn_ump gcn_gsmp)
SEEDS=(0 1 2 3 4)
EPOCHS=${EPOCHS:-200}
EVAL_STEPS=${EVAL_STEPS:-10}
HIDDEN_DIM=${HIDDEN_DIM:-512}
NUM_LAYERS=${NUM_LAYERS:-3}

for PROP in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        case "${PROP}" in
            gcn) METHOD=baseline ;;
            gcn_smp) METHOD=smp ;;
            gcn_ump) METHOD=ump ;;
            gcn_gsmp) METHOD=gsmp ;;
            *) echo "Unknown propagation method: ${PROP}" >&2; exit 1 ;;
        esac

        echo "============================================================"
        echo "method: ${PROP}"
        echo "code method: ${METHOD}"
        echo "seed: ${SEED}"
        echo "epochs: ${EPOCHS}"
        echo "eval steps: ${EVAL_STEPS}"
        echo "hidden dim: ${HIDDEN_DIM}"
        echo "num layers: ${NUM_LAYERS}"
        echo "job id: ${SLURM_JOB_ID}"
        echo "start date: $(date)"
        echo "============================================================"

        python -u main.py \
            --dataset ogbn-mag \
            --model hgamLP_hope \
            --method ${METHOD} \
            --epochs ${EPOCHS} \
            --eval-steps ${EVAL_STEPS} \
            --hidden-dim ${HIDDEN_DIM} \
            --num-layers ${NUM_LAYERS} \
            --dropout 0.5 \
            --lr 0.001 \
            --weight-decay 0.0 \
            --seed ${SEED} \
            --device cuda:0 \
            --root ./data \
            --save-dir ./results/ogbn_mag_hgamLP_hope_gcn_temporal

        echo "end date: $(date)"
    done
done
