#!/bin/bash
#SBATCH --job-name=hope-grid-noes3
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=52G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-215%12
#SBATCH --output=/users/PAS1289/jyp531/GSMP-research/260621_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3_no_stage3_earlystop/logs/grid_no_stage3_earlystop_%A_%a.out
#SBATCH --error=/users/PAS1289/jyp531/GSMP-research/260621_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3_no_stage3_earlystop/logs/grid_no_stage3_earlystop_%A_%a.err

set -euo pipefail

# Fallback grid only. Do not submit unless the first no-stage3-earlystop run is not good enough.
# Grid size: 3 priority settings x 4 seeds x 3 lr x 3 dropout x 3 threshold x 2 lambda = 216 tasks.

PRIORITIES=(p1 p2 p3)
SEEDS=(0 1 2 3)
LRS=(0.001 0.0007 0.0005)
DROPOUTS=(0.45 0.50 0.60)
THRESHOLDS=(0.65 0.75 0.85)
LAMBS=(0.3 0.5)

task=${SLURM_ARRAY_TASK_ID:-0}

lamb_idx=$((task % ${#LAMBS[@]}))
task=$((task / ${#LAMBS[@]}))
threshold_idx=$((task % ${#THRESHOLDS[@]}))
task=$((task / ${#THRESHOLDS[@]}))
dropout_idx=$((task % ${#DROPOUTS[@]}))
task=$((task / ${#DROPOUTS[@]}))
lr_idx=$((task % ${#LRS[@]}))
task=$((task / ${#LRS[@]}))
seed_idx=$((task % ${#SEEDS[@]}))
task=$((task / ${#SEEDS[@]}))
priority_idx=$((task % ${#PRIORITIES[@]}))

PRIORITY=${PRIORITIES[$priority_idx]}
SEED=${SEEDS[$seed_idx]}
LR=${LRS[$lr_idx]}
DROPOUT=${DROPOUTS[$dropout_idx]}
THRESHOLD=${THRESHOLDS[$threshold_idx]}
LAMB=${LAMBS[$lamb_idx]}

case "${PRIORITY}" in
    p1)
        PRIORITY_NAME="p1_stage0_all_hops"
        export IMPACT_APPLY_TO="${IMPACT_APPLY_TO:-both}"
        export IMPACT_STAGES="${IMPACT_STAGES:-0}"
        export GSMP_LAYER_MODE="${GSMP_LAYER_MODE:-all}"
        ;;
    p2)
        PRIORITY_NAME="p2_stage0to3_all_hops"
        export IMPACT_APPLY_TO="${IMPACT_APPLY_TO:-both}"
        export IMPACT_STAGES="${IMPACT_STAGES:-0-3}"
        export GSMP_LAYER_MODE="${GSMP_LAYER_MODE:-all}"
        ;;
    p3)
        PRIORITY_NAME="p3_stage0_first_hop"
        export IMPACT_APPLY_TO="${IMPACT_APPLY_TO:-both}"
        export IMPACT_STAGES="${IMPACT_STAGES:-0}"
        export GSMP_LAYER_MODE="${GSMP_LAYER_MODE:-first}"
        ;;
    *)
        echo "Unknown priority '${PRIORITY}'" >&2
        exit 2
        ;;
esac

export RUN_NAME="${RUN_NAME:-grid_${PRIORITY_NAME}_lr${LR}_drop${DROPOUT}_thr${THRESHOLD}_lamb${LAMB}_no_stage3_earlystop}"
export JOB_SPEC="gsmp:${SEED}"
export STAGES="${STAGES:-300 300 300 300}"
export EVAL_EVERY="${EVAL_EVERY:-1}"
export BATCH_SIZE="${BATCH_SIZE:-10000}"
export PATIENCE="${PATIENCE:-100}"
export GSMP_NO_EARLY_STOP_STAGES="${GSMP_NO_EARLY_STOP_STAGES:-3}"
export KEEP_STAGE_CHECKPOINTS="${KEEP_STAGE_CHECKPOINTS:-1}"
export CONDA_ENV="${CONDA_ENV:-hope_official}"
export CACHE_DIR="${CACHE_DIR:-/users/PAS1289/jyp531/GSMP-research/260621_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3_no_stage3_earlystop/impact_cache_official_env}"
export EXTRA_ARGS="--lr ${LR} --dropout ${DROPOUT} --threshold ${THRESHOLD} --lamb ${LAMB} --gc-every 10 --no-empty-cache-every-epoch ${EXTRA_ARGS:-}"

source "/users/PAS1289/jyp531/GSMP-research/260621_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3_no_stage3_earlystop/slurm/run_one_common.sh"
