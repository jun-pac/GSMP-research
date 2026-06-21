#!/bin/bash
#SBATCH --job-name=hope-p2-noes3
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=52G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/users/PAS1289/jyp531/GSMP-research/260621_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3_no_stage3_earlystop/logs/priority2_no_stage3_earlystop_%A_%a.out
#SBATCH --error=/users/PAS1289/jyp531/GSMP-research/260621_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3_no_stage3_earlystop/logs/priority2_no_stage3_earlystop_%A_%a.err

set -euo pipefail

# Priority 2:
#   GSMP is active in stages 0,1,2,3.
#   GSMP is applied to all eligible direct paper-paper propagation hops:
#     P->PP, PP->PPP, PA->PPA, PF->PPF.
#   Early stopping is disabled only for GSMP stage 3.
export RUN_NAME="${RUN_NAME:-envmatched_p2_gsmp_stage0to3_all_hops_no_stage3_earlystop}"
export JOB_SPECS="${JOB_SPECS:-gsmp:0 gsmp:1 gsmp:2 gsmp:3}"
export STAGES="${STAGES:-300 300 300 300}"
export EVAL_EVERY="${EVAL_EVERY:-1}"
export BATCH_SIZE="${BATCH_SIZE:-10000}"
export PATIENCE="${PATIENCE:-100}"
export IMPACT_APPLY_TO="${IMPACT_APPLY_TO:-both}"
export IMPACT_STAGES="${IMPACT_STAGES:-0-3}"
export GSMP_LAYER_MODE="${GSMP_LAYER_MODE:-all}"
export GSMP_NO_EARLY_STOP_STAGES="${GSMP_NO_EARLY_STOP_STAGES:-3}"
export KEEP_STAGE_CHECKPOINTS="${KEEP_STAGE_CHECKPOINTS:-1}"
export CONDA_ENV="${CONDA_ENV:-hope_official}"
export CACHE_DIR="${CACHE_DIR:-/users/PAS1289/jyp531/GSMP-research/260621_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3_no_stage3_earlystop/impact_cache_official_env}"

if [[ -z "${EXTRA_ARGS:-}" ]]; then
    export EXTRA_ARGS="--gc-every 10 --no-empty-cache-every-epoch"
fi

source "/users/PAS1289/jyp531/GSMP-research/260621_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3_no_stage3_earlystop/slurm/run_one_common.sh"
