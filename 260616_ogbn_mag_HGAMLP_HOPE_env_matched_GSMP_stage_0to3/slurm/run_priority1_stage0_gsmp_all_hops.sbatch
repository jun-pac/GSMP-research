#!/bin/bash
#SBATCH --job-name=hope-mag-v2-p1
#SBATCH --account=PAS1289
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=52G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/users/PAS1289/jyp531/GSMP-research/260616_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3/logs/priority1_%A_%a.out
#SBATCH --error=/users/PAS1289/jyp531/GSMP-research/260616_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3/logs/priority1_%A_%a.err

set -euo pipefail

# Priority 1:
#   GSMP is active only in stage 0.
#   GSMP is applied to all eligible direct paper-paper propagation hops:
#     P->PP, PP->PPP, PA->PPA, PF->PPF.
export RUN_NAME="${RUN_NAME:-envmatched_p1_gsmp_stage0_all_hops}"
export JOB_SPECS="${JOB_SPECS:-gsmp:1 gsmp:2 gsmp:3}"
export STAGES="${STAGES:-300 300 300 300}"
export EVAL_EVERY="${EVAL_EVERY:-1}"
export BATCH_SIZE="${BATCH_SIZE:-10000}"
export PATIENCE="${PATIENCE:-100}"
export IMPACT_APPLY_TO="${IMPACT_APPLY_TO:-both}"
export IMPACT_STAGES="${IMPACT_STAGES:-0}"
export GSMP_LAYER_MODE="${GSMP_LAYER_MODE:-all}"
export CONDA_ENV="${CONDA_ENV:-hope_official}"
export CACHE_DIR="${CACHE_DIR:-/users/PAS1289/jyp531/GSMP-research/260616_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3/impact_cache_official_env}"

if [[ -z "${EXTRA_ARGS:-}" ]]; then
    export EXTRA_ARGS="--gc-every 10 --no-empty-cache-every-epoch"
fi

source "/users/PAS1289/jyp531/GSMP-research/260616_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3/slurm/run_one_common.sh"
