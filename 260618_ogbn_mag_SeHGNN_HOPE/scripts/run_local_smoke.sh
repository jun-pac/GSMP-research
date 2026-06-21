#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export METHODS="${METHODS:-baseline gsmp}"
export SEEDS="${SEEDS:-1}"
export STAGES="${STAGES:-2}"
export PATIENCE="${PATIENCE:-2}"
export RUN_NAME="${RUN_NAME:-local_smoke}"
export SKIP_ENV_ACTIVATE="${SKIP_ENV_ACTIVATE:-1}"

"${PROJECT_DIR}/scripts/prepare_data_links.sh"
for task in 0 1; do
    SLURM_ARRAY_TASK_ID="${task}" "${PROJECT_DIR}/slurm/run_one_common.sh"
done
