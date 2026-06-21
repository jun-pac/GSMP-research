#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAPE_DIR="${ROOT}/upstream/TAPE"

MODEL="${MODEL:-RevGAT}"
USE_GSMP="${USE_GSMP:-False}"
GSMP_NORM="${GSMP_NORM:-scale_preserve}"
FULL="${FULL:-0}"
RUN_KIND="${RUN_KIND:-smoke}"
GPU="${GPU:-0}"
LR="${LR:-0.002}"
DROPOUT="${DROPOUT:-0.75}"
LOG_EVERY="${LOG_EVERY:-10}"
EVAL_EVERY="${EVAL_EVERY:-10}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${RUN_KIND}_${MODEL}}"

if [[ "${FULL}" == "1" ]]; then
  FEATURE_TYPE="${FEATURE_TYPE:-TA_P_E}"
  EPOCHS="${EPOCHS:-200}"
  RUNS="${RUNS:-3}"
  SEED_ARGS=(runs "${RUNS}")
else
  FEATURE_TYPE="${FEATURE_TYPE:-ogb}"
  EPOCHS="${EPOCHS:-5}"
  SEED="${SEED:-0}"
  SEED_ARGS=(seed "${SEED}" runs 1)
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TAPE_GSMP_RUN_ID="${RUN_ID}"
export OGB_DATA_ROOT="${OGB_DATA_ROOT:-${ROOT}/../data}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

if [[ -n "${VENV_DIR:-}" && -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
elif [[ -f "${ROOT}/../.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/../.venv/bin/activate"
fi

cd "${TAPE_DIR}"
mkdir -p "${ROOT}/logs" "${ROOT}/results/tape_revgat_gsmp" "${ROOT}/cache/gsmp" output

echo "============================================================"
echo "TAPE GSMP run"
echo "============================================================"
echo "hostname=$(hostname)"
date
nvidia-smi || true
git rev-parse HEAD || true
python --version
python - <<'PY'
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda)
except Exception as exc:
    print("torch import failed", repr(exc))
    raise
PY
echo "MODEL=${MODEL}"
echo "USE_GSMP=${USE_GSMP}"
echo "GSMP_NORM=${GSMP_NORM}"
echo "FEATURE_TYPE=${FEATURE_TYPE}"
echo "FULL=${FULL}"
echo "EPOCHS=${EPOCHS}"
echo "RUN_ID=${RUN_ID}"
echo "OGB_DATA_ROOT=${OGB_DATA_ROOT}"
echo "============================================================"

python -m core.trainEnsemble \
  dataset ogbn-arxiv \
  device "${GPU}" \
  gnn.model.name "${MODEL}" \
  gnn.model.use_gsmp "${USE_GSMP}" \
  gnn.model.gsmp_norm "${GSMP_NORM}" \
  gnn.model.gsmp_cache_dir "${ROOT}/cache/gsmp" \
  gnn.model.graph_direction dgl_bidirected_self_loop \
  gnn.train.feature_type "${FEATURE_TYPE}" \
  gnn.train.lr "${LR}" \
  gnn.train.dropout "${DROPOUT}" \
  gnn.train.epochs "${EPOCHS}" \
  gnn.train.log_every "${LOG_EVERY}" \
  gnn.train.eval_every "${EVAL_EVERY}" \
  gnn.train.result_root "${ROOT}/results/tape_revgat_gsmp" \
  gnn.train.run_id "${RUN_ID}" \
  "${SEED_ARGS[@]}"

echo "============================================================"
echo "Finished ${RUN_ID}"
echo "Results: ${ROOT}/results/tape_revgat_gsmp/${RUN_ID}"
echo "============================================================"
