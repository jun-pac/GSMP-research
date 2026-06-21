#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LD_DIR="${ROOT}/upstream/LD"
TRANSFORMER_DIR="${LD_DIR}/transformer"

RUN_KIND="${RUN_KIND:-smoke}"
LINEAR="${LINEAR:-false}"
USE_GSMP="${USE_GSMP:-false}"
GSMP_NORM="${GSMP_NORM:-scale_preserve}"
FULL="${FULL:-0}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${RUN_KIND}_seed${SEED}}"

if [[ "${FULL}" == "1" ]]; then
  EPOCHS="${EPOCHS:-300}"
  EVAL_STEPS="${EVAL_STEPS:-2}"
  SAVE_MODEL="${SAVE_MODEL:-true}"
else
  EPOCHS="${EPOCHS:-3}"
  EVAL_STEPS="${EVAL_STEPS:-1}"
  SAVE_MODEL="${SAVE_MODEL:-false}"
fi

ROOT_DATA="${OGB_ROOT:-${ROOT}/data}"
is_placeholder_path() {
  [[ "$1" == /path/to/* || "$1" == PATH_TO_* ]]
}

if [[ -z "${LD_LM_PATH:-}" ]]; then
  LD_LM_PATH="${HOME}/model/deberta-base"
elif is_placeholder_path "${LD_LM_PATH}"; then
  LD_LM_PATH="${HOME}/model/deberta-base"
fi
DEFAULT_LOCAL_TOKEN_FOLDER="${ROOT}/downloads/ld_tokens/extracted/ogbn_arxiv/token/microsoft/deberta-base"
if [[ -n "${LD_TOKEN_FOLDER:-}" ]] && is_placeholder_path "${LD_TOKEN_FOLDER}"; then
  unset LD_TOKEN_FOLDER
fi
if [[ -z "${LD_TOKEN_FOLDER:-}" && -f "${DEFAULT_LOCAL_TOKEN_FOLDER}/input_ids.npy" ]]; then
  LD_TOKEN_FOLDER="${DEFAULT_LOCAL_TOKEN_FOLDER}"
else
  LD_TOKEN_FOLDER="${LD_TOKEN_FOLDER:-/OGB/ogbn_arxiv/token/microsoft/deberta-base}"
fi
LD_CKPT_DIR="${LD_CKPT_DIR:-${ROOT}/checkpoints}"
FINETUNE_PREFIX="${FINETUNE_PREFIX:-null}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
export PYTHON_RANDOM_SEED="${SEED}"
export LD_GSMP_ROOT="${ROOT}"
export LD_GSMP_RUN_ID="${RUN_ID}"
export LD_GSMP_RESULT_ROOT="${ROOT}/results/ld_revgat_gsmp"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

if [[ -n "${VENV_DIR:-}" && -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
elif [[ -f "${ROOT}/../.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/../.venv/bin/activate"
else
  source ~/.bashrc || true
  if command -v conda >/dev/null 2>&1; then
    conda activate "${LD_CONDA_ENV:-ld}" || true
  fi
fi

mkdir -p "${ROOT}/logs" "${ROOT}/results/ld_revgat_gsmp" "${ROOT}/cache/gsmp" "${ROOT}/outputs" "${LD_CKPT_DIR}"

cd "${TRANSFORMER_DIR}"

echo "============================================================"
echo "LD RevGAT GSMP run"
echo "============================================================"
echo "hostname=$(hostname)"
date
echo "pwd=$(pwd)"
git -C "${LD_DIR}" rev-parse HEAD || true
nvidia-smi || true
python --version
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY
echo "RUN_KIND=${RUN_KIND}"
echo "LINEAR=${LINEAR}"
echo "USE_GSMP=${USE_GSMP}"
echo "GSMP_NORM=${GSMP_NORM}"
echo "FULL=${FULL}"
echo "EPOCHS=${EPOCHS}"
echo "EVAL_STEPS=${EVAL_STEPS}"
echo "SAVE_MODEL=${SAVE_MODEL}"
echo "SEED=${SEED}"
echo "RUN_ID=${RUN_ID}"
echo "OGB_ROOT=${ROOT_DATA}"
echo "LD_LM_PATH=${LD_LM_PATH}"
echo "LD_TOKEN_FOLDER=${LD_TOKEN_FOLDER}"
echo "LD_CKPT_DIR=${LD_CKPT_DIR}"
echo "FINETUNE_PREFIX=${FINETUNE_PREFIX}"
echo "============================================================"

expected_hidden="${LD_LM_PATH%/}/arxiv_seed${SEED}/hidden_state.pt"
if [[ "${LD_REQUIRE_HIDDEN_STATE:-0}" == "1" && ! -f "${expected_hidden}" ]]; then
  echo "Refusing to run because LD_REQUIRE_HIDDEN_STATE=1 and missing ${expected_hidden}" >&2
  exit 3
fi

python main_bertgnn.py \
  model=revgat \
  dataset=arxiv \
  LM=deberta-base \
  phase=pre_gnn \
  root="${ROOT_DATA}" \
  LM.path="${LD_LM_PATH}" \
  LM.params.arxiv.token_folder="${LD_TOKEN_FOLDER}" \
  phase.params.arxiv.revgat.epochs="${EPOCHS}" \
  phase.params.arxiv.revgat.eval_steps="${EVAL_STEPS}" \
  phase.params.arxiv.revgat.save_model="${SAVE_MODEL}" \
  phase.params.arxiv.revgat.finetune_prefix="${FINETUNE_PREFIX}" \
  phase.params.arxiv.revgat.out_dir="${ROOT}/huggingface_logs/${RUN_ID}" \
  phase.params.arxiv.revgat.ckpt="${LD_CKPT_DIR}" \
  model.params.arxiv.architecture.linear="${LINEAR}" \
  model.params.arxiv.architecture.use_gsmp="${USE_GSMP}" \
  model.params.arxiv.architecture.gsmp_norm="${GSMP_NORM}" \
  model.params.arxiv.architecture.gsmp_cache_dir="${ROOT}/cache/gsmp" \
  model.params.arxiv.architecture.graph_direction=dgl_bidirected_self_loop \
  hydra.run.dir="${ROOT}/outputs/${RUN_ID}" \
  hydra.output_subdir=.hydra

echo "============================================================"
echo "Finished ${RUN_ID}"
echo "Results: ${ROOT}/results/ld_revgat_gsmp/${RUN_ID}"
echo "============================================================"
