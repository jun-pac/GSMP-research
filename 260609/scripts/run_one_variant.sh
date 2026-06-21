#!/bin/bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 DATASET VARIANT SEED EPOCHS BERT_X_DIR RUN_NAME" >&2
  exit 2
fi

DATASET="$1"
VARIANT="$2"
SEED="$3"
EPOCHS="$4"
BERT_X_DIR="$5"
RUN_NAME="$6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

USE_GPT_PREDS="${USE_GPT_PREDS:-0}"

if [[ "${USE_GPT_PREDS}" != "1" && ! -f "${BERT_X_DIR}" ]]; then
  if [[ -f "${PROJECT_DIR}/${BERT_X_DIR}" ]]; then
    BERT_X_DIR="${PROJECT_DIR}/${BERT_X_DIR}"
  elif [[ -f "${REPO_DIR}/${BERT_X_DIR}" ]]; then
    BERT_X_DIR="${REPO_DIR}/${BERT_X_DIR}"
  else
    echo "ERROR: cached embeddings not found: ${BERT_X_DIR}" >&2
    echo "Put x_embs.pt under SimTeG/out/... or run scripts/download_embeddings.sh explicitly." >&2
    exit 1
  fi
fi

VENV_DIR="${REPO_DIR}/.venv"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
  echo "Activated venv: ${VENV_DIR}"
else
  if command -v module >/dev/null 2>&1; then
    module load miniconda3/24.1.2-py310 2>/dev/null || true
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: neither ${VENV_DIR} nor conda is available." >&2
    exit 1
  fi
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
  fi
  ENV_NAME="${CONDA_ENV_NAME:-graphenv}"
  conda activate "${ENV_NAME}"
  echo "Activated conda env: ${ENV_NAME}"
fi

mkdir -p "results/${RUN_NAME}"

CMD=(
  python -u train_ogbn_arxiv_smpumpgsmp.py
  --dataset "${DATASET}"
  --mp_variant "${VARIANT}"
  --seeds "${SEED}"
  --max_epochs "${EPOCHS}"
  --run_name "${RUN_NAME}"
  --output_root results
  --data_root ../data
  --device "${DEVICE:-cuda:0}"
  --num_workers "${NUM_WORKERS:-0}"
  --num_neighbors "${NUM_NEIGHBORS:-15,10,5,5}"
  --gnn_batch_size "${GNN_BATCH_SIZE:-10000}"
  --gnn_eval_batch_size "${GNN_EVAL_BATCH_SIZE:-10000}"
  --gnn_epochs "${GNN_EPOCHS:-100}"
  --gnn_dropout "${GNN_DROPOUT:-0.4}"
  --gnn_label_smoothing "${GNN_LABEL_SMOOTHING:-0.4}"
  --gnn_lr "${GNN_LR:-0.01}"
  --gnn_num_layers "${GNN_NUM_LAYERS:-2}"
  --gnn_weight_decay "${GNN_WEIGHT_DECAY:-4e-6}"
  --gnn_eval_interval "${GNN_EVAL_INTERVAL:-1}"
  --aggregation_chunk_size "${AGGREGATION_CHUNK_SIZE:-200000}"
  --eval_mode "${EVAL_MODE:-mini}"
)

if [[ "${USE_GPT_PREDS}" == "1" ]]; then
  CMD+=(
    --use_gpt_preds
    --gpt_preds_path "${GPT_PREDS_PATH:-resources/ogbn-arxiv-gpt-preds.csv}"
    --embedding_name "${EMBEDDING_NAME:-gpt-preds}"
  )
else
  CMD+=(
    --use_bert_x
    --bert_x_dir "${BERT_X_DIR}"
    --embedding_name "${EMBEDDING_NAME:-e5-large}"
  )
fi

if [[ "${RUN_UNIT_TEST:-0}" == "1" ]]; then
  CMD+=(--run_unit_test)
fi

echo "COMMAND ${CMD[*]}"
"${CMD[@]}"
