#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

VENV_DIR="${REPO_DIR}/.venv"
if [[ -x "${VENV_DIR}/bin/hf" ]]; then
  HF="${VENV_DIR}/bin/hf"
else
  HF="$(command -v hf || true)"
fi

if [[ -z "${HF}" ]]; then
  echo "ERROR: Hugging Face CLI 'hf' not found. Install huggingface_hub or use ${VENV_DIR}." >&2
  exit 1
fi

LOCAL_DIR="${LOCAL_DIR:-${REPO_DIR}/SimTeG/out}"
ALL_COMPONENTS="${ALL_COMPONENTS:-1}"
DATASET="${DATASET:-ogbn-arxiv}"
EMBEDDING_NAME="${EMBEDDING_NAME:-e5-large}"

if [[ "${ALL_COMPONENTS}" == "1" ]]; then
  INCLUDE_PATHS=(
    "ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt"
    "ogbn-arxiv/all-roberta-large-v1/main/cached_embs/x_embs.pt"
    "ogbn-arxiv-tape/e5-large/main/cached_embs/x_embs.pt"
    "ogbn-arxiv-tape/all-roberta-large-v1/main/cached_embs/x_embs.pt"
  )
else
  INCLUDE_PATHS=("${DATASET}/${EMBEDDING_NAME}/main/cached_embs/x_embs.pt")
fi

for path in "${INCLUDE_PATHS[@]}"; do
  if [[ -f "${LOCAL_DIR}/${path}" ]]; then
    echo "Already present: ${LOCAL_DIR}/${path}"
    continue
  fi
  echo "Downloading ${path} into ${LOCAL_DIR}"
  "${HF}" download vermouthdky/SimTeG \
    --repo-type dataset \
    --include "${path}" \
    --local-dir "${LOCAL_DIR}"
done
