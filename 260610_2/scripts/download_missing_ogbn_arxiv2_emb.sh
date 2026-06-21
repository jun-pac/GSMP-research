#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/upstream/TAPE/prt_lm/ogbn-arxiv2/microsoft"
mkdir -p "${DEST}"

if [[ -f "${ROOT}/../.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/../.venv/bin/activate"
fi

command -v gdown >/dev/null 2>&1 || {
  echo "gdown not found. Install with: pip install gdown" >&2
  exit 1
}

download_one() {
  local file_id="$1"
  local name="$2"
  local path="${DEST}/${name}"
  if [[ -s "${path}" ]]; then
    echo "OK existing ${path}"
    return 0
  fi
  echo "Downloading ${name}"
  gdown --continue "https://drive.google.com/uc?id=${file_id}" -O "${path}"
}

download_one "1jnKdDYT_F9Fx9w29h1uD6aeDm_JK0kDi" "deberta-base-seed0.emb"
download_one "16FIlpl2uOhwvf2cvSWJIWBLftAs-SVtw" "deberta-base-seed1.emb"
download_one "1DFaBWGCCp99fEODvFx8XHv1mJKlAEtHl" "deberta-base-seed2.emb"

bash "${ROOT}/scripts/check_tape_features.sh"
