#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-${ROOT}/downloads/official_tape_assets}"
URL="${TAPE_ASSETS_URL:-https://drive.google.com/drive/folders/1nF8NDGObIqU0kCkzVaisWooGEQlcNSIN?usp=sharing}"

if [[ "${CONFIRM_TAPE_DOWNLOAD:-0}" != "1" ]]; then
  cat >&2 <<EOF
This downloads the official TAPE Google Drive asset folder, which may be large.
Set CONFIRM_TAPE_DOWNLOAD=1 to continue.

Destination:
  ${DEST}

Command:
  CONFIRM_TAPE_DOWNLOAD=1 bash scripts/download_official_tape_assets.sh
EOF
  exit 2
fi

if [[ -f "${ROOT}/../.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/../.venv/bin/activate"
fi

command -v gdown >/dev/null 2>&1 || {
  echo "gdown not found. Install with: pip install gdown" >&2
  exit 1
}

mkdir -p "${DEST}"
gdown --folder "${URL}" -O "${DEST}/" --continue

cat <<EOF
Download finished.

Now locate the .emb files and copy or rsync them into:
  ${ROOT}/upstream/TAPE/prt_lm/ogbn-arxiv/microsoft/
  ${ROOT}/upstream/TAPE/prt_lm/ogbn-arxiv2/microsoft/

Then run:
  bash scripts/check_tape_features.sh
EOF
