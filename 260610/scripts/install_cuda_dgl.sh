#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-${REPO_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if command -v module >/dev/null 2>&1; then
  module load cuda/12.4.1 2>/dev/null || true
fi

echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" - <<'PY'
import sys, torch
print("python", sys.version.split()[0])
print("torch", torch.__version__, "torch_cuda", torch.version.cuda)
if torch.version.cuda != "12.4":
    raise SystemExit("Expected torch CUDA 12.4; refusing to install DGL cu124 into this environment.")
PY

"${PYTHON_BIN}" -m pip install --upgrade \
  "dgl==2.5.0+cu124" \
  -f https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html

echo
echo "Installed DGL. Run this on a GPU node to verify CUDA graph transfer:"
cat <<'EOF'
python - <<'PY'
import torch, dgl
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("dgl", dgl.__version__, dgl.__file__)
g = dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=1)
g = g.to("cuda")
print("dgl graph device", g.device)
PY
EOF
