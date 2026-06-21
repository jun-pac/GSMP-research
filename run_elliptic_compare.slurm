#!/bin/bash
#SBATCH --job-name=elliptic-compare
#SBATCH --account=PAS1289
#SBATCH --output=logs/elliptic_compare_%j.out
#SBATCH --error=logs/elliptic_compare_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results

ELLIPTIC_ROOT="${ELLIPTIC_ROOT:-/path/to/elliptic_bitcoin_dataset}"
EPOCHS="${EPOCHS:-200}"

module load python/3.10
source .venv/bin/activate

echo "Running on $(hostname)"
echo "Python=$(command -v python3)"
echo "ELLIPTIC_ROOT=${ELLIPTIC_ROOT}"
echo "EPOCHS=${EPOCHS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"

if [[ "${ELLIPTIC_ROOT}" == "/path/to/elliptic_bitcoin_dataset" ]]; then
  echo "ERROR: Set ELLIPTIC_ROOT to the directory containing Elliptic CSV files before submitting." >&2
  exit 2
fi

if [[ ! -f "${ELLIPTIC_ROOT}/elliptic_txs_features.csv" ]]; then
  echo "ERROR: Missing ${ELLIPTIC_ROOT}/elliptic_txs_features.csv" >&2
  exit 2
fi

python3 - <<'PY'
import importlib.util
missing = [pkg for pkg in ("pandas", "torch", "torch_geometric") if importlib.util.find_spec(pkg) is None]
if missing:
    raise SystemExit("ERROR: Missing Python packages: " + ", ".join(missing))
PY

echo "=== Vanilla GCN ==="
python3 elliptic_bitcoin_smp.py \
  --root "${ELLIPTIC_ROOT}" \
  --device cuda:0 \
  --hidden-channels 128 \
  --num-layers 2 \
  --dropout 0.5 \
  --lr 0.01 \
  --weight-decay 5e-4 \
  --epochs "${EPOCHS}" \
  --result-dir results \
  --tag gcn_undirected_L2_H128_lr0.01_wd5e-4

echo "=== GCN + SMP ==="
python3 elliptic_bitcoin_smp.py \
  --root "${ELLIPTIC_ROOT}" \
  --device cuda:0 \
  --use-smp \
  --hidden-channels 128 \
  --num-layers 2 \
  --dropout 0.5 \
  --lr 0.01 \
  --weight-decay 5e-4 \
  --epochs "${EPOCHS}" \
  --result-dir results \
  --tag gcn_smp_undirected_L2_H128_lr0.01_wd5e-4

echo "=== Result summary ==="
grep -H "best_\\|final_\\|num_edges\\|use_smp" \
  results/elliptic_bitcoin_gcn_undirected_L2_H128_lr0.01_wd5e-4.txt \
  results/elliptic_bitcoin_gcn_smp_undirected_L2_H128_lr0.01_wd5e-4.txt
