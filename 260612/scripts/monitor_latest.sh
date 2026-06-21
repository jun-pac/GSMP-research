#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$EXP_ROOT/results/tunedgcn_gsmp}"
INTERVAL="${INTERVAL:-20}"

while true; do
  clear
  date
  echo
  squeue -u "${USER:-$(whoami)}" || true
  echo

  RUN_DIR="$(ls -td "$RESULT_ROOT"/* 2>/dev/null | head -1 || true)"
  if [[ -z "$RUN_DIR" ]]; then
    echo "No result directories found under $RESULT_ROOT"
  else
    python - "$RUN_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
print(f"run: {run_dir}")

final_path = run_dir / "final_summary.json"
if final_path.exists():
    summary = json.loads(final_path.read_text())
    print("status: final_summary.json exists")
    print(f"method: {summary.get('method')} gsmp_apply: {summary.get('gsmp_apply', 'all_layers')}")
    print(f"best val: {100 * float(summary['val_mean']):.2f}%")
    print(f"test @ best val: {100 * float(summary['test_at_best_val_mean']):.2f}%")
    print(f"best raw test: {100 * float(summary['best_raw_test_mean']):.2f}%")
    print(f"best epoch: {summary['best_epoch_mean']}")
    print(f"runtime: {float(summary['runtime_mean']) / 60:.1f} min")
    print(f"peak gpu: {float(summary['gpu_memory_peak_mean']):.1f} MB")
    raise SystemExit

csv_path = run_dir / "epoch_logs.csv"
if not csv_path.exists():
    print("status: no epoch_logs.csv yet")
    raise SystemExit

rows = list(csv.DictReader(csv_path.open()))
if not rows:
    print("status: epoch_logs.csv has no rows yet")
    raise SystemExit

r = rows[-1]
print("status: running or incomplete")
print(f"method: {r.get('method')} gsmp_apply: {r.get('gsmp_apply', 'all_layers')}")
print(f"epoch: {r['epoch']} / 2000")
print(
    "latest train/val/test: "
    f"{100 * float(r['train_acc']):.2f}% / "
    f"{100 * float(r['val_acc']):.2f}% / "
    f"{100 * float(r['test_acc']):.2f}%"
)
print(f"best val: {100 * float(r['best_val']):.2f}% at epoch {r['best_epoch']}")
print(f"test @ best val: {100 * float(r['test_at_best_val']):.2f}%")
print(f"best raw test: {100 * float(r['best_raw_test']):.2f}% at epoch {r['best_raw_test_epoch']}")
print(f"elapsed: {float(r['total_time']) / 60:.1f} min")
print(f"peak gpu: {float(r['gpu_memory_peak_mb']):.1f} MB")
PY
  fi

  echo
  echo "Ctrl+C to stop. Refreshing every ${INTERVAL}s."
  sleep "$INTERVAL"
done
