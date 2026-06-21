#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-$WORKDIR/results/glem_revgat_gsmp}"
LOG_DIR="${LOG_DIR:-$WORKDIR/logs}"
TAIL_LINES="${TAIL_LINES:-80}"
FOLLOW_LOG="${FOLLOW_LOG:-1}"
WATCH_SECONDS="${WATCH_SECONDS:-0}"

latest_log() {
  find "$LOG_DIR" -type f -name '*.out' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR==1 {print $2}'
}

show_snapshot() {
  local latest="${1:-}"
  echo "== GSMP/GLEM Monitor =="
  date
  echo

  echo "== Slurm Queue =="
  if command -v squeue >/dev/null 2>&1; then
    squeue -u "$USER" -o "%.18i %.9P %.28j %.8u %.2t %.10M %.6D %R" || true
  else
    echo "squeue not available"
  fi
  echo

  python - "$RESULTS_ROOT" <<'PY'
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])

def fnum(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default

def fmt_acc(value):
    if value is None:
        return "n/a"
    return f"{fnum(value):.4f}"

def fmt_mean(values):
    vals = [fnum(v) for v in values if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f}+/-{statistics.pstdev(vals):.4f}"

def method_name(row):
    name = row.get("method", "unknown")
    if str(row.get("use_gsmp", "")).lower() in {"true", "1", "t", "yes"} and "GSMP" not in name:
        name = f"{name}+GSMP"
    norm = row.get("gsmp_norm")
    if norm and norm != "none":
        name = f"{name}({norm})"
    return name

final_rows = []
for path in sorted(root.glob("*/final_summary.json")):
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    data["_run"] = path.parent.name
    data["_mtime"] = path.stat().st_mtime
    final_rows.append(data)

print("== Completed Accuracy Summary ==")
if final_rows:
    grouped = defaultdict(list)
    for row in final_rows:
        grouped[method_name(row)].append(row)
    print(f"{'method':38} {'n':>2} {'seeds':10} {'val':>15} {'test@best':>15} {'best_epoch':>10} {'gpu_GB':>8} {'runtime_s':>9}")
    for method in sorted(grouped):
        rows = grouped[method]
        seeds = ",".join(str(r.get("seed", "?")) for r in sorted(rows, key=lambda r: str(r.get("seed", ""))))
        best_epochs = [fnum(r.get("best_epoch")) for r in rows]
        gpu = [fnum(r.get("gpu_memory_peak_gb")) for r in rows]
        runtime = [fnum(r.get("total_time")) for r in rows]
        print(
            f"{method[:38]:38} {len(rows):2d} {seeds[:10]:10} "
            f"{fmt_mean(r.get('best_val') for r in rows):>15} "
            f"{fmt_mean(r.get('test_at_best_val') for r in rows):>15} "
            f"{fmt_mean(best_epochs):>10} "
            f"{(max(gpu) if gpu else 0.0):8.3f} "
            f"{(sum(runtime) / len(runtime) if runtime else 0.0):9.1f}"
        )
else:
    print("No final_summary.json files found yet.")
print()

epoch_rows = []
for path in root.glob("*/epoch_logs.csv"):
    try:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        continue
    if not rows:
        continue
    row = rows[-1]
    row["_run"] = path.parent.name
    row["_mtime"] = path.stat().st_mtime
    epoch_rows.append(row)

print("== Latest Epoch Progress ==")
if epoch_rows:
    epoch_rows.sort(key=lambda r: r["_mtime"], reverse=True)
    print(f"{'run':42} {'method':28} {'seed':>4} {'ep':>5} {'val':>8} {'test':>8} {'best_val':>8} {'test@best':>9} {'gpu_GB':>8} {'age':>7}")
    now = time.time()
    for row in epoch_rows[:12]:
        age = max(0, int(now - row["_mtime"]))
        print(
            f"{row['_run'][:42]:42} {row.get('method','unknown')[:28]:28} "
            f"{row.get('seed','?'):>4} {row.get('epoch','?'):>5} "
            f"{fmt_acc(row.get('val_acc')):>8} {fmt_acc(row.get('test_acc')):>8} "
            f"{fmt_acc(row.get('best_val')):>8} {fmt_acc(row.get('test_at_best_val')):>9} "
            f"{fnum(row.get('gpu_memory_peak_gb')):8.3f} {age:6d}s"
        )
else:
    print("No epoch_logs.csv files found yet.")
PY

  echo
  echo "== Latest Log Signals =="
  if [[ -n "$latest" && -f "$latest" ]]; then
    echo "latest_log=$latest"
    grep -E "\[RESULT\]|\[SEED_SUMMARY\]|\[FINAL\]|\[BUDGET_GUARD\]|STOPPED_BUDGET_GUARD|Traceback|ERROR|FAILED|CANCELLED" "$latest" \
      | tail -n 30 || true
  else
    echo "No Slurm .out logs found under $LOG_DIR"
  fi
}

if [[ "$WATCH_SECONDS" != "0" ]]; then
  while true; do
    latest="$(latest_log)"
    clear || true
    show_snapshot "$latest"
    sleep "$WATCH_SECONDS"
  done
fi

latest="$(latest_log)"
show_snapshot "$latest"

if [[ "$FOLLOW_LOG" == "1" && -n "${latest:-}" && -f "$latest" ]]; then
  echo
  echo "== Tailing Latest Log =="
  echo "Set FOLLOW_LOG=0 to print the accuracy snapshot without tail -f."
  echo "Tailing $latest"
  tail -n "$TAIL_LINES" -f "$latest"
fi
