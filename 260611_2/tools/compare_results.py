#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def fmt(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return "n/a"
    return f"{arr.mean():.4f}+/-{arr.std(ddof=0):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/glem_revgat_gsmp")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.results_root).glob("*/final_summary.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        rows.append(data)

    grouped = defaultdict(list)
    for row in rows:
        method = row.get("method", "unknown")
        if row.get("use_gsmp"):
            method = f"{method}({row.get('gsmp_norm', 'unknown')})"
        grouped[method].append(row)

    print("method,val_acc_mean+/-std,test_at_best_val_mean+/-std,best_raw_test_mean+/-std,best_epoch_mean+/-std,runtime_mean,gpu_memory_peak_mean")
    for method in sorted(grouped):
        vals = grouped[method]
        runtime = np.asarray([float(v.get("total_time", 0.0)) for v in vals], dtype=float)
        gpu = np.asarray([float(v.get("gpu_memory_peak_gb", 0.0)) for v in vals], dtype=float)
        print(
            f"{method},"
            f"{fmt([v.get('best_val', 0.0) for v in vals])},"
            f"{fmt([v.get('test_at_best_val', 0.0) for v in vals])},"
            f"{fmt([v.get('best_raw_test', 0.0) for v in vals])},"
            f"{fmt([v.get('best_epoch', 0.0) for v in vals])},"
            f"{runtime.mean() if runtime.size else 0.0:.2f},"
            f"{gpu.mean() if gpu.size else 0.0:.3f}"
        )


if __name__ == "__main__":
    main()
