#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def fmt(values):
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return "n/a"
    if len(clean) == 1:
        return f"{clean[0]:.6f}+/-0.000000"
    return f"{mean(clean):.6f}+/-{pstdev(clean):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/ld_revgat_gsmp")
    args = parser.parse_args()

    root = Path(args.results_root)
    groups = defaultdict(list)
    for path in sorted(root.glob("*/final_summary.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        method = payload.get("method", "unknown")
        norm = payload.get("gsmp_norm", "scale_preserve")
        key = f"{method}({norm})" if "GSMP" in method else method
        groups[key].append(payload)

    if not groups:
        raise SystemExit(f"No final_summary.json files found under {root}")

    header = (
        "method",
        "val_acc_mean+/-std",
        "test_at_best_val_mean+/-std",
        "best_raw_test_mean+/-std",
        "best_epoch_mean+/-std",
        "runtime_mean_sec",
        "gpu_memory_peak_mean_gb",
        "runs",
    )
    print("\t".join(header))
    for method, rows in sorted(groups.items()):
        runtime = [r.get("runtime_mean") for r in rows]
        gpu = [r.get("gpu_memory_peak_mean") for r in rows]
        runtime_mean = "n/a" if not [x for x in runtime if x is not None] else f"{mean([float(x) for x in runtime if x is not None]):.2f}"
        gpu_mean = "n/a" if not [x for x in gpu if x is not None] else f"{mean([float(x) for x in gpu if x is not None]):.3f}"
        print(
            "\t".join(
                [
                    method,
                    fmt([r.get("val_mean") for r in rows]),
                    fmt([r.get("test_at_best_val_mean") for r in rows]),
                    fmt([r.get("best_raw_test_mean") for r in rows]),
                    fmt([r.get("best_epoch_mean") for r in rows]),
                    runtime_mean,
                    gpu_mean,
                    str(len(rows)),
                ]
            )
        )


if __name__ == "__main__":
    main()
