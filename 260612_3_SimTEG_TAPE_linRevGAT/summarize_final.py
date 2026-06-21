#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Print a compact comparison table from final_summary.json files.")
    parser.add_argument("summaries", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.summaries:
        with path.open() as handle:
            summary = json.load(handle)
        rows.append(
            [
                summary.get("method", path.parent.name),
                fmt(summary.get("val_mean"), summary.get("val_std")),
                fmt(summary.get("test_at_best_val_mean"), summary.get("test_at_best_val_std")),
                fmt(
                    summary.get("best_raw_test_mean_diagnostic_only"),
                    summary.get("best_raw_test_std_diagnostic_only"),
                ),
                fmt(summary.get("best_epoch_mean"), summary.get("best_epoch_std"), digits=1),
                f"{float(summary.get('runtime_mean', 0.0)):.1f}s",
                f"{float(summary.get('peak_gpu_memory_mean', 0.0)):.1f}MB",
                f"{float(summary.get('preprocessing_time', 0.0)):.1f}s",
                str(summary.get("cache_reused", False)),
            ]
        )
    headers = [
        "method",
        "val_acc_mean+/-std",
        "test_at_best_val_mean+/-std",
        "best_raw_test_mean+/-std",
        "best_epoch_mean+/-std",
        "runtime_mean",
        "peak_gpu_memory_mean",
        "preprocessing_time",
        "cache_reused",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        print(" | ".join(row))


def fmt(mean_value, std_value, digits: int = 4) -> str:
    if mean_value is None:
        return "NA"
    return f"{float(mean_value):.{digits}f}+/-{float(std_value or 0.0):.{digits}f}"


if __name__ == "__main__":
    main()

