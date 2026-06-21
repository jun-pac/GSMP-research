#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


METHOD_LABELS = {
    ("RevGAT", "False", "scale_preserve"): "TAPE+RevGAT",
    ("LinearRevGAT", "False", "scale_preserve"): "TAPE+LinearRevGAT",
    ("LinearRevGAT", "True", "scale_preserve"): "TAPE+LinearRevGAT+GSMP(scale)",
    ("LinearRevGAT", "True", "strict"): "TAPE+LinearRevGAT+GSMP(strict)",
}


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def numeric(rows, name):
    values = []
    for row in rows:
        value = row.get(name, "")
        if value == "":
            continue
        values.append(float(value))
    return values


def fmt(values):
    mean, std = mean_std(values)
    return f"{mean:.4f}+/-{std:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Print comparison table from TAPE GSMP seed summaries.")
    parser.add_argument("--results-root", type=Path, default=Path("results/tape_revgat_gsmp"))
    parser.add_argument("--feature-type", default="ensemble", help="Feature type filter, e.g. ensemble, TA, P, E, ogb, or all.")
    args = parser.parse_args()

    groups = defaultdict(list)
    for path in sorted(args.results_root.glob("*/seed_summary.csv")):
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if args.feature_type != "all" and row.get("feature_type") != args.feature_type:
                    continue
                key = (row.get("model"), row.get("use_gsmp"), row.get("gsmp_norm"))
                groups[key].append(row)

    header = (
        "method                              val_acc_mean+/-std  "
        "test_acc_at_best_val_mean+/-std  best_raw_test_mean+/-std  best_epoch_mean  runtime_mean"
    )
    print(header)
    print("-" * len(header))
    for key, label in METHOD_LABELS.items():
        rows = groups.get(key, [])
        if not rows:
            continue
        vals = numeric(rows, "best_val")
        tests = numeric(rows, "test_at_best_val")
        raw = numeric(rows, "best_raw_test")
        epochs = numeric(rows, "best_epoch")
        runtimes = numeric(rows, "runtime_seconds")
        print(
            f"{label:<35} {fmt(vals):<19} {fmt(tests):<32} "
            f"{fmt(raw):<25} {mean_std(epochs)[0] if epochs else float('nan'):<15.1f} {mean_std(runtimes)[0]:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
