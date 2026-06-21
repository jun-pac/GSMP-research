#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def read_rows(root: Path):
    for path in sorted(root.glob("*/seed_summary.csv")):
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_path"] = str(path)
                yield row


def main() -> int:
    parser = argparse.ArgumentParser(description="Refuse full GSMP runs until RevGAT baseline is close enough.")
    parser.add_argument("--results-root", type=Path, default=Path("results/tape_revgat_gsmp"))
    parser.add_argument("--min-val", type=float, default=0.77)
    parser.add_argument("--model", default="RevGAT")
    args = parser.parse_args()

    rows = [
        row for row in read_rows(args.results_root)
        if row.get("model") == args.model and row.get("feature_type") in {"TA", "P", "E", "ensemble"}
    ]
    if not rows:
        print(f"No {args.model} baseline seed_summary rows found under {args.results_root}", file=sys.stderr)
        return 2
    best = max(float(row.get("best_val", 0.0)) for row in rows)
    if best < args.min_val:
        print(
            f"Baseline gate failed: best_val={best:.4f} < {args.min_val:.4f}. "
            "Debug RevGAT reproduction before running full GSMP.",
            file=sys.stderr,
        )
        return 1
    print(f"Baseline gate passed: best_val={best:.4f} >= {args.min_val:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
