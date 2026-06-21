#!/usr/bin/env python3
"""Summarize HGAMLP-HOPE impact result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METHOD_LABELS = {
    "baseline": "HH",
    "smp": "HH+SMP",
    "ump": "HH+UMP",
    "gsmp": "HH+GSMP",
}


def load_rows(result_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(result_dir.glob("hgamlp_hope_*_seed_*.json")):
        with path.open() as handle:
            payload = json.load(handle)
        results = payload.get("results", [payload])
        for result in results:
            method = result["method"]
            rows.append(
                {
                    "method": method,
                    "label": METHOD_LABELS.get(method, method),
                    "seed": result["seed"],
                    "best_valid_acc": result["best_valid_acc"],
                    "best_test_acc": result["best_test_acc"],
                    "best_epoch": result["best_epoch"],
                    "file": str(path),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.result_dir))
    if not rows:
        print(f"No result JSON files found in {args.result_dir}")
        return

    df = pd.DataFrame(rows).sort_values(["method", "seed"])
    summary = (
        df.groupby(["method", "label"], as_index=False)
        .agg(
            seeds=("seed", "count"),
            mean_valid_acc=("best_valid_acc", "mean"),
            std_valid_acc=("best_valid_acc", "std"),
            mean_test_acc=("best_test_acc", "mean"),
            std_test_acc=("best_test_acc", "std"),
            best_valid_acc=("best_valid_acc", "max"),
            best_test_acc=("best_test_acc", "max"),
        )
        .sort_values("method")
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    print("\nPer-run best results")
    print(df[["label", "seed", "best_valid_acc", "best_test_acc", "best_epoch"]].to_string(index=False))
    print("\nMethod summary")
    print(summary.to_string(index=False))
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
