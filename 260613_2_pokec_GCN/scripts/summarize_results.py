#!/usr/bin/env python3
"""Summarize Pokec temporal GCN/GSMP results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def mean_std(series: pd.Series) -> str:
    return f"{series.mean():.4f} +/- {series.std(ddof=0):.4f}"


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize pokec_temporal_results.csv.")
    parser.add_argument("--results-csv", type=Path, default=Path("results/pokec_temporal_results.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("results/pokec_temporal_summary.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results_csv)
    rows = []
    for method, group in df.groupby("method"):
        rows.append(
            {
                "method": method,
                "runs": len(group),
                "valid_acc": mean_std(group["best_valid_acc"]),
                "test_at_best_valid": mean_std(group["test_acc_at_best_valid"]),
                "best_epochs": ", ".join(str(int(x)) for x in group["best_epoch"].tolist()),
                "max_gpu_mem_gb": mean_std(group["max_gpu_mem_gb"]),
                "runtime_sec": mean_std(group["total_time_sec"]),
            }
        )
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# Pokec Temporal Results Summary\n\n")
        f.write(markdown_table(summary))
        f.write("\n\n")
        f.write("Per-seed rows are in `pokec_temporal_results.csv`.\n")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
