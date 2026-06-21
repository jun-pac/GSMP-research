from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 260609 ogbn-arxiv SMP/UMP/GSMP results.")
    parser.add_argument("--run_dir", required=True, help="Directory containing summary.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    summary_path = run_dir / "summary.csv"
    if not summary_path.is_file():
        print(f"No summary.csv found at {summary_path}")
        return

    with summary_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"No rows in {summary_path}")
        return

    rows = dedupe(rows)
    rows.sort(key=lambda row: (row["variant"], int(row["seed"])))
    print(f"summary: {summary_path}")
    print_table(rows)
    print()
    print_aggregate(rows)


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        latest[(row["variant"], int(row["seed"]))] = row
    return list(latest.values())


def print_table(rows: list[dict[str, str]]) -> None:
    headers = ["variant", "seed", "best_epoch", "best_val", "test_at_best", "oracle_test", "epochs"]
    widths = [10, 6, 10, 10, 13, 12, 8]
    print(format_row(headers, widths))
    print(format_row(["-" * width for width in widths], widths))
    for row in rows:
        print(
            format_row(
                [
                    row["variant"],
                    row["seed"],
                    row["best_epoch"],
                    f"{float(row['best_val_acc']):.6f}",
                    f"{float(row['test_at_best_val']):.6f}",
                    f"{float(row['oracle_best_test_acc_not_for_model_selection']):.6f}",
                    row["epochs_run"],
                ],
                widths,
            )
        )


def print_aggregate(rows: list[dict[str, str]]) -> None:
    headers = ["variant", "seeds", "best_val mean+-std", "test_at_best mean+-std"]
    widths = [10, 8, 22, 24]
    print("aggregate by variant")
    print(format_row(headers, widths))
    print(format_row(["-" * width for width in widths], widths))
    for variant in sorted({row["variant"] for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        val = [float(row["best_val_acc"]) for row in subset]
        test = [float(row["test_at_best_val"]) for row in subset]
        print(
            format_row(
                [
                    variant,
                    str(len(subset)),
                    f"{mean(val):.6f}+-{std(val):.6f}",
                    f"{mean(test):.6f}+-{std(test):.6f}",
                ],
                widths,
            )
        )


def format_row(values: list[str], widths: list[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(values, widths))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


if __name__ == "__main__":
    main()
