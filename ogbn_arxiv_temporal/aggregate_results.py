from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ogbn-arxiv temporal GraphSAGE results.")
    parser.add_argument("--results_dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    records = load_records(results_dir)
    if not records:
        print(f"No result CSV/JSON files found in {results_dir}")
        return

    print_table(records)
    print()
    print_mode_summary(records)


def load_records(results_dir: Path) -> list[dict[str, object]]:
    by_key: dict[tuple[str, int], dict[str, object]] = {}

    for path in sorted(results_dir.glob("ogbn_arxiv_simteg_tape_sage_*_seed*.csv")):
        record = record_from_csv(path)
        if record is not None:
            by_key[(str(record["mode"]), int(record["seed"]))] = record

    for path in sorted(results_dir.glob("run_summary_*_seed*.json")):
        record = record_from_json(path)
        if record is not None:
            by_key[(str(record["mode"]), int(record["seed"]))] = record

    return sorted(by_key.values(), key=lambda item: (str(item["mode"]), int(item["seed"])))


def record_from_csv(path: Path) -> dict[str, object] | None:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    final = rows[-1]
    return {
        "mode": final["mode"],
        "seed": int(final["seed"]),
        "best_valid_acc": float(final["best_valid_acc"]),
        "test_acc_at_best_valid": float(final["test_acc_at_best_valid"]),
        "final_train_acc": float(final["train_acc"]),
        "total_epochs_run": int(final["epoch"]),
        "runtime_sec": float(final["elapsed_sec"]),
        "source": str(path),
    }


def record_from_json(path: Path) -> dict[str, object] | None:
    data = json.loads(path.read_text())
    required = {"mode", "seed", "best_valid_acc", "test_acc_at_best_valid"}
    if not required.issubset(data):
        return None
    return {
        "mode": data["mode"],
        "seed": int(data["seed"]),
        "best_valid_acc": float(data["best_valid_acc"]),
        "test_acc_at_best_valid": float(data["test_acc_at_best_valid"]),
        "final_train_acc": float(data.get("final_train_acc", 0.0)),
        "total_epochs_run": int(data.get("total_epochs_run", 0)),
        "runtime_sec": float(data.get("runtime_sec", 0.0)),
        "source": str(path),
    }


def print_table(records: list[dict[str, object]]) -> None:
    headers = [
        "mode",
        "seed",
        "best_valid",
        "test_at_best",
        "final_train",
        "epochs",
        "runtime",
    ]
    widths = [10, 6, 12, 13, 13, 8, 10]
    print(format_row(headers, widths))
    print(format_row(["-" * width for width in widths], widths))
    for record in records:
        print(
            format_row(
                [
                    str(record["mode"]),
                    str(record["seed"]),
                    f"{float(record['best_valid_acc']):.6f}",
                    f"{float(record['test_acc_at_best_valid']):.6f}",
                    f"{float(record['final_train_acc']):.6f}",
                    str(record["total_epochs_run"]),
                    format_duration(float(record["runtime_sec"])),
                ],
                widths,
            )
        )


def print_mode_summary(records: list[dict[str, object]]) -> None:
    print("mode summary")
    headers = ["mode", "seeds", "best_valid mean+-std", "test_at_best mean+-std"]
    widths = [10, 8, 24, 24]
    print(format_row(headers, widths))
    print(format_row(["-" * width for width in widths], widths))
    for mode in sorted({str(record["mode"]) for record in records}):
        subset = [record for record in records if record["mode"] == mode]
        valid = [float(record["best_valid_acc"]) for record in subset]
        test = [float(record["test_acc_at_best_valid"]) for record in subset]
        print(
            format_row(
                [
                    mode,
                    str(len(subset)),
                    f"{mean(valid):.6f}+-{std(valid):.6f}",
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
    return statistics.stdev(values) if len(values) > 1 else 0.0


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()
