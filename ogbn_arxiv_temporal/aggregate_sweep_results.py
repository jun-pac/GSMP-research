from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate recursive ogbn-arxiv SMP/GSMP sweep results.")
    parser.add_argument("--sweep_dir", type=str, default="outputs/e5_smp_gsmp_sweep")
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    records = load_records(sweep_dir)
    if not records:
        print(f"No run_summary JSON files found under {sweep_dir}")
        return

    records.sort(key=lambda item: float(item["best_valid_acc"]), reverse=True)
    print(f"loaded {len(records)} completed sweep runs from {sweep_dir}")
    print()
    print_top(records, top=args.top)
    print()
    print_best_by_mode(records)
    print()
    print_config_summary(records)


def load_records(sweep_dir: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(sweep_dir.rglob("run_summary_*_seed*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if {"mode", "seed", "best_valid_acc", "test_acc_at_best_valid"}.issubset(data):
            args = data.get("args", {})
            records.append(
                {
                    "mode": data["mode"],
                    "seed": int(data["seed"]),
                    "best_valid_acc": float(data["best_valid_acc"]),
                    "test_acc_at_best_valid": float(data["test_acc_at_best_valid"]),
                    "best_epoch": int(data.get("best_epoch", 0)),
                    "total_epochs_run": int(data.get("total_epochs_run", 0)),
                    "runtime_sec": float(data.get("runtime_sec", 0.0)),
                    "label_smoothing": float(args.get("label_smoothing", 0.0)),
                    "dropout": float(args.get("dropout", 0.0)),
                    "weight_decay": str(args.get("weight_decay", "")),
                    "lr": str(args.get("lr", "")),
                    "batch_size": str(args.get("batch_size", "")),
                    "num_neighbors": str(args.get("num_neighbors", "")),
                    "source": str(path),
                }
            )
    return records


def print_top(records: list[dict[str, object]], top: int) -> None:
    print(f"top {min(top, len(records))} by validation accuracy")
    headers = [
        "rank",
        "mode",
        "seed",
        "valid",
        "test",
        "epoch",
        "ls",
        "drop",
        "wd",
        "lr",
        "batch",
        "fanout",
        "runtime",
    ]
    widths = [5, 6, 6, 10, 10, 7, 6, 6, 9, 7, 7, 8, 9]
    print(format_row(headers, widths))
    print(format_row(["-" * width for width in widths], widths))
    for rank, record in enumerate(records[:top], start=1):
        print(
            format_row(
                [
                    str(rank),
                    str(record["mode"]),
                    str(record["seed"]),
                    f"{float(record['best_valid_acc']):.6f}",
                    f"{float(record['test_acc_at_best_valid']):.6f}",
                    str(record["best_epoch"]),
                    f"{float(record['label_smoothing']):.2f}",
                    f"{float(record['dropout']):.2f}",
                    str(record["weight_decay"]),
                    str(record["lr"]),
                    str(record["batch_size"]),
                    str(record["num_neighbors"]),
                    format_duration(float(record["runtime_sec"])),
                ],
                widths,
            )
        )


def print_best_by_mode(records: list[dict[str, object]]) -> None:
    print("best per mode")
    headers = ["mode", "valid", "test", "seed", "epoch", "ls", "drop", "wd", "lr", "batch", "fanout"]
    widths = [6, 10, 10, 6, 7, 6, 6, 9, 7, 7, 8]
    print(format_row(headers, widths))
    print(format_row(["-" * width for width in widths], widths))
    for mode in sorted({str(record["mode"]) for record in records}):
        best = max(
            [record for record in records if record["mode"] == mode],
            key=lambda item: float(item["best_valid_acc"]),
        )
        print(
            format_row(
                [
                    str(best["mode"]),
                    f"{float(best['best_valid_acc']):.6f}",
                    f"{float(best['test_acc_at_best_valid']):.6f}",
                    str(best["seed"]),
                    str(best["best_epoch"]),
                    f"{float(best['label_smoothing']):.2f}",
                    f"{float(best['dropout']):.2f}",
                    str(best["weight_decay"]),
                    str(best["lr"]),
                    str(best["batch_size"]),
                    str(best["num_neighbors"]),
                ],
                widths,
            )
        )


def print_config_summary(records: list[dict[str, object]]) -> None:
    print("mean by mode and hyperparameters")
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for record in records:
        key = (
            record["mode"],
            record["label_smoothing"],
            record["dropout"],
            record["weight_decay"],
            record["lr"],
            record["batch_size"],
            record["num_neighbors"],
        )
        grouped.setdefault(key, []).append(record)

    rows = []
    for key, group in grouped.items():
        valid = [float(item["best_valid_acc"]) for item in group]
        test = [float(item["test_acc_at_best_valid"]) for item in group]
        rows.append(
            {
                "key": key,
                "num": len(group),
                "valid_mean": mean(valid),
                "valid_std": std(valid),
                "test_mean": mean(test),
                "test_std": std(test),
            }
        )
    rows.sort(key=lambda item: float(item["valid_mean"]), reverse=True)

    headers = ["mode", "runs", "valid mean+-std", "test mean+-std", "ls", "drop", "wd", "lr", "batch", "fanout"]
    widths = [6, 6, 18, 18, 6, 6, 9, 7, 7, 8]
    print(format_row(headers, widths))
    print(format_row(["-" * width for width in widths], widths))
    for row in rows[:20]:
        mode, label_smoothing, dropout, weight_decay, lr, batch_size, num_neighbors = row["key"]
        print(
            format_row(
                [
                    str(mode),
                    str(row["num"]),
                    f"{row['valid_mean']:.6f}+-{row['valid_std']:.6f}",
                    f"{row['test_mean']:.6f}+-{row['test_std']:.6f}",
                    f"{float(label_smoothing):.2f}",
                    f"{float(dropout):.2f}",
                    str(weight_decay),
                    str(lr),
                    str(batch_size),
                    str(num_neighbors),
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
