#!/usr/bin/env python
import argparse
import csv
import math
import re
import statistics
from pathlib import Path


RESULT_RE = re.compile(
    r"RESULT method=(?P<method>\S+) seed=(?P<seed>\d+) stage=(?P<stage>\d+) "
    r"epoch=(?P<epoch>\d+) train_acc=(?P<train_acc>[0-9.eE+-]+) "
    r"val_acc=(?P<val_acc>[0-9.eE+-]+) test_acc=(?P<test_acc>[0-9.eE+-]+) "
    r"best_epoch=(?P<best_epoch>\d+) best_val=(?P<best_val>[0-9.eE+-]+) "
    r"best_test_at_best_val=(?P<best_test_at_best_val>[0-9.eE+-]+) "
    r"elapsed_sec=(?P<elapsed_sec>[0-9.eE+-]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize SeHGNN-HOPE ogbn-mag runs.")
    parser.add_argument("--progress-file", default="results/live_progress.tsv")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--out", default="results")
    return parser.parse_args()


def to_float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def read_progress(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if not row.get("method"):
                continue
            row.setdefault("job_id", "")
            rows.append(row)
    return rows


def read_logs(log_dir):
    rows = []
    for path in sorted(log_dir.glob("**/*")):
        if not path.is_file() or path.suffix not in ("", ".log", ".out"):
            continue
        text = path.read_text(errors="replace")
        for match in RESULT_RE.finditer(text):
            rows.append(match.groupdict())
    return rows


def best_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["method"], int(row["seed"]))
        rank = (int(row.get("stage", 0)), to_float(row, "best_val"), int(row.get("epoch", 0)))
        if key not in grouped or rank >= grouped[key][0]:
            grouped[key] = (rank, row)
    return [item[1] for item in sorted(grouped.values(), key=lambda x: (x[1]["method"], int(x[1]["seed"])))]


def mean_std(values):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def write_tsv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_path = Path(args.progress_file)
    rows = read_progress(progress_path)
    source = str(progress_path)
    if not rows:
        rows = read_logs(Path(args.log_dir))
        source = args.log_dir

    per_seed = []
    for row in best_rows(rows):
        per_seed.append({
            "method": row["method"],
            "seed": int(row["seed"]),
            "stage": int(row.get("stage", 0)),
            "best_epoch": int(row.get("best_epoch", row.get("epoch", 0))),
            "best_valid": f"{to_float(row, 'best_val'):.8f}",
            "test_at_best_valid": f"{to_float(row, 'best_test_at_best_val'):.8f}",
            "last_valid": f"{to_float(row, 'val_acc'):.8f}",
            "last_test": f"{to_float(row, 'test_acc'):.8f}",
            "elapsed_sec": f"{to_float(row, 'elapsed_sec'):.2f}",
        })

    by_method = {}
    for row in per_seed:
        by_method.setdefault(row["method"], []).append(row)

    summary = []
    for method, method_rows in sorted(by_method.items()):
        valid_mean, valid_std = mean_std([float(r["best_valid"]) for r in method_rows])
        test_mean, test_std = mean_std([float(r["test_at_best_valid"]) for r in method_rows])
        summary.append({
            "method": method,
            "runs": len(method_rows),
            "best_valid_mean": f"{valid_mean:.8f}",
            "best_valid_std": f"{valid_std:.8f}",
            "test_at_best_valid_mean": f"{test_mean:.8f}",
            "test_at_best_valid_std": f"{test_std:.8f}",
        })

    per_seed_fields = [
        "method", "seed", "stage", "best_epoch", "best_valid",
        "test_at_best_valid", "last_valid", "last_test", "elapsed_sec",
    ]
    summary_fields = [
        "method", "runs", "best_valid_mean", "best_valid_std",
        "test_at_best_valid_mean", "test_at_best_valid_std",
    ]
    write_tsv(out_dir / "per_seed.tsv", per_seed, per_seed_fields)
    write_tsv(out_dir / "summary.tsv", summary, summary_fields)

    lines = [
        "# SeHGNN-HOPE ogbn-mag Summary",
        "",
        f"Source: `{source}`",
        "",
        "| method | runs | best valid | test at best valid |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['runs']} | "
            f"{row['best_valid_mean']} +/- {row['best_valid_std']} | "
            f"{row['test_at_best_valid_mean']} +/- {row['test_at_best_valid_std']} |"
        )
    lines.extend(["", "## Per Seed", ""])
    lines.append("| method | seed | stage | best epoch | best valid | test at best valid |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in per_seed:
        lines.append(
            f"| {row['method']} | {row['seed']} | {row['stage']} | {row['best_epoch']} | "
            f"{row['best_valid']} | {row['test_at_best_valid']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_dir / 'summary.tsv'}")
    print(f"Wrote {out_dir / 'per_seed.tsv'}")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
