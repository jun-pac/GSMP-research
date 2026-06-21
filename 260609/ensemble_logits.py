from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble 260609 best-by-validation logits.")
    parser.add_argument("--run_dirs", required=True, help="Space-separated component run directories.")
    parser.add_argument("--variant", default="baseline", choices=["baseline", "smp", "ump", "gsmp"])
    parser.add_argument("--seeds", default="42 43 44")
    parser.add_argument("--weights", default=None, help="Optional space-separated weights matching --run_dirs.")
    parser.add_argument("--data_root", default="../data")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    run_dirs = [resolve_path(part, project_dir) for part in args.run_dirs.split() if part.strip()]
    seeds = [int(part) for part in args.seeds.replace(",", " ").split() if part.strip()]
    if not run_dirs:
        raise ValueError("--run_dirs must contain at least one directory.")
    weights = parse_weights(args.weights, len(run_dirs))

    data_root = resolve_path(args.data_root, project_dir)
    with torch_load_weights_only_false():
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(data_root))
        data = dataset[0]
    y_true = data.y.view(-1, 1).cpu()
    split_idx = {key: value.long() for key, value in dataset.get_idx_split().items()}
    evaluator = Evaluator(name="ogbn-arxiv")

    output_dir = resolve_path(args.output_dir, project_dir) if args.output_dir else project_dir / "results" / "ensembles"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in seeds:
        logits = []
        for run_dir in run_dirs:
            path = run_dir / "cached_embs" / args.variant / f"logits_seed{seed}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing logits for seed={seed}: {path}")
            logits.append(torch.load(path, map_location="cpu"))
        y_prob = sum(logit.softmax(dim=-1) * weight for logit, weight in zip(logits, weights))
        y_pred = y_prob.argmax(dim=-1, keepdim=True)
        val_acc = float(evaluator.eval({"y_true": y_true[split_idx["valid"]], "y_pred": y_pred[split_idx["valid"]]})["acc"])
        test_acc = float(evaluator.eval({"y_true": y_true[split_idx["test"]], "y_pred": y_pred[split_idx["test"]]})["acc"])
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "variant": args.variant,
            "seed": seed,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "run_dirs": " ".join(str(path) for path in run_dirs),
            "weights": " ".join(str(weight) for weight in weights),
        }
        rows.append(row)
        print(
            f"ENSEMBLE_RESULT variant={args.variant} seed={seed} "
            f"val_acc={val_acc:.4f} test_acc={test_acc:.4f}",
            flush=True,
        )

    csv_path = output_dir / f"ensemble_{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    val = [float(row["val_acc"]) for row in rows]
    test = [float(row["test_acc"]) for row in rows]
    summary = {
        "variant": args.variant,
        "num_seeds": len(rows),
        "val_acc_mean": mean(val),
        "val_acc_std": std(val),
        "test_acc_mean": mean(test),
        "test_acc_std": std(test),
        "csv_path": str(csv_path),
        "run_dirs": [str(path) for path in run_dirs],
        "weights": weights,
    }
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(
        f"ENSEMBLE_SUMMARY variant={args.variant} seeds={len(rows)} "
        f"val_acc={summary['val_acc_mean']:.4f}+-{summary['val_acc_std']:.4f} "
        f"test_acc={summary['test_acc_mean']:.4f}+-{summary['test_acc_std']:.4f}",
        flush=True,
    )
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)


def parse_weights(value: str | None, count: int) -> list[float]:
    if value is None:
        return [1.0 / count for _ in range(count)]
    weights = [float(part) for part in value.replace(",", " ").split() if part.strip()]
    if len(weights) != count:
        raise ValueError(f"Got {len(weights)} weights for {count} run_dirs.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return [weight / total for weight in weights]


def resolve_path(path: str | Path, project_dir: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


class torch_load_weights_only_false:
    def __enter__(self):
        self.original_load = torch.load

        def load_with_legacy_default(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return self.original_load(*args, **kwargs)

        torch.load = load_with_legacy_default

    def __exit__(self, exc_type, exc, tb):
        torch.load = self.original_load
        return False


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


if __name__ == "__main__":
    main()
