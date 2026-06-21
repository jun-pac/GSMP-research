#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import torch
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Ensemble saved best-validation logits.")
    parser.add_argument("--run-dirs", required=True, help="Space-separated run dirs containing cached_embs/logits_seed*.pt")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--weights", default=None, help="Space-separated weights matching --run-dirs.")
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--output-dir", default="results/ensembles")
    parser.add_argument("--name", default="ensemble")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [resolve_path(part) for part in args.run_dirs.split() if part.strip()]
    seeds = [int(part) for part in args.seeds.replace(",", " ").split() if part.strip()]
    weights = parse_weights(args.weights, len(run_dirs))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch_load_weights_only_false():
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(resolve_path(args.data_root)))
        data = dataset[0]
    y_true = data.y.view(-1, 1).cpu()
    split_idx = {key: value.long() for key, value in dataset.get_idx_split().items()}
    evaluator = Evaluator(name="ogbn-arxiv")

    rows = []
    for seed in seeds:
        logits = []
        for run_dir in run_dirs:
            path = run_dir / "cached_embs" / f"logits_seed{seed}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing logits for seed={seed}: {path}")
            logits.append(torch.load(path, map_location="cpu", weights_only=False))
        y_prob = sum(logit.softmax(dim=-1) * weight for logit, weight in zip(logits, weights))
        y_pred = y_prob.argmax(dim=-1, keepdim=True)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "name": args.name,
            "seed": seed,
            "train_acc": eval_acc(evaluator, y_true, y_pred, split_idx["train"]),
            "val_acc": eval_acc(evaluator, y_true, y_pred, split_idx["valid"]),
            "test_at_best_val": eval_acc(evaluator, y_true, y_pred, split_idx["test"]),
            "run_dirs": " ".join(str(path) for path in run_dirs),
            "weights": " ".join(f"{weight:.8f}" for weight in weights),
        }
        rows.append(row)
        print(
            f"[ENSEMBLE_RESULT] name={args.name} seed={seed} "
            f"val_acc={row['val_acc']:.4f} test_at_best_val={row['test_at_best_val']:.4f}",
            flush=True,
        )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "name": args.name,
        "num_seeds": len(rows),
        "seeds": " ".join(str(seed) for seed in seeds),
        "val_mean": mean([float(row["val_acc"]) for row in rows]),
        "val_std": std([float(row["val_acc"]) for row in rows]),
        "test_at_best_val_mean": mean([float(row["test_at_best_val"]) for row in rows]),
        "test_at_best_val_std": std([float(row["test_at_best_val"]) for row in rows]),
        "run_dirs": [str(path) for path in run_dirs],
        "weights": weights,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{args.name}_{stamp}.csv"
    json_path = csv_path.with_suffix(".json")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"[ENSEMBLE_FINAL] name={args.name} seeds={len(rows)} "
        f"val={summary['val_mean']:.4f}+/-{summary['val_std']:.4f} "
        f"test_at_best_val={summary['test_at_best_val_mean']:.4f}+/-{summary['test_at_best_val_std']:.4f}",
        flush=True,
    )


def eval_acc(evaluator: Evaluator, y_true: torch.Tensor, y_pred: torch.Tensor, idx: torch.Tensor) -> float:
    return float(evaluator.eval({"y_true": y_true[idx], "y_pred": y_pred[idx]})["acc"])


def parse_weights(value: str | None, count: int) -> list[float]:
    if value is None:
        return [1.0 / count for _ in range(count)]
    weights = [float(part) for part in value.replace(",", " ").split() if part.strip()]
    if len(weights) != count:
        raise ValueError(f"Got {len(weights)} weights for {count} run dirs.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return [weight / total for weight in weights]


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


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

