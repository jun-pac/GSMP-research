#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
TARGET_TEST = 0.7803
TARGET_VAL = 0.7846


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Verify cached official SimTeG+TAPE+RevGAT anchor logits if available.")
    parser.add_argument("--seeds", default="1 2 3 4 5 6 7 8 9 10")
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--simteg-root", default="../SimTeG")
    parser.add_argument("--output-dir", default="results/anchor_verification")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    simteg_root = resolve_path(args.simteg_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    component_dirs = official_component_dirs(simteg_root)
    missing = missing_logits(component_dirs, seeds)
    if missing:
        print("[WARNING] SimTeG+TAPE+RevGAT leaderboard anchor was not reproduced in this run.", flush=True)
        print("[WARNING] Interpret linearRevGAT GSMP comparisons as internal ablations only.", flush=True)
        print("[ANCHOR] missing_cached_logits_count=" + str(len(missing)), flush=True)
        for path in missing[:20]:
            print(f"[ANCHOR] missing={path}", flush=True)
        write_json(
            output_dir / f"anchor_missing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            {
                "status": "missing_cached_logits",
                "missing_count": len(missing),
                "missing_examples": [str(path) for path in missing[:50]],
                "expected_test": "0.7803 +/- 0.0007",
                "expected_validation": "0.7846 +/- 0.0004",
            },
        )
        return

    with torch_load_weights_only_false():
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(resolve_path(args.data_root)))
        data = dataset[0]
    y_true = data.y.view(-1, 1).cpu()
    split = {key: value.long() for key, value in dataset.get_idx_split().items()}
    evaluator = Evaluator(name="ogbn-arxiv")
    weights = normalize([2, 2, 1, 1, 1])
    rows = []
    for seed in seeds:
        probs = []
        for component_dir, weight in zip(component_dirs, weights):
            logits = torch.load(component_dir / f"logits_seed{seed}.pt", map_location="cpu", weights_only=False)
            probs.append(logits.softmax(dim=-1) * weight)
        y_prob = sum(probs)
        y_pred = y_prob.argmax(dim=-1, keepdim=True)
        val = eval_acc(evaluator, y_true, y_pred, split["valid"])
        test = eval_acc(evaluator, y_true, y_pred, split["test"])
        rows.append({"seed": seed, "val_acc": val, "test_acc": test})
        print(f"[ANCHOR_RESULT] seed={seed} val_acc={val:.4f} test_acc={test:.4f}", flush=True)
    val_values = [row["val_acc"] for row in rows]
    test_values = [row["test_acc"] for row in rows]
    summary = {
        "status": "verified_from_cached_logits",
        "seeds": seeds,
        "val_acc_mean": mean(val_values),
        "val_acc_std": std(val_values),
        "test_acc_mean": mean(test_values),
        "test_acc_std": std(test_values),
        "target_val_acc": TARGET_VAL,
        "target_test_acc": TARGET_TEST,
        "component_dirs": [str(path) for path in component_dirs],
    }
    write_json(output_dir / f"anchor_verified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", summary)
    print(
        f"[ANCHOR_FINAL] val_acc={summary['val_acc_mean']:.4f}+/-{summary['val_acc_std']:.4f} "
        f"test_acc={summary['test_acc_mean']:.4f}+/-{summary['test_acc_std']:.4f}",
        flush=True,
    )


def official_component_dirs(simteg_root: Path) -> list[Path]:
    return [
        simteg_root / "out/ogbn-arxiv/revgat/ensemble_X_e5-large/cached_embs",
        simteg_root / "out/ogbn-arxiv/revgat/ensemble_X_all-roberta-large-v1/cached_embs",
        simteg_root / "out/ogbn-arxiv-tape/revgat/ensemble_X_e5-large/cached_embs",
        simteg_root / "out/ogbn-arxiv-tape/revgat/ensemble_X_all-roberta-large-v1/cached_embs",
        simteg_root / "out/ogbn-arxiv/revgat/ensemble_preds/cached_embs",
    ]


def missing_logits(component_dirs: list[Path], seeds: list[int]) -> list[Path]:
    missing = []
    for component_dir in component_dirs:
        for seed in seeds:
            path = component_dir / f"logits_seed{seed}.pt"
            if not path.is_file():
                missing.append(path)
    return missing


def eval_acc(evaluator: Evaluator, y_true: torch.Tensor, y_pred: torch.Tensor, idx: torch.Tensor) -> float:
    return float(evaluator.eval({"y_true": y_true[idx], "y_pred": y_pred[idx]})["acc"])


def parse_seeds(value: str) -> list[int]:
    return [int(part) for part in value.replace(",", " ").split() if part.strip()]


def normalize(weights: list[float]) -> list[float]:
    total = sum(weights)
    return [weight / total for weight in weights]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def write_json(path: Path, obj: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


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


if __name__ == "__main__":
    main()

