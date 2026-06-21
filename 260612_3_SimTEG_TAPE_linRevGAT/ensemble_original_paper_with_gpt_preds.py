#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset


PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_COMPONENTS = ("arxiv_e5", "arxiv_roberta", "tape_e5", "tape_roberta", "gpt_preds")
DEFAULT_WEIGHTS = (2.0, 2.0, 1.0, 1.0, 1.0)
DEFAULT_MODES = ("baseline", "gsmp_first_layer")
FEATURE_LABELS = {
    "arxiv_e5": "ogbn-arxiv_e5-large",
    "arxiv_roberta": "ogbn-arxiv_all-roberta-large-v1",
    "tape_e5": "ogbn-arxiv-tape_e5-large",
    "tape_roberta": "ogbn-arxiv-tape_all-roberta-large-v1",
    "gpt_preds": "ogbn-arxiv_gpt-preds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Compute SimTeG-style original-paper ensemble with GPT preds for baseline and GSMP."
    )
    parser.add_argument("--prefix", required=True, help="Run prefix, e.g. 20260612_043110.")
    parser.add_argument("--seeds", default="1 2 3")
    parser.add_argument("--modes", default=" ".join(DEFAULT_MODES))
    parser.add_argument("--components", default=" ".join(DEFAULT_COMPONENTS))
    parser.add_argument("--weights", default=" ".join(str(weight) for weight in DEFAULT_WEIGHTS))
    parser.add_argument("--results-root", default="results/simteg_tape_linearrevgat_gsmp")
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--output-json", default="results_ensemble_with_gpt_preds_original_paper.json")
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_ints(args.seeds)
    modes = parse_words(args.modes)
    components = parse_words(args.components)
    weights = normalize(parse_floats(args.weights))
    if len(components) != len(weights):
        raise ValueError(f"Got {len(components)} components but {len(weights)} weights.")
    if any(mode.startswith("pgsmp") for mode in modes):
        raise ValueError("P-GSMP is intentionally excluded from GPT-pred ensembles.")

    results_root = resolve_path(args.results_root)
    data_root = resolve_path(args.data_root)
    output_json = resolve_path(args.output_json)
    output_csv = resolve_path(args.output_csv) if args.output_csv else output_json.with_suffix(".csv")

    missing = collect_missing_logits(results_root, args.prefix, modes, components, seeds)
    if missing:
        print("[MISSING] GPT/text ensemble logits are not complete yet.", file=sys.stderr)
        for path in missing[:20]:
            print(f"[MISSING] {path}", file=sys.stderr)
        if len(missing) > 20:
            print(f"[MISSING] ... and {len(missing) - 20} more", file=sys.stderr)
        print("", file=sys.stderr)
        print("Run the missing GPT-pred jobs first:", file=sys.stderr)
        print(
            f'RUN_ID_PREFIX={args.prefix} SEEDS="{args.seeds}" EPOCHS=200 COMPONENTS="gpt_preds" '
            f'EXPERIMENT_MODES="{" ".join(modes)}" '
            "bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh all",
            file=sys.stderr,
        )
        raise SystemExit(2)

    with torch_load_weights_only_false():
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(data_root))
        data = dataset[0]
    y_true = data.y.view(-1, 1).cpu()
    split_idx = {key: value.long() for key, value in dataset.get_idx_split().items()}
    evaluator = Evaluator(name="ogbn-arxiv")

    all_rows = []
    summaries = []
    for mode in modes:
        rows = []
        for seed in seeds:
            y_prob = None
            logit_paths = []
            for component, weight in zip(components, weights):
                logit_path = logit_file(results_root, args.prefix, mode, component, seed)
                logits = torch.load(logit_path, map_location="cpu", weights_only=False)
                weighted_prob = logits.softmax(dim=-1) * weight
                y_prob = weighted_prob if y_prob is None else y_prob + weighted_prob
                logit_paths.append(str(logit_path))
            assert y_prob is not None
            y_pred = y_prob.argmax(dim=-1, keepdim=True)
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "prefix": args.prefix,
                "mode": mode,
                "seed": seed,
                "components": " ".join(components),
                "feature_sources": " ".join(FEATURE_LABELS.get(component, component) for component in components),
                "raw_weights": args.weights,
                "normalized_weights": " ".join(f"{weight:.8f}" for weight in weights),
                "train_acc": eval_acc(evaluator, y_true, y_pred, split_idx["train"]),
                "val_acc": eval_acc(evaluator, y_true, y_pred, split_idx["valid"]),
                "test_at_best_val": eval_acc(evaluator, y_true, y_pred, split_idx["test"]),
                "logit_paths": " ".join(logit_paths),
            }
            rows.append(row)
            all_rows.append(row)
            print(
                f"[ENSEMBLE_RESULT] mode={mode} seed={seed} "
                f"val={row['val_acc']:.4f} test={row['test_at_best_val']:.4f}",
                flush=True,
            )
        summary = {
            "mode": mode,
            "num_seeds": len(rows),
            "seeds": seeds,
            "train_mean": mean([row["train_acc"] for row in rows]),
            "train_std": std([row["train_acc"] for row in rows]),
            "val_mean": mean([row["val_acc"] for row in rows]),
            "val_std": std([row["val_acc"] for row in rows]),
            "test_at_best_val_mean": mean([row["test_at_best_val"] for row in rows]),
            "test_at_best_val_std": std([row["test_at_best_val"] for row in rows]),
            "rows": rows,
        }
        summaries.append(summary)
        print(
            f"[ENSEMBLE_FINAL] mode={mode} "
            f"val={summary['val_mean']:.4f}+/-{summary['val_std']:.4f} "
            f"test={summary['test_at_best_val_mean']:.4f}+/-{summary['test_at_best_val_std']:.4f}",
            flush=True,
        )

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "name": "results_ensemble_with_gpt_preds_original_paper",
        "prefix": args.prefix,
        "algorithm": "SimTeG-style weighted average of softmax probabilities, then argmax.",
        "components": components,
        "feature_sources": [FEATURE_LABELS.get(component, component) for component in components],
        "raw_weights": parse_floats(args.weights),
        "normalized_weights": weights,
        "modes": modes,
        "summaries": summaries,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[SAVED] json={output_json}", flush=True)
    print(f"[SAVED] csv={output_csv}", flush=True)


def collect_missing_logits(
    results_root: Path,
    prefix: str,
    modes: list[str],
    components: list[str],
    seeds: list[int],
) -> list[Path]:
    missing = []
    for mode in modes:
        for component in components:
            for seed in seeds:
                path = logit_file(results_root, prefix, mode, component, seed)
                if not path.is_file():
                    missing.append(path)
    return missing


def logit_file(results_root: Path, prefix: str, mode: str, component: str, seed: int) -> Path:
    return results_root / f"{prefix}_{mode}_{component}_seed{seed}" / "cached_embs" / f"logits_seed{seed}.pt"


def eval_acc(
    evaluator: Evaluator,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    idx: torch.Tensor,
) -> float:
    return float(evaluator.eval({"y_true": y_true[idx], "y_pred": y_pred[idx]})["acc"])


def parse_words(value: str) -> list[str]:
    return [part for part in value.replace(",", " ").split() if part.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(part) for part in parse_words(value)]


def parse_floats(value: str) -> list[float]:
    return [float(part) for part in parse_words(value)]


def normalize(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return [weight / total for weight in weights]


def resolve_path(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("Path cannot be None.")
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
