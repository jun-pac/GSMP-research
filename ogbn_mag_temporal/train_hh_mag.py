#!/usr/bin/env python3
"""Train HH/HGAMLP-HOPE ablations on ogbn-mag with tail-friendly logging."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import torch
import torch.nn.functional as F

from models import HHModel
from propagation import (
    build_base_propagation,
    build_gsmp_propagation,
    build_smp_propagation,
    build_ump_propagation,
)
from utils import (
    CSVLogger,
    METRIC_COLUMNS,
    Timer,
    ensure_dirs,
    format_eval_line,
    resolve_device,
    set_seed,
    upsert_summary_row,
    validate_split,
)


logger = logging.getLogger(__name__)


def _torch_load_with_weights_only_false(*args: Any, **kwargs: Any) -> Any:
    """Compatibility shim for OGB/PyG processed files under PyTorch >= 2.6."""
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


_ORIGINAL_TORCH_LOAD = torch.load


def _legacy_ogb_data_to_heterodata(data: Any) -> Any:
    """Convert OGB's legacy ogbn-mag object into a PyG HeteroData object."""
    if hasattr(data, "node_types") and hasattr(data, "edge_types"):
        return data

    try:
        from torch_geometric.data import HeteroData
    except ImportError as exc:
        raise RuntimeError(
            "torch_geometric is required to convert legacy ogbn-mag data. "
            "Install PyTorch Geometric or use an already processed HeteroData dataset."
        ) from exc

    hetero = HeteroData()
    for node_type, num_nodes in data.num_nodes_dict.items():
        hetero[node_type].num_nodes = int(num_nodes)
        if hasattr(data, "x_dict") and node_type in data.x_dict:
            hetero[node_type].x = data.x_dict[node_type]
        if hasattr(data, "y_dict") and node_type in data.y_dict:
            hetero[node_type].y = data.y_dict[node_type].view(-1)
        if hasattr(data, "node_year") and node_type in data.node_year:
            hetero[node_type].year = data.node_year[node_type].view(-1)

    for edge_type, edge_index in data.edge_index_dict.items():
        hetero[edge_type].edge_index = edge_index

    return hetero


def load_ogbn_mag(root: str) -> Tuple[Any, Dict[str, torch.Tensor], int]:
    """Load ogbn-mag with clear dependency and data errors."""
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'ogb'. Install it with `pip install ogb` before "
            "running train_hh_mag.py."
        ) from exc

    original_torch_load = torch.load
    torch.load = _torch_load_with_weights_only_false
    try:
        dataset = PygNodePropPredDataset(name="ogbn-mag", root=root)
    finally:
        torch.load = original_torch_load

    data = _legacy_ogb_data_to_heterodata(dataset[0])
    split_idx = dataset.get_idx_split()
    if isinstance(split_idx.get("train"), Mapping):
        split_idx = {key: value["paper"].view(-1).long() for key, value in split_idx.items()}
    else:
        split_idx = {key: value.view(-1).long() for key, value in split_idx.items()}

    return data, split_idx, int(dataset.num_classes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HH/HGAMLP-HOPE ogbn-mag ablation")

    parser.add_argument("--method", choices=["hh", "hh_smp", "hh_ump", "hh_gsmp"], required=True)
    parser.add_argument("--dataset", default="ogbn-mag")
    parser.add_argument("--root", default="./data")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-hops", type=int, default=6)
    parser.add_argument("--use-label-feats", action="store_true")
    parser.add_argument("--use-precomputed", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")

    parser.add_argument(
        "--bucket-style",
        choices=["auto", "coarse", "yearly"],
        default="auto",
        help="Temporal bucket style. auto = yearly for GSMP, coarse otherwise.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16_384,
        help="Paper-node mini-batch size. Use <=0 for full-batch training.",
    )
    parser.add_argument(
        "--max-train-nodes",
        type=int,
        default=0,
        help="Use only the first N official train nodes; 0 keeps the full split.",
    )
    parser.add_argument(
        "--max-valid-nodes",
        type=int,
        default=0,
        help="Use only the first N official validation nodes; 0 keeps the full split.",
    )
    parser.add_argument(
        "--max-test-nodes",
        type=int,
        default=0,
        help="Use only the first N official test nodes; 0 keeps the full split.",
    )
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--no-hope", action="store_true")

    return parser.parse_args()


def _effective_bucket_style(args: argparse.Namespace) -> str:
    if args.bucket_style != "auto":
        return args.bucket_style
    return "yearly" if args.method == "hh_gsmp" else "coarse"


def _validate_ogbn_mag_data(
    data: Any,
    split_idx: Mapping[str, torch.Tensor],
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if "paper" not in getattr(data, "node_types", []):
        raise ValueError("Loaded data does not contain node type 'paper'.")

    paper = data["paper"]
    x = getattr(paper, "x", None)
    y = getattr(paper, "y", None)
    years = getattr(paper, "year", None)
    if x is None:
        raise ValueError(
            "Missing paper node features (data['paper'].x). The default pipeline "
            "uses propagated paper features only and does not synthesize non-paper "
            "features."
        )
    if y is None:
        raise ValueError("Missing paper labels (data['paper'].y).")
    if years is None:
        raise ValueError("Missing paper publication years (data['paper'].year).")

    labels = y.view(-1).cpu().long()
    years = years.view(-1).cpu()
    num_papers = x.shape[0]
    if labels.numel() != num_papers:
        raise ValueError(
            f"Paper label count {labels.numel()} does not match feature rows {num_papers}."
        )
    if years.numel() != num_papers:
        raise ValueError(
            f"Paper year count {years.numel()} does not match feature rows {num_papers}."
        )
    if labels.min().item() < 0 or labels.max().item() >= num_classes:
        raise ValueError(
            f"Paper labels are outside [0, {num_classes - 1}]. "
            f"Observed min={labels.min().item()}, max={labels.max().item()}."
        )

    validate_split(split_idx)
    for split_name, indices in split_idx.items():
        if indices.min().item() < 0 or indices.max().item() >= num_papers:
            raise ValueError(f"Split '{split_name}' contains out-of-range paper ids.")

    if ("paper", "cites", "paper") not in getattr(data, "edge_types", []):
        paper_edges = [
            edge_type
            for edge_type in getattr(data, "edge_types", [])
            if edge_type[0] == "paper" and edge_type[2] == "paper"
        ]
        if not paper_edges:
            raise ValueError("Missing paper-paper citation edges.")

    return labels, years


def _limit_split(
    split_idx: Mapping[str, torch.Tensor],
    max_train_nodes: int,
    max_valid_nodes: int,
    max_test_nodes: int,
) -> Dict[str, torch.Tensor]:
    """Optionally reduce official splits for fast smoke/mini experiments."""
    limits = {
        "train": max_train_nodes,
        "valid": max_valid_nodes,
        "test": max_test_nodes,
    }
    limited: Dict[str, torch.Tensor] = {}
    for split_name, indices in split_idx.items():
        limit = limits.get(split_name, 0)
        if limit is not None and limit > 0:
            if limit > indices.numel():
                raise ValueError(
                    f"--max-{split_name}-nodes={limit} exceeds official "
                    f"{split_name} split size {indices.numel()}."
                )
            limited[split_name] = indices[:limit].clone()
        else:
            limited[split_name] = indices.clone()
    validate_split(limited)
    return limited


def _cache_path(args: argparse.Namespace, bucket_style: str) -> Path:
    precompute_dir = Path(args.output_dir) / "precomputed"
    return precompute_dir / (
        f"{args.dataset}_{args.method}_h{args.num_hops}_bucket{bucket_style}.pt"
    )


def _load_cached_features(path: Path) -> Dict[str, torch.Tensor]:
    features = torch.load(path, map_location="cpu")
    if not isinstance(features, dict):
        raise ValueError(f"Cached propagation file {path} did not contain a dict.")
    for key, value in features.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Cached feature '{key}' is not a torch.Tensor.")
    return features


def _build_or_load_features(
    args: argparse.Namespace,
    data: Any,
    split_idx: Mapping[str, torch.Tensor],
    years: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    bucket_style = _effective_bucket_style(args)
    cache_path = _cache_path(args, bucket_style)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if args.use_precomputed and cache_path.exists() and not args.force_recompute:
        print(f"[CACHE] Loading propagated features from {cache_path}", flush=True)
        return _load_cached_features(cache_path)

    print(
        f"[PROPAGATION] method={args.method} num_hops={args.num_hops} "
        f"bucket_style={bucket_style}",
        flush=True,
    )
    if args.method == "hh":
        features = build_base_propagation(data=data, num_hops=args.num_hops)
    elif args.method == "hh_smp":
        features = build_smp_propagation(
            data=data,
            split_idx=split_idx,
            years=years,
            num_hops=args.num_hops,
            bucket_style=bucket_style,
        )
    elif args.method == "hh_ump":
        features = build_ump_propagation(
            data=data,
            split_idx=split_idx,
            years=years,
            num_hops=args.num_hops,
            bucket_style=bucket_style,
        )
    elif args.method == "hh_gsmp":
        features = build_gsmp_propagation(
            data=data,
            split_idx=split_idx,
            years=years,
            num_hops=args.num_hops,
            bucket_style=bucket_style,
        )
    else:
        raise ValueError(f"Unknown method {args.method}")

    torch.save(features, cache_path)
    print(f"[CACHE] Saved propagated features to {cache_path}", flush=True)
    return features


def _add_label_features(
    features: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    num_classes: int,
) -> None:
    label_features = torch.zeros(labels.numel(), num_classes, dtype=torch.float32)
    label_features[train_idx] = F.one_hot(labels[train_idx], num_classes=num_classes).float()
    features["paper__train_label_onehot"] = label_features


def _validate_features(features: Mapping[str, torch.Tensor], num_papers: int) -> None:
    if "paper" not in features:
        raise ValueError("feature_dict must contain the raw 'paper' channel.")
    for key, value in features.items():
        if value.ndim != 2:
            raise ValueError(f"Feature '{key}' must be 2D, got {tuple(value.shape)}.")
        if value.shape[0] != num_papers:
            raise ValueError(
                f"Feature '{key}' has {value.shape[0]} rows; expected {num_papers}."
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Feature '{key}' contains NaN or Inf values.")


def _iter_batches(
    indices: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> Iterable[torch.Tensor]:
    indices = indices.detach().cpu().long().view(-1)
    if batch_size <= 0:
        yield indices
        return

    if shuffle:
        perm = torch.randperm(indices.numel())
        indices = indices[perm]

    for start in range(0, indices.numel(), batch_size):
        yield indices[start : start + batch_size]


def _batch_features(
    features: Mapping[str, torch.Tensor],
    batch_idx: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.index_select(0, batch_idx).to(device, non_blocking=True)
        for key, value in features.items()
    }


def train_one_epoch(
    model: HHModel,
    features: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch_idx in _iter_batches(train_idx, batch_size=batch_size, shuffle=True):
        batch = _batch_features(features, batch_idx, device)
        y = labels.index_select(0, batch_idx).to(device)
        logits = model(batch)
        if logits.shape[0] != y.shape[0]:
            raise ValueError(
                f"Shape mismatch: logits rows={logits.shape[0]}, labels={y.shape[0]}."
            )
        loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"NaN/Inf loss detected: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.numel()
        total_examples += y.numel()

    if total_examples == 0:
        raise ValueError("No training examples were processed.")
    return total_loss / total_examples


@torch.no_grad()
def evaluate_split(
    model: HHModel,
    features: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch_idx in _iter_batches(indices, batch_size=batch_size, shuffle=False):
        batch = _batch_features(features, batch_idx, device)
        y = labels.index_select(0, batch_idx).to(device)
        logits = model(batch)
        if logits.shape[0] != y.shape[0]:
            raise ValueError(
                f"Shape mismatch: logits rows={logits.shape[0]}, labels={y.shape[0]}."
            )
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    if total == 0:
        raise ValueError("Cannot evaluate an empty split.")
    return correct / total


def evaluate_all(
    model: HHModel,
    features: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    split_idx: Mapping[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> Tuple[float, float, float]:
    train_acc = evaluate_split(model, features, labels, split_idx["train"], device, batch_size)
    valid_acc = evaluate_split(model, features, labels, split_idx["valid"], device, batch_size)
    test_acc = evaluate_split(model, features, labels, split_idx["test"], device, batch_size)
    return train_acc, valid_acc, test_acc


def save_checkpoint(
    checkpoint_path: Path,
    model: HHModel,
    args: argparse.Namespace,
    input_dims: Mapping[str, int],
    epoch: int,
    best_valid_acc: float,
    best_test_acc: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "input_dims": dict(input_dims),
            "epoch": epoch,
            "best_valid_acc": best_valid_acc,
            "best_test_acc": best_test_acc,
        },
        checkpoint_path,
    )


def main() -> None:
    args = parse_args()
    if args.dataset != "ogbn-mag":
        raise ValueError("This pipeline currently supports --dataset ogbn-mag only.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    results_dir = output_dir / "results"
    checkpoints_dir = output_dir / "checkpoints"
    ensure_dirs([logs_dir, results_dir, checkpoints_dir, output_dir / "precomputed"])

    print(
        f"[START][METHOD={args.method}][SEED={args.seed}] device={device} "
        f"epochs={args.epochs} batch_size={args.batch_size}",
        flush=True,
    )
    print(f"[DATA] Loading {args.dataset} from {args.root}", flush=True)
    data, split_idx, num_classes = load_ogbn_mag(args.root)
    labels, years = _validate_ogbn_mag_data(data, split_idx, num_classes)

    official_split_sizes = {key: int(value.numel()) for key, value in split_idx.items()}
    split_idx = _limit_split(
        split_idx=split_idx,
        max_train_nodes=args.max_train_nodes,
        max_valid_nodes=args.max_valid_nodes,
        max_test_nodes=args.max_test_nodes,
    )
    active_split_sizes = {key: int(value.numel()) for key, value in split_idx.items()}
    print(
        f"[DATA] papers={labels.numel()} classes={num_classes} "
        f"official_train={official_split_sizes['train']} "
        f"official_valid={official_split_sizes['valid']} "
        f"official_test={official_split_sizes['test']}",
        flush=True,
    )
    print(
        f"[DATA] active_train={active_split_sizes['train']} "
        f"active_valid={active_split_sizes['valid']} "
        f"active_test={active_split_sizes['test']}",
        flush=True,
    )
    print(
        "[DATA] Non-paper node features are not synthesized in this pipeline; "
        "HH uses paper features propagated over paper citation channels.",
        flush=True,
    )

    features = _build_or_load_features(args, data, split_idx, years)
    if args.use_label_feats:
        _add_label_features(features, labels, split_idx["train"], num_classes)
        print("[FEATURES] Added train-only one-hot label features.", flush=True)

    _validate_features(features, labels.numel())
    input_dims = {key: int(value.shape[1]) for key, value in features.items()}
    print(
        f"[FEATURES] channels={len(features)} "
        f"dims={','.join(f'{key}:{dim}' for key, dim in input_dims.items())}",
        flush=True,
    )

    model = HHModel(
        input_dims=input_dims,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        use_hope=not args.no_hope,
        num_experts=args.num_experts,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    metric_path = results_dir / f"{args.method}_seed{args.seed}_metrics.csv"
    csv_logger = CSVLogger(metric_path, METRIC_COLUMNS, overwrite=True)
    checkpoint_path = checkpoints_dir / f"{args.method}_seed{args.seed}_best.pt"

    timer = Timer()
    best_valid_acc = -1.0
    best_test_acc = -1.0
    final_train_acc = 0.0
    final_valid_acc = 0.0
    final_test_acc = 0.0
    last_loss = float("nan")

    for epoch in range(1, args.epochs + 1):
        last_loss = train_one_epoch(
            model=model,
            features=features,
            labels=labels,
            train_idx=split_idx["train"],
            optimizer=optimizer,
            device=device,
            batch_size=args.batch_size,
        )

        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if not should_eval:
            continue

        train_acc, valid_acc, test_acc = evaluate_all(
            model=model,
            features=features,
            labels=labels,
            split_idx=split_idx,
            device=device,
            batch_size=args.batch_size,
        )
        final_train_acc = train_acc
        final_valid_acc = valid_acc
        final_test_acc = test_acc

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_test_acc = test_acc
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                args=args,
                input_dims=input_dims,
                epoch=epoch,
                best_valid_acc=best_valid_acc,
                best_test_acc=best_test_acc,
            )

        elapsed = timer.elapsed()
        print(
            format_eval_line(
                method=args.method,
                seed=args.seed,
                epoch=epoch,
                train_acc=train_acc,
                valid_acc=valid_acc,
                test_acc=test_acc,
                loss=last_loss,
                best_valid=best_valid_acc,
                best_test=best_test_acc,
                elapsed=elapsed,
            ),
            flush=True,
        )
        csv_logger.append(
            {
                "method": args.method,
                "seed": args.seed,
                "epoch": epoch,
                "loss": f"{last_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "valid_acc": f"{valid_acc:.6f}",
                "test_acc": f"{test_acc:.6f}",
                "best_valid_acc": f"{best_valid_acc:.6f}",
                "best_test_acc": f"{best_test_acc:.6f}",
                "elapsed_sec": f"{elapsed:.3f}",
            }
        )

    if best_valid_acc < 0:
        raise RuntimeError("Training finished without any evaluation step.")

    total_time = timer.elapsed()
    summary_row = {
        "method": args.method,
        "seed": args.seed,
        "best_valid_acc": f"{best_valid_acc:.6f}",
        "best_test_acc": f"{best_test_acc:.6f}",
        "final_train_acc": f"{final_train_acc:.6f}",
        "final_valid_acc": f"{final_valid_acc:.6f}",
        "final_test_acc": f"{final_test_acc:.6f}",
    }
    upsert_summary_row(results_dir / "summary.csv", summary_row)

    print(
        f"[FINAL][METHOD={args.method}][SEED={args.seed}] "
        f"best_valid={best_valid_acc:.4f} best_test={best_test_acc:.4f} "
        f"final_valid={final_valid_acc:.4f} final_test={final_test_acc:.4f} "
        f"total_time={total_time:.1f}s",
        flush=True,
    )
    print(f"[CHECKPOINT] {checkpoint_path}", flush=True)
    print(f"[METRICS] {metric_path}", flush=True)
    print(f"[SUMMARY] {results_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
