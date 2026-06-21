#!/usr/bin/env python3
"""Mini-batch GCN vs first-layer GSMP-GCN on Pokec temporal split."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

from data_utils import (
    add_self_loops_to_directed,
    class_distribution,
    edge_index_diagnostics,
    gpu_memory_string,
    load_temporal_tensors,
    log,
    normalize_features,
    read_json,
    set_seed,
)
from eval_utils import evaluate_full_graph, format_epoch_log
from gsmp import (
    compute_gcn_norm,
    compute_gsmp_edge_weights,
    gsmp_weight_stats,
    make_sparse_adj,
    sanity_check_gsmp_identity,
)
from models import TemporalGCN


SPLITS = ("train", "valid", "test")


def parse_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        log("[device] CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        return args.seeds[: args.runs]
    return list(range(args.runs))


def print_quality_checks(
    metadata: Dict,
    y: torch.Tensor,
    split: Dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    log("[quality] dataset/training checks")
    log(f"  nodes: {metadata['num_nodes']:,}")
    log(f"  directed raw edges: {metadata['num_directed_edges']:,}")
    log(f"  training edges: {edge_index.size(1):,}")
    log(f"  feature dim: {metadata['num_features']}")
    log(f"  classes: {metadata['num_classes']}")
    log(f"  min registration year: {metadata['min_registration_year']}")
    log(f"  max registration year: {metadata['max_registration_year']}")
    for name in SPLITS:
        idx = split[name]
        log(
            f"  {name}: nodes={idx.numel():,}, labels="
            f"{class_distribution(y, idx, metadata['num_classes'])}"
        )
    log(f"  split edge counts directed: {metadata.get('temporal_stats', {}).get('split_directed_counts')}")
    log(f"  graph for training: {'directed ablation' if args.use_directed else 'undirected+self-loop'}")
    log(f"  GSMP enabled: {args.method == 'gcn_gsmp_first'}")
    log(f"  GSMP first-layer-only: {args.method == 'gcn_gsmp_first'}")
    log(f"  GSMP recompute per batch: {args.gsmp_recompute_per_batch}")
    log(f"  GSMP precompute global approximation: {args.gsmp_precompute_global}")
    if not args.use_directed and metadata.get("main_training_graph") != "undirected_self_loop":
        raise RuntimeError("Main training graph metadata does not say undirected_self_loop.")
    if not args.use_directed and "undirected" not in metadata.get("main_training_graph", ""):
        raise RuntimeError("Main training graph was accidentally left directed.")


def iter_induced_batches(
    edge_index: torch.Tensor,
    num_nodes: int,
    batch_size: int,
    generator: torch.Generator,
) -> Iterable[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    perm = torch.randperm(num_nodes, generator=generator)
    num_batches = math.ceil(num_nodes / batch_size)
    group = torch.full((num_nodes,), -1, dtype=torch.int32)
    for batch_id in range(num_batches):
        nodes = perm[batch_id * batch_size : min((batch_id + 1) * batch_size, num_nodes)]
        group[nodes] = int(batch_id)

    src_group = group[edge_index[0]]
    dst_group = group[edge_index[1]]
    same_group = (src_group == dst_group) & (src_group >= 0)
    edge_pos_all = torch.where(same_group)[0]
    edge_batch_all = src_group[edge_pos_all].long()
    order = torch.argsort(edge_batch_all)
    edge_pos_sorted = edge_pos_all[order]
    edge_batch_sorted = edge_batch_all[order]

    for batch_id in range(num_batches):
        nodes = perm[batch_id * batch_size : min((batch_id + 1) * batch_size, num_nodes)]
        batch_tensor = torch.tensor(batch_id, dtype=edge_batch_sorted.dtype)
        left = torch.searchsorted(edge_batch_sorted, batch_tensor, right=False)
        right = torch.searchsorted(edge_batch_sorted, batch_tensor, right=True)
        edge_pos = edge_pos_sorted[left:right]
        edges_global = edge_index[:, edge_pos]
        local_map = torch.full((num_nodes,), -1, dtype=torch.long)
        local_map[nodes] = torch.arange(nodes.numel(), dtype=torch.long)
        edge_local = local_map[edges_global]
        if (edge_local < 0).any():
            raise RuntimeError("Induced subgraph relabeling failed.")
        yield batch_id, nodes.long(), edge_local.long().contiguous(), edge_pos.long()


def build_sparse_adjs(
    edge_index: torch.Tensor,
    year_idx: torch.Tensor,
    num_nodes: int,
    num_timestamps: int,
    method: str,
    device: torch.device,
    gsmp_include_self_loops: bool,
    gsmp_weighted_degree: bool,
    precomputed_gsmp_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, float]]]:
    edge_index = edge_index.to(device, non_blocking=True)
    year_idx = year_idx.to(device, non_blocking=True)

    regular_norm = compute_gcn_norm(edge_index, num_nodes)
    adj_rest = make_sparse_adj(edge_index, regular_norm, num_nodes)
    gsmp_stats = None

    if method == "gcn_gsmp_first":
        if precomputed_gsmp_weight is None:
            gsmp_weight = compute_gsmp_edge_weights(
                edge_index,
                year_idx,
                num_nodes=num_nodes,
                num_timestamps=num_timestamps,
                include_self_loops=gsmp_include_self_loops,
            )
        else:
            gsmp_weight = precomputed_gsmp_weight.to(device, dtype=torch.float32, non_blocking=True)
        gsmp_stats = gsmp_weight_stats(gsmp_weight)
        degree_weight = gsmp_weight if gsmp_weighted_degree else None
        first_norm = compute_gcn_norm(
            edge_index,
            num_nodes,
            message_weight=gsmp_weight,
            degree_weight=degree_weight,
        )
        adj_first = make_sparse_adj(edge_index, first_norm, num_nodes)
    else:
        adj_first = adj_rest

    return adj_first, adj_rest, gsmp_stats


def checkpoint_dir(args: argparse.Namespace) -> Path:
    return args.checkpoint_dir if args.checkpoint_dir is not None else Path(args.results_dir) / "checkpoints"


def checkpoint_path(args: argparse.Namespace, method: str, seed: int, kind: str) -> Path:
    return checkpoint_dir(args) / f"{method}_seed{seed}_{kind}.pt"


def save_checkpoint(
    path: Path,
    model: TemporalGCN,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    seed: int,
    best_valid: float,
    test_at_best: float,
    best_epoch: int,
    evals_without_improvement: int,
    final_metrics: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "seed": seed,
            "method": args.method,
            "best_valid": best_valid,
            "test_at_best": test_at_best,
            "best_epoch": best_epoch,
            "evals_without_improvement": evals_without_improvement,
            "final_metrics": final_metrics,
            "args": vars(args),
        },
        path,
    )


def load_resume_checkpoint(
    args: argparse.Namespace,
    seed: int,
    model: TemporalGCN,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[int, float, float, int, int, Dict[str, float]]:
    if args.resume_path is not None:
        path = args.resume_path
    else:
        path = checkpoint_path(args, args.method, seed, "last")

    if not path.exists():
        if args.resume_from_checkpoint:
            log(f"[resume] no checkpoint found at {path}; starting seed={seed} from scratch")
        return 1, -1.0, -1.0, 0, 0, {"train": float("nan"), "valid": float("nan"), "test": float("nan")}

    log(f"[resume] loading {path}")
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("method") not in (None, args.method):
        raise RuntimeError(f"Checkpoint method {ckpt.get('method')} does not match requested {args.method}")
    if ckpt.get("seed") not in (None, seed):
        raise RuntimeError(f"Checkpoint seed {ckpt.get('seed')} does not match requested {seed}")

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    epoch = int(ckpt.get("epoch", 0))
    start_epoch = epoch + 1
    best_valid = float(ckpt.get("best_valid", ckpt.get("valid_acc", -1.0)))
    test_at_best = float(ckpt.get("test_at_best", ckpt.get("test_acc", -1.0)))
    best_epoch = int(ckpt.get("best_epoch", epoch if best_valid >= 0 else 0))
    evals_without_improvement = int(ckpt.get("evals_without_improvement", 0))
    final_metrics = ckpt.get("final_metrics") or {"train": float("nan"), "valid": float("nan"), "test": float("nan")}
    log(
        "[resume] "
        f"seed={seed} resumes at epoch={start_epoch}; best_epoch={best_epoch}; "
        f"best_valid={best_valid:.6f}; test_at_best={test_at_best:.6f}"
    )
    return start_epoch, best_valid, test_at_best, best_epoch, evals_without_improvement, final_metrics


def evaluate(
    model: TemporalGCN,
    x: torch.Tensor,
    y: torch.Tensor,
    split: Dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    year_idx: torch.Tensor,
    num_timestamps: int,
    args: argparse.Namespace,
    eval_device: torch.device,
) -> Dict[str, float]:
    adj_first, adj_rest, _ = build_sparse_adjs(
        edge_index,
        year_idx,
        num_nodes=x.size(0),
        num_timestamps=num_timestamps,
        method=args.method,
        device=eval_device,
        gsmp_include_self_loops=args.gsmp_include_self_loops,
        gsmp_weighted_degree=args.gsmp_weighted_degree,
    )
    return evaluate_full_graph(
        model,
        x,
        y,
        split,
        adj_first,
        adj_rest,
        device=eval_device,
    )


def train_one_seed(
    seed: int,
    x: torch.Tensor,
    y: torch.Tensor,
    year_idx: torch.Tensor,
    split_group: torch.Tensor,
    split: Dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    metadata: Dict,
    args: argparse.Namespace,
) -> Dict:
    set_seed(seed)
    device = parse_device(args.device)
    eval_device = parse_device(args.eval_device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    model = TemporalGCN(
        in_channels=x.size(1),
        hidden_channels=args.hidden_channels,
        out_channels=metadata["num_classes"],
        num_layers=args.num_layers,
        dropout=args.dropout,
        in_dropout=args.in_dropout,
        use_bn=args.bn,
        use_residual=args.res,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    num_timestamps = int(year_idx.max().item() + 1)

    global_gsmp_weight = None
    if args.method == "gcn_gsmp_first" and args.gsmp_precompute_global:
        log("[gsmp] precomputing global GSMP weights for approximate mini-batch slicing")
        global_gsmp_weight = compute_gsmp_edge_weights(
            edge_index,
            year_idx,
            num_nodes=x.size(0),
            num_timestamps=num_timestamps,
            include_self_loops=args.gsmp_include_self_loops,
        )
        log(f"[gsmp] global weight stats: {gsmp_weight_stats(global_gsmp_weight)}")

    start_epoch = 1
    best_valid = -1.0
    test_at_best = -1.0
    best_epoch = 0
    evals_without_improvement = 0
    completed_epochs = 0
    final_metrics = {"train": float("nan"), "valid": float("nan"), "test": float("nan")}
    if args.resume_from_checkpoint or args.resume_path is not None:
        (
            start_epoch,
            best_valid,
            test_at_best,
            best_epoch,
            evals_without_improvement,
            final_metrics,
        ) = load_resume_checkpoint(args, seed, model, optimizer, scaler, device)
        if start_epoch > args.epochs:
            log(f"[resume] checkpoint epoch is already >= requested epochs ({args.epochs}); skipping training")

    start_time = time.time()
    max_gpu_mem = 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        completed_epochs = epoch
        model.train()
        epoch_loss_sum = 0.0
        epoch_train_nodes = 0
        generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)

        for batch_id, nodes, edge_local, edge_pos in iter_induced_batches(
            edge_index,
            num_nodes=x.size(0),
            batch_size=args.batch_size,
            generator=generator,
        ):
            train_local = torch.where(split_group[nodes] == 0)[0].long()
            if train_local.numel() == 0:
                continue

            x_batch = x[nodes].to(device, non_blocking=True)
            y_batch = y[nodes].to(device, non_blocking=True)
            year_batch = year_idx[nodes].to(device, non_blocking=True)
            precomputed = None
            if global_gsmp_weight is not None:
                precomputed = global_gsmp_weight[edge_pos]

            try:
                adj_first, adj_rest, gsmp_stats = build_sparse_adjs(
                    edge_local,
                    year_batch,
                    num_nodes=nodes.numel(),
                    num_timestamps=num_timestamps,
                    method=args.method,
                    device=device,
                    gsmp_include_self_loops=args.gsmp_include_self_loops,
                    gsmp_weighted_degree=args.gsmp_weighted_degree,
                    precomputed_gsmp_weight=precomputed,
                )
                if epoch == 1 and batch_id < args.print_gsmp_batches and gsmp_stats is not None:
                    log(f"[gsmp] epoch=1 batch={batch_id} weight_stats={gsmp_stats}")

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    logits = model(x_batch, adj_first, adj_rest)
                    loss = F.cross_entropy(logits[train_local.to(device)], y_batch[train_local.to(device)])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss_sum += float(loss.item()) * int(train_local.numel())
                epoch_train_nodes += int(train_local.numel())
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and device.type == "cuda":
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA OOM during mini-batch training. Try --batch-size 400000, "
                        "250000, or 100000; also consider --eval-device cpu."
                    ) from exc
                raise

        avg_loss = epoch_loss_sum / max(epoch_train_nodes, 1)

        should_eval = epoch == args.epochs or (
            epoch > args.eval_start_epoch and epoch % args.eval_step == 0
        )
        if should_eval:
            metrics = evaluate(
                model,
                x,
                y,
                split,
                edge_index,
                year_idx,
                num_timestamps,
                args,
                eval_device,
            )
            final_metrics = metrics
            if metrics["valid"] > best_valid + args.early_stop_min_delta:
                best_valid = metrics["valid"]
                test_at_best = metrics["test"]
                best_epoch = epoch
                evals_without_improvement = 0
                if args.save_checkpoints:
                    save_checkpoint(
                        checkpoint_path(args, args.method, seed, "best"),
                        model,
                        optimizer,
                        scaler,
                        epoch,
                        seed,
                        best_valid,
                        test_at_best,
                        best_epoch,
                        evals_without_improvement,
                        final_metrics,
                        args,
                    )
            else:
                evals_without_improvement += 1
            if args.save_checkpoints:
                save_checkpoint(
                    checkpoint_path(args, args.method, seed, "last"),
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    seed,
                    best_valid,
                    test_at_best,
                    best_epoch,
                    evals_without_improvement,
                    final_metrics,
                    args,
                )
            if device.type == "cuda":
                max_gpu_mem = max(max_gpu_mem, torch.cuda.max_memory_allocated() / (1024**3))
            log(
                format_epoch_log(
                    epoch,
                    avg_loss,
                    metrics,
                    best_valid,
                    test_at_best,
                    best_epoch,
                    device,
                    start_time,
                )
            )
            if (
                args.early_stop_patience > 0
                and epoch >= args.min_epochs
                and evals_without_improvement >= args.early_stop_patience
            ):
                log(
                    "[early-stop] "
                    f"stopping at epoch={epoch}; best_epoch={best_epoch}; "
                    f"no valid improvement for {evals_without_improvement} eval checks"
                )
                break

    total_time = time.time() - start_time
    if device.type == "cuda":
        max_gpu_mem = max(max_gpu_mem, torch.cuda.max_memory_allocated() / (1024**3))
    if args.save_checkpoints and completed_epochs > 0:
        save_checkpoint(
            checkpoint_path(args, args.method, seed, "last"),
            model,
            optimizer,
            scaler,
            completed_epochs,
            seed,
            best_valid,
            test_at_best,
            best_epoch,
            evals_without_improvement,
            final_metrics,
            args,
        )
    result = {
        "method": args.method,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_valid_acc": best_valid,
        "test_acc_at_best_valid": test_at_best,
        "final_train_acc": final_metrics["train"],
        "final_valid_acc": final_metrics["valid"],
        "final_test_acc": final_metrics["test"],
        "total_time_sec": total_time,
        "max_gpu_mem_gb": max_gpu_mem,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "in_dropout": args.in_dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "completed_epochs": completed_epochs,
        "eval_step": args.eval_step,
        "eval_start_epoch": args.eval_start_epoch,
        "min_epochs": args.min_epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "use_directed": args.use_directed,
        "gsmp_weighted_degree": args.gsmp_weighted_degree,
        "gsmp_include_self_loops": args.gsmp_include_self_loops,
        "gsmp_precompute_global": args.gsmp_precompute_global,
    }
    log(
        "FINAL "
        f"method={result['method']} seed={seed} best_epoch={best_epoch} "
        f"best_valid_acc={best_valid:.6f} test_acc_at_best_valid={test_at_best:.6f} "
        f"final_test_acc={final_metrics['test']:.6f} total_time={total_time:.1f}s "
        f"max_gpu_mem={max_gpu_mem:.3f}GB"
    )
    return result


def append_results(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        for key in rows[0].keys():
            if key not in fieldnames:
                fieldnames.append(key)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(row)
            for row in rows:
                writer.writerow(row)
    else:
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    log(f"[results] appended {len(rows)} row(s) to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pokec temporal GCN/GSMP.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/pokec_temporal"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--method", choices=("gcn", "gcn_gsmp_first"), required=True)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--in-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=550_000)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--eval-step", type=int, default=10)
    parser.add_argument(
        "--eval-start-epoch",
        type=int,
        default=0,
        help="Only evaluate when epoch > this value, plus always at final epoch.",
    )
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many eval checks without validation improvement. Disabled at 0.",
    )
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--res", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-directed", action="store_true")
    parser.add_argument("--no-normalize-features", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--resume-path", type=Path, default=None)
    parser.add_argument("--run-gsmp-sanity-check", action="store_true")
    parser.add_argument("--gsmp-recompute-per-batch", action="store_true", default=True)
    parser.add_argument("--gsmp-precompute-global", action="store_true")
    parser.add_argument("--gsmp-include-self-loops", action="store_true")
    parser.add_argument("--gsmp-weighted-degree", action="store_true")
    parser.add_argument("--print-gsmp-batches", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.run_gsmp_sanity_check:
        diff = sanity_check_gsmp_identity("cpu")
        log(f"[sanity] GSMP identity vs PyG GCNConv max_abs_diff={diff:.8g}")

    metadata = read_json(args.data_dir / "metadata.json")
    tensors = load_temporal_tensors(args.data_dir)
    x = tensors["x"].float()
    if not args.no_normalize_features:
        x = normalize_features(x)
    y = tensors["y"].long()
    year_idx = tensors["year_idx"].long()
    split_group = tensors["split_group"].long()
    split = {
        "train": tensors["train_idx"].long(),
        "valid": tensors["valid_idx"].long(),
        "test": tensors["test_idx"].long(),
    }
    if args.use_directed:
        edge_index = add_self_loops_to_directed(tensors["edge_index_directed"].long(), x.size(0))
    else:
        edge_index = tensors["edge_index_undirected_self_loop"].long()
    edge_index_diagnostics(edge_index, x.size(0), "training_edge_index")

    print_quality_checks(metadata, y, split, edge_index, args)
    seeds = resolve_seeds(args)
    all_rows = []
    for seed in seeds:
        row = train_one_seed(
            seed,
            x,
            y,
            year_idx,
            split_group,
            split,
            edge_index,
            metadata,
            args,
        )
        all_rows.append(row)
        append_results(args.results_dir / "pokec_temporal_results.csv", [row])

    df = pd.DataFrame(all_rows)
    log("[summary]")
    for column in ("best_valid_acc", "test_acc_at_best_valid", "final_test_acc", "max_gpu_mem_gb", "total_time_sec"):
        log(f"  {column}: {df[column].mean():.6f} +/- {df[column].std(ddof=0):.6f}")


if __name__ == "__main__":
    main()
