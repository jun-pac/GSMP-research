from __future__ import annotations

import argparse
import csv
import json
import os
import random
import socket
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from data_loading import ArxivBundle, load_ogbn_arxiv
from edge_preprocessing import EdgePreprocessResult, preprocess_edges, stats_as_log_lines
from models import WeightedGraphSAGE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SimTeG/TAPE feature reuse + weighted GraphSAGE SMP/UMP/GSMP on ogbn-arxiv."
    )
    parser.add_argument("--mode", choices=["baseline", "smp", "ump", "gsmp"], default="baseline")
    parser.add_argument("--features_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--aggregation_chunk_size", type=int, default=200000)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=2e-6)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--directed", action="store_true", help="Do not symmetrize the ogbn-arxiv edge_index.")
    parser.add_argument(
        "--no_auto_features",
        action="store_true",
        help="Disable automatic local SimTeG/TAPE feature discovery when --features_path is omitted.",
    )
    parser.add_argument("--training_mode", choices=["full", "mini"], default="full")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument(
        "--num_neighbors",
        type=str,
        default="15,10,5",
        help="Comma-separated NeighborLoader fanouts used only with --training_mode mini.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else repo_root / "data"
    save_dir = Path(args.save_dir).expanduser().resolve()
    results_dir = save_dir / "results"
    checkpoints_dir = save_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print_header(args=args, data_root=data_root, save_dir=save_dir)

    bundle = load_ogbn_arxiv(
        data_root=data_root,
        repo_root=repo_root,
        features_path=args.features_path,
        allow_auto_features=not args.no_auto_features,
    )
    edge_result = preprocess_edges(
        edge_index=bundle.data.edge_index,
        node_year=bundle.node_year,
        num_nodes=bundle.data.num_nodes,
        mode=args.mode,
        make_undirected=not args.directed,
    )
    print_preprocess_stats(bundle, edge_result)

    if args.dry_run:
        print("DRY RUN: data loaded and graph/weights constructed; exiting before training.", flush=True)
        write_dry_run_stats(results_dir, args, bundle, edge_result)
        return

    device = resolve_device(args.device)
    print_device_stats(device)

    summaries = []
    for run_idx, seed in enumerate(range(args.seed, args.seed + args.runs), start=1):
        print(f"Starting run {run_idx}/{args.runs} with seed={seed}", flush=True)
        summary = run_one_seed(
            args=args,
            seed=seed,
            bundle=bundle,
            edge_result=edge_result,
            device=device,
            results_dir=results_dir,
            checkpoints_dir=checkpoints_dir,
        )
        summaries.append(summary)
        write_run_summary(results_dir, summary)
        update_summary_json(results_dir)

    print("Completed runs:", flush=True)
    for summary in summaries:
        print(
            f"  mode={summary['mode']} seed={summary['seed']} "
            f"best_valid={summary['best_valid_acc']:.6f} "
            f"test_at_best_valid={summary['test_acc_at_best_valid']:.6f} "
            f"epochs={summary['total_epochs_run']} runtime={format_duration(summary['runtime_sec'])}",
            flush=True,
        )


def print_header(args: argparse.Namespace, data_root: Path, save_dir: Path) -> None:
    print("=== ogbn-arxiv SimTeG/TAPE GraphSAGE temporal experiment ===", flush=True)
    print(f"host={socket.gethostname()}", flush=True)
    print(f"date={datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(f"python={sys.version.split()[0]}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    try:
        import torch_geometric

        print(f"torch_geometric={torch_geometric.__version__}", flush=True)
    except Exception as exc:
        print(f"torch_geometric_version_unavailable={exc}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"data_root={data_root}", flush=True)
    print(f"save_dir={save_dir}", flush=True)
    print(f"args={json.dumps(vars(args), sort_keys=True)}", flush=True)


def print_preprocess_stats(bundle: ArxivBundle, edge_result: EdgePreprocessResult) -> None:
    print("=== preprocessing statistics ===", flush=True)
    for line in stats_as_log_lines(edge_result.stats):
        print(line, flush=True)
    print(f"feature_dim={bundle.data.x.size(1)}", flush=True)
    print(f"feature_source={bundle.feature_source}", flush=True)


def print_device_stats(device: torch.device) -> None:
    if device.type != "cuda":
        print(f"device={device}", flush=True)
        return
    print(f"device={device}", flush=True)
    print(f"gpu_name={torch.cuda.get_device_name(device)}", flush=True)
    print(f"gpu_memory_allocated_mb={torch.cuda.memory_allocated(device) / 1024**2:.2f}", flush=True)
    print(f"gpu_memory_reserved_mb={torch.cuda.memory_reserved(device) / 1024**2:.2f}", flush=True)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg.isdigit():
        device_arg = f"cuda:{device_arg}"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: requested {device_arg}, but CUDA is unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_arg)


def run_one_seed(
    args: argparse.Namespace,
    seed: int,
    bundle: ArxivBundle,
    edge_result: EdgePreprocessResult,
    device: torch.device,
    results_dir: Path,
    checkpoints_dir: Path,
) -> dict[str, object]:
    set_seed(seed)

    model = WeightedGraphSAGE(
        in_channels=bundle.data.x.size(1),
        hidden_channels=args.hidden_channels,
        out_channels=bundle.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        aggregation_chunk_size=args.aggregation_chunk_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.use_amp and device.type == "cuda")
    scaler = make_grad_scaler(use_amp)

    data_cpu = build_data(bundle, edge_result, device=torch.device("cpu"))
    data_device = build_data(bundle, edge_result, device=device)
    split_device = {name: idx.to(device) for name, idx in bundle.split_idx.items()}

    loader = None
    if args.training_mode == "mini":
        loader = build_neighbor_loader(args, data_cpu, bundle.split_idx["train"])

    rows: list[dict[str, object]] = []
    best_valid = -1.0
    best_test = -1.0
    best_epoch = 0
    final_train = 0.0
    final_valid = 0.0
    final_test = 0.0
    last_loss = float("nan")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if args.training_mode == "full":
            last_loss = train_full_epoch(
                model,
                data_device,
                split_device["train"],
                optimizer,
                scaler,
                use_amp,
                args.label_smoothing,
            )
        else:
            assert loader is not None
            last_loss = train_mini_epoch(
                model,
                loader,
                optimizer,
                scaler,
                use_amp,
                device,
                args.label_smoothing,
            )

        should_eval = (
            epoch == 1
            or epoch == args.epochs
            or epoch % max(1, args.eval_every) == 0
            or epoch % max(1, args.log_every) == 0
        )
        if not should_eval:
            continue

        train_acc, valid_acc, test_acc = evaluate(model, data_device, bundle.split_idx, bundle.evaluator)
        final_train, final_valid, final_test = train_acc, valid_acc, test_acc
        improved = valid_acc > best_valid
        if improved:
            best_valid = valid_acc
            best_test = test_acc
            best_epoch = epoch
            if args.save_checkpoint:
                save_checkpoint(checkpoints_dir, args, seed, epoch, model, edge_result)

        elapsed = time.time() - start_time
        row = {
            "mode": args.mode,
            "seed": seed,
            "epoch": epoch,
            "loss": last_loss,
            "train_acc": train_acc,
            "valid_acc": valid_acc,
            "test_acc": test_acc,
            "best_valid_acc": best_valid,
            "test_acc_at_best_valid": best_test,
            "best_epoch": best_epoch,
            "elapsed_sec": elapsed,
        }
        rows.append(row)

        if epoch == 1 or epoch % max(1, args.log_every) == 0:
            print_epoch_line(args.mode, seed, epoch, row)

        if args.patience > 0 and best_epoch > 0 and epoch - best_epoch >= args.patience:
            print(
                f"Early stopping at epoch={epoch}; best_epoch={best_epoch}; patience={args.patience}.",
                flush=True,
            )
            break

    runtime_sec = time.time() - start_time
    csv_path = results_dir / f"ogbn_arxiv_simteg_tape_sage_{args.mode}_seed{seed}.csv"
    write_rows_csv(csv_path, rows)

    summary = {
        "mode": args.mode,
        "seed": seed,
        "best_valid_acc": best_valid,
        "test_acc_at_best_valid": best_test,
        "best_epoch": best_epoch,
        "final_train_acc": final_train,
        "final_valid_acc": final_valid,
        "final_test_acc": final_test,
        "total_epochs_run": rows[-1]["epoch"] if rows else 0,
        "runtime_sec": runtime_sec,
        "csv_path": str(csv_path),
        "feature_source": bundle.feature_source,
        "feature_dim": int(bundle.data.x.size(1)),
        "preprocess_stats": edge_result.stats,
        "args": vars(args),
    }
    return summary


def build_data(bundle: ArxivBundle, edge_result: EdgePreprocessResult, device: torch.device) -> Data:
    data = Data(
        x=bundle.data.x.detach().cpu(),
        y=bundle.data.y.detach().cpu(),
        edge_index=edge_result.edge_index,
        edge_weight=edge_result.edge_weight,
        num_nodes=bundle.data.num_nodes,
    )
    return data.to(device)


def train_full_epoch(
    model: WeightedGraphSAGE,
    data: Data,
    train_idx: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    label_smoothing: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(use_amp):
        out = model(data.x, data.edge_index, data.edge_weight)
        loss = F.cross_entropy(
            out[train_idx],
            data.y[train_idx],
            label_smoothing=float(label_smoothing),
        )
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach().item())


def train_mini_epoch(
    model: WeightedGraphSAGE,
    loader: Iterable[Data],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    device: torch.device,
    label_smoothing: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        batch_size = int(batch.batch_size)
        edge_weight = getattr(batch, "edge_weight", None)
        with autocast_context(use_amp):
            out = model(batch.x, batch.edge_index, edge_weight)
            loss = F.cross_entropy(
                out[:batch_size],
                batch.y[:batch_size].view(-1),
                label_smoothing=float(label_smoothing),
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
    return total_loss / max(1, total_examples)


@torch.no_grad()
def evaluate(
    model: WeightedGraphSAGE,
    data: Data,
    split_idx: dict[str, torch.Tensor],
    evaluator,
) -> tuple[float, float, float]:
    model.eval()
    out = model(data.x, data.edge_index, data.edge_weight)
    y_pred = out.argmax(dim=-1, keepdim=True).cpu()
    y_true = data.y.view(-1, 1).cpu()

    scores = []
    for name in ("train", "valid", "test"):
        idx = split_idx[name].cpu()
        scores.append(
            float(
                evaluator.eval(
                    {
                        "y_true": y_true[idx],
                        "y_pred": y_pred[idx],
                    }
                )["acc"]
            )
        )
    return scores[0], scores[1], scores[2]


def build_neighbor_loader(args: argparse.Namespace, data_cpu: Data, train_idx: torch.Tensor):
    try:
        from torch_geometric.loader import NeighborLoader
    except ImportError as exc:
        raise RuntimeError("Mini-batch mode requires torch_geometric.loader.NeighborLoader.") from exc

    num_neighbors = parse_num_neighbors(args.num_neighbors, args.num_layers)
    print(
        f"Using mini-batch NeighborLoader: batch_size={args.batch_size} "
        f"num_neighbors={num_neighbors} num_workers={args.num_workers}",
        flush=True,
    )
    return NeighborLoader(
        data_cpu,
        input_nodes=train_idx,
        num_neighbors=num_neighbors,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )


def parse_num_neighbors(value: str, num_layers: int) -> list[int]:
    neighbors = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not neighbors:
        raise ValueError("--num_neighbors must contain at least one integer.")
    while len(neighbors) < num_layers:
        neighbors.append(neighbors[-1])
    return neighbors[:num_layers]


def autocast_context(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_epoch_line(mode: str, seed: int, epoch: int, row: dict[str, object]) -> None:
    print(
        f"[mode={mode} seed={seed} epoch={epoch}] "
        f"loss={row['loss']:.6f} "
        f"train_acc={row['train_acc']:.6f} "
        f"valid_acc={row['valid_acc']:.6f} "
        f"test_acc={row['test_acc']:.6f} "
        f"best_valid={row['best_valid_acc']:.6f} "
        f"best_test_at_best_valid={row['test_acc_at_best_valid']:.6f} "
        f"elapsed={format_duration(float(row['elapsed_sec']))}",
        flush=True,
    )


def save_checkpoint(
    checkpoints_dir: Path,
    args: argparse.Namespace,
    seed: int,
    epoch: int,
    model: WeightedGraphSAGE,
    edge_result: EdgePreprocessResult,
) -> None:
    path = checkpoints_dir / f"ogbn_arxiv_simteg_tape_sage_{args.mode}_seed{seed}_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mode": args.mode,
            "seed": seed,
            "epoch": epoch,
            "preprocess_stats": edge_result.stats,
            "args": vars(args),
        },
        path,
    )


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "mode",
        "seed",
        "epoch",
        "loss",
        "train_acc",
        "valid_acc",
        "test_acc",
        "best_valid_acc",
        "test_acc_at_best_valid",
        "best_epoch",
        "elapsed_sec",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {path}", flush=True)


def write_run_summary(results_dir: Path, summary: dict[str, object]) -> None:
    path = results_dir / f"run_summary_{summary['mode']}_seed{summary['seed']}.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote run summary: {path}", flush=True)


def update_summary_json(results_dir: Path) -> None:
    import fcntl

    lock_path = results_dir / ".summary.lock"
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        summaries = []
        for path in sorted(results_dir.glob("run_summary_*_seed*.json")):
            summaries.append(json.loads(path.read_text()))
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "num_runs": len(summaries),
            "runs": summaries,
            "by_mode": summarize_by_mode(summaries),
        }
        tmp_path = results_dir / "summary.json.tmp"
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(results_dir / "summary.json")
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
    print(f"Updated summary JSON: {results_dir / 'summary.json'}", flush=True)


def summarize_by_mode(summaries: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for mode in sorted({str(item["mode"]) for item in summaries}):
        mode_items = [item for item in summaries if item["mode"] == mode]
        best_valid = [float(item["best_valid_acc"]) for item in mode_items]
        best_test = [float(item["test_acc_at_best_valid"]) for item in mode_items]
        out[mode] = {
            "num_seeds": len(mode_items),
            "best_valid_mean": mean(best_valid),
            "best_valid_std": std(best_valid),
            "test_at_best_valid_mean": mean(best_test),
            "test_at_best_valid_std": std(best_test),
        }
    return out


def write_dry_run_stats(
    results_dir: Path,
    args: argparse.Namespace,
    bundle: ArxivBundle,
    edge_result: EdgePreprocessResult,
) -> None:
    path = results_dir / f"dry_run_{args.mode}.json"
    payload = {
        "mode": args.mode,
        "feature_source": bundle.feature_source,
        "feature_dim": int(bundle.data.x.size(1)),
        "preprocess_stats": edge_result.stats,
        "args": vars(args),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote dry-run stats: {path}", flush=True)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return float((sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5)


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()
