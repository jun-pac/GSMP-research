from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import to_undirected


VARIANTS = ("baseline", "smp", "ump", "gsmp")
CSV_FIELDS = [
    "timestamp",
    "dataset",
    "embedding_name",
    "variant",
    "seed",
    "epoch",
    "train_loss",
    "val_acc",
    "test_acc",
    "best_val_acc",
    "test_at_best_val",
    "best_epoch",
    "oracle_best_test_acc_not_for_model_selection",
    "lr",
    "gpu_name",
    "peak_gpu_mem_mb",
    "elapsed_sec",
]
SUMMARY_FIELDS = [
    "timestamp",
    "dataset",
    "embedding_name",
    "variant",
    "seed",
    "best_epoch",
    "best_val_acc",
    "test_at_best_val",
    "oracle_best_test_acc_not_for_model_selection",
    "epochs_run",
    "lr",
    "gpu_name",
    "peak_gpu_mem_mb",
    "elapsed_sec",
    "run_name",
]


@dataclass
class TemporalMP:
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    stats: dict[str, object]


@dataclass
class Bundle:
    data: Data
    split_idx: dict[str, torch.Tensor]
    evaluator: Evaluator
    years: torch.Tensor
    feature_source: str
    embedding_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="260609 SimTeG/TAPE GraphSAGE baseline/SMP/UMP/GSMP runner for ogbn-arxiv."
    )
    parser.add_argument("--dataset", default="ogbn-arxiv", choices=["ogbn-arxiv", "ogbn-arxiv-tape"])
    parser.add_argument("--mp_variant", default="baseline", choices=VARIANTS)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--seeds", default="1", help="Comma/space separated seeds, e.g. '1,2,3'.")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--output_root", default="results")
    parser.add_argument("--data_root", default="../data")
    parser.add_argument("--log_csv", default=None)
    parser.add_argument("--results_jsonl", default=None)
    parser.add_argument("--checkpoint_root", default=None)
    parser.add_argument("--use_bert_x", action="store_true")
    parser.add_argument("--bert_x_dir", default=None)
    parser.add_argument("--use_gpt_preds", action="store_true")
    parser.add_argument("--gpt_preds_path", default=None)
    parser.add_argument("--embedding_name", default="e5-large")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_neighbors", default="15,10,5,5")
    parser.add_argument("--gnn_batch_size", type=int, default=10000)
    parser.add_argument("--gnn_eval_batch_size", type=int, default=10000)
    parser.add_argument("--gnn_epochs", type=int, default=100)
    parser.add_argument("--gnn_dropout", type=float, default=0.4)
    parser.add_argument("--gnn_label_smoothing", type=float, default=0.4)
    parser.add_argument("--gnn_lr", type=float, default=0.01)
    parser.add_argument("--gnn_num_layers", type=int, default=2)
    parser.add_argument("--gnn_weight_decay", type=float, default=4e-6)
    parser.add_argument("--gnn_eval_interval", type=int, default=1)
    parser.add_argument("--hidden_channels", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--aggregation_chunk_size", type=int, default=200000)
    parser.add_argument("--eval_mode", default="mini", choices=["mini", "full"])
    parser.add_argument("--run_unit_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.max_epochs = 5 if args.max_epochs is None else min(args.max_epochs, 5)
        args.seeds = "1"
    epochs = args.max_epochs if args.max_epochs is not None else args.gnn_epochs
    if epochs <= 0:
        raise ValueError("--max_epochs/--gnn_epochs must be positive.")

    project_dir = Path(__file__).resolve().parent
    output_root = resolve_path(args.output_root, project_dir)
    run_name = args.run_name or default_run_name(args)
    args.run_name = run_name
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_csv = resolve_path(args.log_csv, project_dir) if args.log_csv else run_dir / "epoch_metrics.csv"
    results_jsonl = (
        resolve_path(args.results_jsonl, project_dir) if args.results_jsonl else run_dir / "epoch_metrics.jsonl"
    )
    checkpoint_root = (
        resolve_path(args.checkpoint_root, project_dir) if args.checkpoint_root else run_dir / "checkpoints"
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    if args.run_unit_test:
        run_temporal_mp_unit_test()

    print_header(args, run_dir, log_csv, results_jsonl)
    bundle = load_bundle(args, project_dir)
    temporal_mp = build_temporal_mp(
        bundle.data.edge_index,
        bundle.years,
        args.mp_variant,
        int(bundle.data.num_nodes),
    )
    print_temporal_mp_stats(args.mp_variant, temporal_mp)
    print_edge_direction_sanity(temporal_mp.edge_index, bundle.years)

    seeds = parse_seeds(args.seeds)
    summaries = []
    for seed in seeds:
        summary = run_seed(
            args=args,
            seed=seed,
            epochs=epochs,
            bundle=bundle,
            temporal_mp=temporal_mp,
            log_csv=log_csv,
            results_jsonl=results_jsonl,
            checkpoint_root=checkpoint_root,
        )
        append_summary(run_dir / "summary.csv", summary)
        summaries.append(summary)
        print_summary_line(summary)

    print_aggregate_summary(load_summary_rows(run_dir / "summary.csv"))


def print_header(args: argparse.Namespace, run_dir: Path, log_csv: Path, results_jsonl: Path) -> None:
    print("=== 260609 ogbn-arxiv SimTeG/TAPE GraphSAGE temporal MP experiment ===", flush=True)
    print(f"host={socket.gethostname()}", flush=True)
    print(f"date={datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    print(f"log_csv={log_csv}", flush=True)
    print(f"results_jsonl={results_jsonl}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    try:
        import torch_geometric

        print(f"torch_geometric={torch_geometric.__version__}", flush=True)
    except Exception as exc:
        print(f"torch_geometric_unavailable={exc}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"args={json.dumps(vars(args), sort_keys=True)}", flush=True)


def load_bundle(args: argparse.Namespace, project_dir: Path) -> Bundle:
    if bool(args.use_bert_x) == bool(args.use_gpt_preds):
        raise ValueError(
            "Choose exactly one feature source: --use_bert_x with --bert_x_dir, "
            "or --use_gpt_preds with --gpt_preds_path."
        )
    if args.use_bert_x and not args.bert_x_dir:
        raise ValueError(
            "Missing --bert_x_dir. Put cached embeddings at something like "
            "../SimTeG/out/ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt and pass that path."
        )

    data_root = resolve_path(args.data_root, project_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    with torch_load_weights_only_false():
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(data_root))
        data = dataset[0]
    data.y = data.y.view(-1).long()
    if args.dataset == "ogbn-arxiv":
        before_edges = int(data.edge_index.size(1))
        data.edge_index = to_undirected(data.edge_index, num_nodes=int(data.num_nodes))
        print(
            f"Applied official SimTeG ogbn-arxiv ToUndirected transform: "
            f"edges {before_edges} -> {int(data.edge_index.size(1))}",
            flush=True,
        )
    years = load_node_year(data, dataset).view(-1).long()

    if args.use_bert_x:
        bert_x_dir = resolve_path(args.bert_x_dir, project_dir)
        if not bert_x_dir.is_file():
            raise FileNotFoundError(
                f"Cached embeddings not found: {bert_x_dir}\n"
                "Do not launch LM fine-tuning by accident. Download/reuse x_embs.pt first, "
                "or run scripts/download_embeddings.sh as an explicit separate step."
            )
        x = load_feature_matrix(bert_x_dir, int(data.num_nodes))
        feature_source = str(bert_x_dir)
        print(f"Loaded cached embeddings from {bert_x_dir} shape={tuple(x.shape)}", flush=True)
    else:
        gpt_preds_path = resolve_gpt_preds_path(args.gpt_preds_path, project_dir)
        x = load_gpt_pred_matrix(gpt_preds_path, int(data.num_nodes))
        feature_source = str(gpt_preds_path)
        print(f"Loaded GPT prediction labels from {gpt_preds_path} shape={tuple(x.shape)}", flush=True)
    data.x = x
    split_idx = {name: idx.long() for name, idx in dataset.get_idx_split().items()}
    evaluator = Evaluator(name="ogbn-arxiv")
    print(f"dataset={args.dataset} ogb_graph=ogbn-arxiv feature_dim={x.size(1)}", flush=True)
    print(f"node_year_range=[{int(years.min())}, {int(years.max())}]", flush=True)
    return Bundle(
        data=data,
        split_idx=split_idx,
        evaluator=evaluator,
        years=years,
        feature_source=feature_source,
        embedding_name=args.embedding_name,
    )


@contextmanager
def torch_load_weights_only_false():
    original_load = torch.load

    def load_with_legacy_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_legacy_default
    try:
        yield
    finally:
        torch.load = original_load


def load_node_year(data: Data, dataset: PygNodePropPredDataset) -> torch.Tensor:
    for attr in ("node_year", "node_years", "year", "years"):
        if hasattr(data, attr):
            value = getattr(data, attr)
            if value is not None:
                return torch.as_tensor(value).view(-1)
    root = Path(dataset.root)
    candidates = [
        root / "raw" / "node_year.csv.gz",
        root / "raw" / "node-year.csv.gz",
        root / "raw" / "node_year.csv",
        root / "raw" / "node-year.csv",
    ]
    candidates.extend(sorted(root.rglob("*year*.csv*")))
    for path in candidates:
        if path.is_file():
            frame = pd.read_csv(path, compression="infer", header=None)
            year = torch.from_numpy(frame.values.reshape(-1))
            if year.numel() == data.num_nodes:
                return year
    raise FileNotFoundError("Could not find ogbn-arxiv node_year metadata.")


def load_feature_matrix(path: Path, num_nodes: int) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu") if path.suffix in {".pt", ".pth"} else np.load(path)
    tensor = tensor_from_object(obj, num_nodes, str(path))
    if tensor.dim() != 2:
        raise ValueError(f"Expected a 2D feature matrix from {path}, got shape={tuple(tensor.shape)}")
    if tensor.size(0) != num_nodes and tensor.size(1) == num_nodes:
        tensor = tensor.t()
    if tensor.size(0) != num_nodes:
        raise ValueError(f"Feature rows {tensor.size(0)} do not match num_nodes {num_nodes} for {path}")
    return tensor.contiguous().float()


def resolve_gpt_preds_path(value: str | None, project_dir: Path) -> Path:
    candidates = []
    if value:
        candidates.append(resolve_path(value, project_dir))
    candidates.extend(
        [
            project_dir / "resources" / "ogbn-arxiv-gpt-preds.csv",
            project_dir.parent / "SimTeG" / "src" / "misc" / "gpt_preds" / "ogbn-arxiv.csv",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "GPT prediction CSV not found. Run scripts/download_gpt_preds.sh first, "
        f"or pass --gpt_preds_path.\nSearched:\n{searched}"
    )


def load_gpt_pred_matrix(path: Path, num_nodes: int) -> torch.Tensor:
    rows: list[list[int]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            values = [int(value) for value in row[:5] if value != ""]
            rows.append(values)
    if len(rows) != num_nodes:
        raise ValueError(f"GPT prediction rows {len(rows)} do not match num_nodes {num_nodes} for {path}")
    pred = torch.zeros(num_nodes, 5, dtype=torch.long)
    for idx, values in enumerate(rows):
        if values:
            pred[idx, : len(values)] = torch.tensor(values, dtype=torch.long) + 1
    return pred.contiguous()


def tensor_from_object(obj, num_nodes: int, source: str) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(np.asarray(obj))
    if isinstance(obj, dict):
        for key in ("x", "x_embs", "embs", "embeddings", "features", "feat"):
            if key in obj:
                return tensor_from_object(obj[key], num_nodes, f"{source}:{key}")
        for key, value in obj.items():
            try:
                tensor = tensor_from_object(value, num_nodes, f"{source}:{key}")
            except Exception:
                continue
            if tensor.dim() == 2 and num_nodes in tensor.shape:
                return tensor
    raise ValueError(f"Could not find a feature tensor in {source}")


def build_temporal_mp(edge_index: torch.Tensor, years: torch.Tensor, variant: str, num_nodes: int) -> TemporalMP:
    variant = variant.lower()
    if variant not in VARIANTS:
        raise ValueError(f"Unknown mp_variant={variant}")

    edge_index = edge_index.detach().cpu().long().contiguous()
    years = years.detach().cpu().long().view(-1)
    src, dst = edge_index
    before = int(edge_index.size(1))
    stats: dict[str, object] = {
        "variant": variant,
        "num_nodes": int(num_nodes),
        "edges_before": before,
        "year_min": int(years.min()),
        "year_max": int(years.max()),
        "edge_direction": "edge_index[0] source neighbor u -> edge_index[1] target node v",
    }

    if variant == "baseline":
        edge_weight = torch.ones(before, dtype=torch.float32)
    elif variant == "smp":
        src_time = years[src].float()
        dst_time = years[dst].float()
        t_min = float(years.min())
        t_max = float(years.max())
        boundary = torch.minimum(
            torch.full_like(dst_time, t_max) - dst_time,
            dst_time - torch.full_like(dst_time, t_min),
        )
        single = (src_time == dst_time) | ((src_time - dst_time).abs() > boundary)
        edge_weight = torch.where(single, torch.full_like(src_time, 2.0), torch.ones_like(src_time)).float()
        stats.update(
            {
                "smp_single_edges": int(single.sum()),
                "smp_double_edges": int((~single).sum()),
                "smp_single_fraction": float(single.float().mean()) if single.numel() else 0.0,
            }
        )
    elif variant == "ump":
        keep = years[src] <= years[dst]
        edge_index = edge_index[:, keep]
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
        edge_index, edge_weight, added = add_zero_incoming_self_loops(edge_index, edge_weight, num_nodes)
        stats.update(
            {
                "ump_edges_dropped": before - int(keep.sum()),
                "ump_edges_kept": int(keep.sum()),
                "zero_incoming_self_loops_added": added,
            }
        )
    else:
        unique_years, year_inverse = torch.unique(years, sorted=True, return_inverse=True)
        src_year_id = year_inverse[src]
        num_years = int(unique_years.numel())
        key = dst * num_years + src_year_id
        group_count = torch.bincount(key, minlength=num_nodes * num_years).float()
        edge_weight = 1.0 / group_count[key].clamp_min(1.0)
        edge_index, edge_weight, added = add_zero_incoming_self_loops(edge_index, edge_weight.float(), num_nodes)
        stats.update(
            {
                "gsmp_num_years": num_years,
                "gsmp_nonempty_target_year_groups": int((group_count > 0).sum()),
                "zero_incoming_self_loops_added": added,
            }
        )

    stats["edges_after"] = int(edge_index.size(1))
    stats["weight_min"] = float(edge_weight.min()) if edge_weight.numel() else 0.0
    stats["weight_mean"] = float(edge_weight.mean()) if edge_weight.numel() else 0.0
    stats["weight_max"] = float(edge_weight.max()) if edge_weight.numel() else 0.0
    return TemporalMP(edge_index=edge_index.contiguous(), edge_weight=edge_weight.contiguous(), stats=stats)


def add_zero_incoming_self_loops(
    edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    incoming = torch.bincount(edge_index[1], minlength=num_nodes)
    zero_nodes = torch.nonzero(incoming == 0, as_tuple=False).view(-1)
    if zero_nodes.numel() == 0:
        return edge_index, edge_weight, 0
    loops = torch.stack([zero_nodes, zero_nodes], dim=0)
    loop_weight = torch.ones(zero_nodes.numel(), dtype=edge_weight.dtype)
    return torch.cat([edge_index, loops], dim=1), torch.cat([edge_weight, loop_weight]), int(zero_nodes.numel())


def print_temporal_mp_stats(variant: str, temporal_mp: TemporalMP) -> None:
    print("=== temporal message passing ===", flush=True)
    print(f"mp_variant={variant}", flush=True)
    for key in sorted(temporal_mp.stats):
        print(f"mp_{key}={temporal_mp.stats[key]}", flush=True)


def print_edge_direction_sanity(edge_index: torch.Tensor, years: torch.Tensor) -> None:
    if edge_index.numel() == 0:
        print("SANITY edge_direction_check=no_edges", flush=True)
        return
    src = int(edge_index[0, 0])
    dst = int(edge_index[1, 0])
    print(
        "SANITY edge_direction_check="
        f"edge_index[0]=source_neighbor={src} year={int(years[src])} "
        f"edge_index[1]=target_aggregated_into={dst} year={int(years[dst])}",
        flush=True,
    )


def run_temporal_mp_unit_test() -> None:
    years = torch.tensor([2000, 2001, 2001, 2004])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 2, 0]])
    baseline = build_temporal_mp(edge_index, years, "baseline", 4)
    smp = build_temporal_mp(edge_index, years, "smp", 4)
    ump = build_temporal_mp(edge_index, years, "ump", 4)
    gsmp = build_temporal_mp(edge_index, years, "gsmp", 4)
    assert baseline.edge_index.size(1) == 4
    assert smp.edge_weight.size(0) == 4
    assert (((ump.edge_index[0] == 3) & (ump.edge_index[1] == 0)).sum().item()) == 0
    assert gsmp.edge_weight.size(0) >= 4
    print("UNIT_TEST temporal_mp=passed", flush=True)


class WeightedSAGEConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, aggregation_chunk_size: int = 200000):
        super().__init__()
        self.lin_l = Linear(in_channels, out_channels, bias=True)
        self.lin_r = Linear(in_channels, out_channels, bias=False)
        self.aggregation_chunk_size = int(aggregation_chunk_size)

    def reset_parameters(self) -> None:
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None) -> torch.Tensor:
        src, dst = edge_index
        weights = (
            torch.ones(src.numel(), device=x.device, dtype=x.dtype)
            if edge_weight is None
            else edge_weight.to(device=x.device, dtype=x.dtype).view(-1)
        )
        out = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
        chunk = self.aggregation_chunk_size
        if chunk <= 0 or src.numel() <= chunk:
            out.index_add_(0, dst, x[src] * weights.view(-1, 1))
        else:
            for start in range(0, src.numel(), chunk):
                end = min(start + chunk, src.numel())
                out.index_add_(0, dst[start:end], x[src[start:end]] * weights[start:end].view(-1, 1))
        denom = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        denom.index_add_(0, dst, weights)
        out = out / denom.clamp_min(torch.finfo(x.dtype).eps).view(-1, 1)
        return self.lin_l(out) + self.lin_r(x)


class PyGGraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        use_gpt_preds: bool = False,
        gpt_pred_slots: int = 5,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = float(dropout)
        self.encoder = nn.Embedding(out_channels + 1, hidden_channels) if use_gpt_preds else None
        if self.encoder is not None:
            in_channels = gpt_pred_slots * hidden_channels
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        del edge_weight
        if self.encoder is not None:
            x = torch.flatten(self.encoder(x.long()), start_dim=1)
        for layer_idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if layer_idx < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class WeightedGraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        aggregation_chunk_size: int,
        use_gpt_preds: bool = False,
        gpt_pred_slots: int = 5,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = float(dropout)
        self.encoder = nn.Embedding(out_channels + 1, hidden_channels) if use_gpt_preds else None
        if self.encoder is not None:
            in_channels = gpt_pred_slots * hidden_channels
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(WeightedSAGEConv(in_channels, out_channels, aggregation_chunk_size))
        else:
            self.convs.append(WeightedSAGEConv(in_channels, hidden_channels, aggregation_chunk_size))
            for _ in range(num_layers - 2):
                self.convs.append(WeightedSAGEConv(hidden_channels, hidden_channels, aggregation_chunk_size))
            self.convs.append(WeightedSAGEConv(hidden_channels, out_channels, aggregation_chunk_size))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        if self.encoder is not None:
            x = torch.flatten(self.encoder(x.long()), start_dim=1)
        for layer_idx, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if layer_idx < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def run_seed(
    args: argparse.Namespace,
    seed: int,
    epochs: int,
    bundle: Bundle,
    temporal_mp: TemporalMP,
    log_csv: Path,
    results_jsonl: Path,
    checkpoint_root: Path,
) -> dict[str, object]:
    set_seed(seed)
    device = resolve_device(args.device)
    gpu_name = get_gpu_name(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data_cpu = Data(
        x=bundle.data.x.detach().cpu(),
        y=bundle.data.y.detach().cpu(),
        edge_index=temporal_mp.edge_index,
        edge_weight=temporal_mp.edge_weight,
        num_nodes=int(bundle.data.num_nodes),
    )
    hidden_channels = args.hidden_channels or (768 if args.use_gpt_preds else int(data_cpu.x.size(1)))
    if args.mp_variant == "baseline":
        model = PyGGraphSAGE(
            in_channels=int(data_cpu.x.size(1)),
            hidden_channels=hidden_channels,
            out_channels=40,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            use_gpt_preds=args.use_gpt_preds,
        ).to(device)
        print("model_layer=PyG SAGEConv exact baseline", flush=True)
    else:
        model = WeightedGraphSAGE(
            in_channels=int(data_cpu.x.size(1)),
            hidden_channels=hidden_channels,
            out_channels=40,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            aggregation_chunk_size=args.aggregation_chunk_size,
            use_gpt_preds=args.use_gpt_preds,
        ).to(device)
        print("model_layer=weighted SAGEConv temporal variant", flush=True)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.gnn_lr, weight_decay=args.gnn_weight_decay)
    train_loader = make_neighbor_loader(args, data_cpu, bundle.split_idx["train"], shuffle=True)
    eval_loader = None
    data_device = None
    if args.eval_mode == "mini":
        eval_loader = make_neighbor_loader(args, data_cpu, None, shuffle=False, eval_loader=True)
    else:
        data_device = data_cpu.to(device)

    best_val_acc = -math.inf
    test_at_best_val = None
    best_epoch = -1
    oracle_best_test_acc = -math.inf
    start_time = time.time()
    ckpt_dir = checkpoint_root / args.mp_variant / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.gnn_label_smoothing)
        if epoch % max(1, args.gnn_eval_interval) != 0 and epoch != epochs:
            continue
        if args.eval_mode == "mini":
            assert eval_loader is not None
            val_acc, test_acc, logits = evaluate_mini(
                model,
                eval_loader,
                int(bundle.data.num_nodes),
                bundle.data.y.detach().cpu(),
                bundle.split_idx,
                bundle.evaluator,
                device,
            )
        else:
            assert data_device is not None
            val_acc, test_acc, logits = evaluate_full(
                model,
                data_device,
                bundle.data.y.detach().cpu(),
                bundle.split_idx,
                bundle.evaluator,
            )
        oracle_best_test_acc = max(oracle_best_test_acc, test_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_at_best_val = test_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "variant": args.mp_variant,
                    "seed": seed,
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "test_at_best_val": test_at_best_val,
                    "args": vars(args),
                    "temporal_mp_stats": temporal_mp.stats,
                },
                ckpt_dir / "best_by_val.pt",
            )
            logits_dir = checkpoint_root.parent / "cached_embs" / args.mp_variant
            logits_dir.mkdir(parents=True, exist_ok=True)
            torch.save(logits, logits_dir / f"logits_seed{seed}.pt")

        elapsed = time.time() - start_time
        peak_mem = peak_gpu_mem_mb(device)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "embedding_name": bundle.embedding_name,
            "variant": args.mp_variant,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "best_val_acc": best_val_acc,
            "test_at_best_val": test_at_best_val,
            "best_epoch": best_epoch,
            "oracle_best_test_acc_not_for_model_selection": oracle_best_test_acc,
            "lr": args.gnn_lr,
            "gpu_name": gpu_name,
            "peak_gpu_mem_mb": peak_mem,
            "elapsed_sec": elapsed,
        }
        append_csv(log_csv, row, CSV_FIELDS)
        append_jsonl(results_jsonl, row)
        print_result_line(args.mp_variant, seed, epoch, row)

        if args.early_stop_patience > 0 and best_epoch > 0 and epoch - best_epoch >= args.early_stop_patience:
            print(
                f"EARLY_STOP variant={args.mp_variant} seed={seed} epoch={epoch} best_epoch={best_epoch}",
                flush=True,
            )
            break

    elapsed = time.time() - start_time
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "embedding_name": bundle.embedding_name,
        "variant": args.mp_variant,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_at_best_val": test_at_best_val,
        "oracle_best_test_acc_not_for_model_selection": oracle_best_test_acc,
        "epochs_run": epoch,
        "lr": args.gnn_lr,
        "gpu_name": gpu_name,
        "peak_gpu_mem_mb": peak_gpu_mem_mb(device),
        "elapsed_sec": elapsed,
        "run_name": args.run_name or default_run_name(args),
    }


def make_neighbor_loader(
    args: argparse.Namespace,
    data_cpu: Data,
    input_nodes: torch.Tensor | None,
    shuffle: bool,
    eval_loader: bool = False,
) -> NeighborLoader:
    if eval_loader:
        num_neighbors = [-1]
        batch_size = args.gnn_eval_batch_size
    else:
        num_neighbors = parse_num_neighbors(args.num_neighbors, args.gnn_num_layers)
        batch_size = args.gnn_batch_size
    print(
        f"NeighborLoader eval={eval_loader} batch_size={batch_size} "
        f"num_neighbors={num_neighbors} num_workers={args.num_workers}",
        flush=True,
    )
    return NeighborLoader(
        data_cpu,
        input_nodes=input_nodes,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
    )


def train_epoch(
    model: WeightedGraphSAGE,
    loader: Iterable[Data],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        batch_size = int(batch.batch_size)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch.x, batch.edge_index, getattr(batch, "edge_weight", None))[:batch_size]
        y = batch.y[:batch_size].view(-1)
        loss = F.cross_entropy(out, y, label_smoothing=float(label_smoothing))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
    return total_loss / max(1, total_examples)


@torch.no_grad()
def evaluate_mini(
    model: WeightedGraphSAGE,
    loader: Iterable[Data],
    num_nodes: int,
    y_true: torch.Tensor,
    split_idx: dict[str, torch.Tensor],
    evaluator: Evaluator,
    device: torch.device,
) -> tuple[float, float, torch.Tensor]:
    model.eval()
    chunks = []
    id_chunks = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, getattr(batch, "edge_weight", None))[: int(batch.batch_size)]
        chunks.append(out.detach().cpu())
        id_chunks.append(batch.n_id[: int(batch.batch_size)].detach().cpu())
    out_all = torch.cat(chunks, dim=0)
    ids = torch.cat(id_chunks, dim=0)
    if ids.unique().numel() != num_nodes:
        raise RuntimeError(
            f"Evaluation loader covered {ids.unique().numel()} unique nodes, expected {num_nodes}."
        )
    logits = torch.empty(num_nodes, out_all.size(-1), dtype=out_all.dtype)
    logits[ids] = out_all
    val_acc, test_acc = evaluate_logits(logits, y_true, split_idx, evaluator)
    return val_acc, test_acc, logits


@torch.no_grad()
def evaluate_full(
    model: WeightedGraphSAGE,
    data: Data,
    y_true: torch.Tensor,
    split_idx: dict[str, torch.Tensor],
    evaluator: Evaluator,
) -> tuple[float, float, torch.Tensor]:
    model.eval()
    logits = model(data.x, data.edge_index, data.edge_weight).detach().cpu()
    val_acc, test_acc = evaluate_logits(logits, y_true, split_idx, evaluator)
    return val_acc, test_acc, logits


def evaluate_logits(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    split_idx: dict[str, torch.Tensor],
    evaluator: Evaluator,
) -> tuple[float, float]:
    y_pred = logits.argmax(dim=-1, keepdim=True)
    y_true = y_true.view(-1, 1)
    val_acc = float(evaluator.eval({"y_true": y_true[split_idx["valid"]], "y_pred": y_pred[split_idx["valid"]]})["acc"])
    test_acc = float(evaluator.eval({"y_true": y_true[split_idx["test"]], "y_pred": y_pred[split_idx["test"]]})["acc"])
    return val_acc, test_acc


def print_result_line(variant: str, seed: int, epoch: int, row: dict[str, object]) -> None:
    print(
        f"RESULT variant={variant} seed={seed} epoch={epoch} "
        f"val_acc={float(row['val_acc']):.4f} test_acc={float(row['test_acc']):.4f} "
        f"best_val_acc={float(row['best_val_acc']):.4f} "
        f"test_at_best_val={float(row['test_at_best_val']):.4f} "
        f"best_epoch={int(row['best_epoch'])} loss={float(row['train_loss']):.6f}",
        flush=True,
    )


def print_summary_line(summary: dict[str, object]) -> None:
    print(
        f"SUMMARY variant={summary['variant']} seed={summary['seed']} "
        f"best_epoch={summary['best_epoch']} "
        f"best_val_acc={float(summary['best_val_acc']):.4f} "
        f"test_at_best_val={float(summary['test_at_best_val']):.4f} "
        "oracle_best_test_acc_not_for_model_selection="
        f"{float(summary['oracle_best_test_acc_not_for_model_selection']):.4f}",
        flush=True,
    )


def append_csv(path: Path, row: dict[str, object], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_summary(path: Path, summary: dict[str, object]) -> None:
    append_csv(path, summary, SUMMARY_FIELDS)


def load_summary_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def print_aggregate_summary(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    print("AGGREGATE mean+-std by variant over completed seeds", flush=True)
    variants = sorted({row["variant"] for row in rows})
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        val = [float(row["best_val_acc"]) for row in subset]
        test = [float(row["test_at_best_val"]) for row in subset]
        print(
            f"AGGREGATE variant={variant} seeds={len(subset)} "
            f"best_val_acc={mean(val):.4f}+-{std(val):.4f} "
            f"test_at_best_val={mean(test):.4f}+-{std(test):.4f}",
            flush=True,
        )


def parse_num_neighbors(value: str, num_layers: int) -> list[int]:
    out = [int(part.strip()) for part in value.replace(" ", ",").split(",") if part.strip()]
    if not out:
        raise ValueError("--num_neighbors must contain at least one integer.")
    while len(out) < num_layers:
        out.append(out[-1])
    return out[:num_layers]


def parse_seeds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.replace(",", " ").split() if part.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg.isdigit():
        device_arg = f"cuda:{device_arg}"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING requested {device_arg}, but CUDA is unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_arg)


def get_gpu_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"


def peak_gpu_mem_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / 1024**2)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def default_run_name(args: argparse.Namespace) -> str:
    stage = "smoke" if args.smoke_test else "main"
    return f"{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def resolve_path(path: str | Path, project_dir: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


if __name__ == "__main__":
    main()
