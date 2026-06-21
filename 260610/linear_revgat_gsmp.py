#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.utils import expand_as_pair
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_SIMTEG_DIR = REPO_DIR / "SimTeG"
if DEFAULT_SIMTEG_DIR.is_dir():
    sys.path.insert(0, str(DEFAULT_SIMTEG_DIR))

with contextlib.redirect_stdout(io.StringIO()):
    from src.misc.revgat.model_rev import ElementWiseLinear  # noqa: E402
    from src.misc.revgat.rev import memgcn  # noqa: E402
    from src.misc.revgat.rev.rev_layer import SharedDropout  # noqa: E402


EPSILON = 1 - math.log(2)
CSV_FIELDS = [
    "timestamp",
    "experiment_name",
    "model",
    "seed",
    "dataset",
    "feature_source",
    "gsmp_enabled",
    "gsmp_norm",
    "use_self_loops",
    "edge_direction",
    "epoch",
    "train_acc",
    "valid_acc",
    "test_acc",
    "best_valid_acc_so_far",
    "test_acc_at_best_valid_so_far",
    "best_valid_epoch",
    "pure_best_test_acc_so_far",
    "pure_best_test_epoch",
    "loss",
    "lr",
    "elapsed_seconds",
    "gpu_name",
    "peak_gpu_mem_mb",
    "git_commit_hash",
]


@dataclass
class LoadedData:
    graph: dgl.DGLGraph
    labels: torch.Tensor
    train_idx: torch.Tensor
    valid_idx: torch.Tensor
    test_idx: torch.Tensor
    evaluator: Evaluator
    node_year: torch.Tensor
    n_node_feats: int
    n_classes: int


class LinearWeightedConv(nn.Module):
    """RevGAT-compatible linear aggregation.

    DGL stores edges as source -> destination. This layer aggregates source
    messages into destination nodes with graph.edata["mp_weight"].
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        num_heads: int = 1,
        edge_drop: float = 0.0,
        residual: bool = False,
        activation=None,
        allow_zero_in_degree: bool = False,
        use_symmetric_norm: bool = False,
    ):
        super().__init__()
        self._num_heads = int(num_heads)
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = int(out_feats)
        self._allow_zero_in_degree = bool(allow_zero_in_degree)
        self._use_symmetric_norm = bool(use_symmetric_norm)
        self.edge_drop = float(edge_drop)
        self._activation = activation
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor | tuple[torch.Tensor, torch.Tensor], perm=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree and (graph.in_degrees() == 0).any():
                raise ValueError("Graph has zero in-degree nodes; add self-loops or allow zero in-degree.")

            if isinstance(feat, tuple):
                h_src, h_dst = feat
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = feat
                feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                else:
                    h_dst = h_src
                    feat_dst = feat_src

            if self._use_symmetric_norm:
                degs = graph.out_degrees().float().clamp(min=1).to(feat_src.device)
                norm = torch.pow(degs, -0.5).view(-1, *([1] * (feat_src.dim() - 1)))
                feat_src = feat_src * norm

            if "mp_weight" not in graph.edata:
                raise KeyError('Missing graph.edata["mp_weight"]. Run attach_message_weights first.')
            weights = graph.edata["mp_weight"].to(device=feat_src.device, dtype=feat_src.dtype).view(-1)
            weights = self._apply_edge_dropout(graph, weights, perm)
            graph.srcdata["ft"] = feat_src
            graph.edata["w"] = weights.view(-1, 1, 1)
            graph.update_all(fn.u_mul_e("ft", "w", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self._use_symmetric_norm:
                degs = graph.in_degrees().float().clamp(min=1).to(rst.device)
                norm = torch.pow(degs, 0.5).view(-1, *([1] * (rst.dim() - 1)))
                rst = rst * norm

            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            if self._activation is not None:
                rst = self._activation(rst)
            return rst

    def _apply_edge_dropout(self, graph: dgl.DGLGraph, weights: torch.Tensor, perm) -> torch.Tensor:
        if not self.training or self.edge_drop <= 0:
            return weights
        if perm is None:
            perm = torch.randperm(graph.number_of_edges(), device=weights.device)
        else:
            perm = perm.squeeze().to(weights.device)
        keep_start = int(graph.number_of_edges() * self.edge_drop)
        keep_eids = perm[keep_start:]
        dropped = torch.zeros_like(weights)
        dropped[keep_eids] = weights[keep_eids]

        _, dst = graph.edges(order="eid")
        dst = dst.to(weights.device)
        denom = torch.zeros(graph.number_of_dst_nodes(), dtype=weights.dtype, device=weights.device)
        denom.scatter_add_(0, dst, dropped)
        return dropped / denom[dst].clamp_min(torch.finfo(weights.dtype).eps)


class LinearRevGATBlock(nn.Module):
    def __init__(
        self,
        node_feats: int,
        out_feats: int,
        n_heads: int = 1,
        edge_drop: float = 0.0,
        residual: bool = True,
        activation=None,
        allow_zero_in_degree: bool = True,
        use_symmetric_norm: bool = False,
    ):
        super().__init__()
        self.norm = nn.BatchNorm1d(n_heads * out_feats)
        self.conv = LinearWeightedConv(
            node_feats,
            out_feats,
            num_heads=n_heads,
            edge_drop=edge_drop,
            residual=residual,
            activation=activation,
            allow_zero_in_degree=allow_zero_in_degree,
            use_symmetric_norm=use_symmetric_norm,
        )
        self.dropout = SharedDropout()

    def forward(self, x, graph, dropout_mask=None, perm=None, efeat=None):
        del efeat
        if perm is not None:
            perm = perm.squeeze()
        out = self.norm(x)
        out = F.relu(out, inplace=True)
        self.dropout.set_mask(dropout_mask)
        out = self.dropout(out)
        return self.conv(graph, out, perm).flatten(1, -1)


class LinearRevGAT(nn.Module):
    """RevGAT shell with attention replaced by fixed linear aggregation."""

    def __init__(
        self,
        in_feats: int,
        n_classes: int,
        n_hidden: int,
        n_layers: int,
        n_heads: int,
        activation,
        dropout: float = 0.0,
        input_drop: float = 0.0,
        edge_drop: float = 0.0,
        use_symmetric_norm: bool = False,
        group: int = 2,
        use_gpt_preds: bool = False,
        input_norm: bool = True,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads
        self.group = group
        self.convs = nn.ModuleList()
        self.norm = nn.BatchNorm1d(n_heads * n_hidden)
        if input_norm:
            self.input_norm = nn.BatchNorm1d(in_feats)
        if use_gpt_preds:
            self.encoder = nn.Embedding(n_classes + 1, n_hidden)

        for layer_idx in range(n_layers):
            in_hidden = n_heads * n_hidden if layer_idx > 0 else in_feats
            out_hidden = n_hidden if layer_idx < n_layers - 1 else n_classes
            num_heads = n_heads if layer_idx < n_layers - 1 else 1
            if layer_idx == 0 or layer_idx == n_layers - 1:
                self.convs.append(
                    LinearWeightedConv(
                        in_hidden,
                        out_hidden,
                        num_heads=num_heads,
                        edge_drop=edge_drop,
                        use_symmetric_norm=use_symmetric_norm,
                        residual=True,
                    )
                )
            else:
                blocks = nn.ModuleList()
                block = LinearRevGATBlock(
                    in_hidden // group,
                    out_hidden // group,
                    n_heads=num_heads,
                    edge_drop=edge_drop,
                    use_symmetric_norm=use_symmetric_norm,
                    residual=True,
                )
                for group_idx in range(group):
                    blocks.append(block if group_idx == 0 else copy_module(block))
                coupling = memgcn.GroupAdditiveCoupling(blocks, group=group)
                self.convs.append(memgcn.InvertibleModuleWrapper(fn=coupling, keep_input=False))

        self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)
        self.input_drop = nn.Dropout(input_drop)
        self.dropout = float(dropout)
        self.dp_last = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        x = feat
        if hasattr(self, "encoder"):
            embs = self.encoder(x[:, :5].to(torch.long))
            x = torch.cat([torch.flatten(embs, start_dim=1), x[:, 5:]], dim=1)
        if hasattr(self, "input_norm"):
            x = self.input_norm(x)
        x = self.input_drop(x)

        perms = [torch.randperm(graph.number_of_edges(), device=graph.device) for _ in range(self.n_layers)]
        x = self.convs[0](graph, x, perms[0]).flatten(1, -1)

        mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
        mask = mask.requires_grad_(False) / max(torch.finfo(x.dtype).eps, 1 - self.dropout)
        for layer_idx in range(1, self.n_layers - 1):
            graph.requires_grad = False
            perm = torch.stack([perms[layer_idx]] * self.group, dim=1)
            x = self.convs[layer_idx](x, graph, mask, perm)

        x = self.norm(x)
        x = self.activation(x, inplace=True)
        x = self.dp_last(x)
        x = self.convs[-1](graph, x, perms[-1])
        x = x.mean(1)
        return self.bias_last(x)


def copy_module(module: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Linearized RevGAT with optional GSMP edge weights.")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--model_variant", choices=["linear", "gsmp"], default="linear")
    parser.add_argument("--dataset", default="ogbn-arxiv")
    parser.add_argument("--feature_source", default="unknown")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_epochs", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dgl_data_root", default="../dgl_data")
    parser.add_argument("--output_root", default="runs/ogbn_arxiv_simteg_tape_revgat_gsmp")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--use_bert_x", action="store_true")
    parser.add_argument("--bert_x_dir", default=None)
    parser.add_argument("--use_gpt_preds", action="store_true")
    parser.add_argument("--gpt_preds_path", default="resources/ogbn-arxiv-gpt-preds.csv")
    parser.add_argument("--use-labels", action="store_true")
    parser.add_argument("--n-label-iters", type=int, default=0)
    parser.add_argument("--mask-rate", type=float, default=0.5)
    parser.add_argument("--use-norm", action="store_true")
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.58)
    parser.add_argument("--input-drop", type=float, default=0.37)
    parser.add_argument("--edge-drop", type=float, default=0.0)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--label_smoothing_factor", type=float, default=0.02)
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--save_pred", action="store_true")
    parser.add_argument("--gsmp_norm", choices=["active_years", "all_years"], default="active_years")
    parser.add_argument("--year_universe", choices=["unique", "range"], default="unique")
    parser.add_argument("--edge_direction", choices=["bidirected", "original", "reverse"], default="bidirected")
    parser.add_argument("--no_self_loops", action="store_true")
    parser.add_argument("--log_json", action="store_true")
    args = parser.parse_args()
    if args.use_bert_x == args.use_gpt_preds:
        raise ValueError("Select exactly one feature source: --use_bert_x or --use_gpt_preds.")
    if args.use_bert_x and not args.bert_x_dir:
        raise ValueError("--bert_x_dir is required with --use_bert_x.")
    if args.n_epochs < 1:
        raise ValueError("--n_epochs must be >= 1.")
    return args


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    experiment_name = args.experiment_name or default_experiment_name(args)
    output_root = resolve_path(args.output_root)
    run_name = args.run_name or experiment_name
    component_dir = output_root / "components" / run_name
    logs_dir = output_root / "logs"
    csv_dir = output_root / "csv"
    for path in (component_dir, logs_dir, csv_dir):
        path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu else f"cuda:{args.gpu}")
    if device.type == "cuda":
        preflight_cuda_runtime(device)
    print_header(args, experiment_name, component_dir, device)
    loaded = load_data(args)
    loaded.graph = preprocess_graph(args, loaded.graph, loaded.node_year)
    device_items = [loaded.graph, loaded.labels, loaded.train_idx, loaded.valid_idx, loaded.test_idx]
    loaded.graph, loaded.labels, loaded.train_idx, loaded.valid_idx, loaded.test_idx = [
        item.to(device) for item in device_items
    ]
    rows_summary = []
    for seed in seeds:
        summary = run_seed(args, experiment_name, component_dir, csv_dir, loaded, seed, device)
        rows_summary.append(summary)
        append_csv(csv_dir / "per_seed_summary.csv", [summary], list(summary.keys()))
        append_csv(component_dir / "per_seed_summary.csv", [summary], list(summary.keys()))

    aggregate = aggregate_summary(args, experiment_name, rows_summary)
    append_csv(csv_dir / "aggregate_summary.csv", [aggregate], list(aggregate.keys()))
    write_json(component_dir / "aggregate_summary.json", aggregate)
    print(
        "AGGREGATE "
        f"experiment={experiment_name} seeds={len(rows_summary)} "
        f"best_valid={aggregate['best_valid_acc_mean']:.6f}+-{aggregate['best_valid_acc_std']:.6f} "
        f"test_at_best_valid={aggregate['test_acc_at_best_valid_mean']:.6f}+-"
        f"{aggregate['test_acc_at_best_valid_std']:.6f}",
        flush=True,
    )


def load_data(args: argparse.Namespace) -> LoadedData:
    data_root = resolve_path(args.dgl_data_root)
    dataset = DglNodePropPredDataset(name=args.dataset, root=str(data_root))
    evaluator = Evaluator(name=args.dataset)
    split = dataset.get_idx_split()
    graph, labels = dataset[0]
    if "year" not in graph.ndata:
        raise KeyError('Expected OGBN-Arxiv DGL graph to expose node years as graph.ndata["year"].')
    node_year = graph.ndata["year"].view(-1).long()
    if args.use_bert_x:
        feat = torch.load(resolve_path(args.bert_x_dir), map_location="cpu")
        graph.ndata["feat"] = feat
        print(f"Loaded cached embeddings shape={tuple(feat.shape)} path={resolve_path(args.bert_x_dir)}", flush=True)
        n_node_feats = int(feat.shape[1])
    else:
        feat = load_gpt_preds(resolve_path(args.gpt_preds_path))
        graph.ndata["feat"] = feat
        print(f"Loaded GPT prediction labels shape={tuple(feat.shape)} path={resolve_path(args.gpt_preds_path)}", flush=True)
        n_node_feats = args.n_hidden * 5
    n_classes = int(labels.max().item() + 1)
    return LoadedData(
        graph=graph,
        labels=labels,
        train_idx=split["train"],
        valid_idx=split["valid"],
        test_idx=split["test"],
        evaluator=evaluator,
        node_year=node_year,
        n_node_feats=n_node_feats,
        n_classes=n_classes,
    )


def preprocess_graph(args: argparse.Namespace, graph: dgl.DGLGraph, node_year: torch.Tensor) -> dgl.DGLGraph:
    ndata = {key: value for key, value in graph.ndata.items()}
    before_edges = graph.number_of_edges()
    if args.edge_direction == "bidirected":
        graph = dgl.to_bidirected(graph, copy_ndata=False)
    elif args.edge_direction == "reverse":
        graph = dgl.reverse(graph, copy_ndata=False)
    for key, value in ndata.items():
        graph.ndata[key] = value
    if not args.no_self_loops:
        graph = graph.remove_self_loop().add_self_loop()
    else:
        graph = graph.remove_self_loop()
    graph.create_formats_()
    stats = attach_message_weights(
        graph,
        node_year,
        use_gsmp=args.model_variant == "gsmp",
        gsmp_norm=args.gsmp_norm,
        year_universe=args.year_universe,
    )
    print(
        "GRAPH "
        f"edge_direction={args.edge_direction} edges_before={before_edges} edges_after={graph.number_of_edges()} "
        f"use_self_loops={not args.no_self_loops}",
        flush=True,
    )
    print("EDGE_WEIGHT_STATS " + json.dumps(stats, sort_keys=True), flush=True)
    return graph


def attach_message_weights(
    graph: dgl.DGLGraph,
    node_year: torch.Tensor,
    use_gsmp: bool,
    gsmp_norm: str,
    year_universe: str,
) -> dict[str, float | int | str | bool]:
    src, dst = graph.edges(order="eid")
    src = src.cpu().long()
    dst = dst.cpu().long()
    num_nodes = graph.num_nodes()
    if use_gsmp:
        weights, year_count, min_year, max_year, active_groups = compute_gsmp_weights(
            src, dst, node_year.cpu().long(), num_nodes, gsmp_norm, year_universe
        )
    else:
        weights = compute_mean_weights(dst, num_nodes)
        years = node_year.view(-1).long()
        year_count = int(torch.unique(years).numel())
        min_year = int(years.min().item())
        max_year = int(years.max().item())
        active_groups = 0
    graph.edata["mp_weight"] = weights.to(torch.float32)
    return {
        "use_gsmp": bool(use_gsmp),
        "gsmp_norm": gsmp_norm,
        "year_universe": year_universe,
        "num_years": int(year_count),
        "year_min": int(min_year),
        "year_max": int(max_year),
        "active_target_year_groups": int(active_groups),
        "weight_min": float(weights.min().item()),
        "weight_mean": float(weights.mean().item()),
        "weight_max": float(weights.max().item()),
        "target_weight_sum_mean": float(target_weight_sum_mean(dst, weights, num_nodes)),
    }


def compute_mean_weights(dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    deg = torch.bincount(dst, minlength=num_nodes).to(torch.float32).clamp_min(1)
    return 1.0 / deg[dst]


def compute_gsmp_weights(
    src: torch.Tensor,
    dst: torch.Tensor,
    node_year: torch.Tensor,
    num_nodes: int,
    gsmp_norm: str,
    year_universe: str,
) -> tuple[torch.Tensor, int, int, int, int]:
    years = node_year.view(-1).long()
    min_year = int(years.min().item())
    max_year = int(years.max().item())
    if year_universe == "range":
        year_idx = years - min_year
        num_years = max_year - min_year + 1
    else:
        unique_years, year_idx = torch.unique(years, sorted=True, return_inverse=True)
        num_years = int(unique_years.numel())
    edge_year = year_idx[src]
    group_key = dst * int(num_years) + edge_year
    unique_keys, inverse, counts = torch.unique(group_key, sorted=False, return_inverse=True, return_counts=True)
    within_year_counts = counts[inverse].to(torch.float32)
    base = 1.0 / within_year_counts
    if gsmp_norm == "active_years":
        group_dst = torch.div(unique_keys, int(num_years), rounding_mode="floor")
        active_years = torch.bincount(group_dst, minlength=num_nodes).to(torch.float32).clamp_min(1)
        denom = active_years[dst]
    else:
        denom = torch.full_like(base, float(num_years))
    return base / denom, int(num_years), min_year, max_year, int(unique_keys.numel())


def run_seed(
    args: argparse.Namespace,
    experiment_name: str,
    component_dir: Path,
    csv_dir: Path,
    loaded: LoadedData,
    seed_value: int,
    device: torch.device,
) -> dict[str, object]:
    set_seed(seed_value)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = LinearRevGAT(
        loaded.n_node_feats + (loaded.n_classes if args.use_labels else 0),
        loaded.n_classes,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        activation=F.relu,
        dropout=args.dropout,
        input_drop=args.input_drop,
        edge_drop=args.edge_drop,
        use_symmetric_norm=args.use_norm,
        group=args.group,
        use_gpt_preds=args.use_gpt_preds,
    ).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f"MODEL params={sum(p.numel() for p in model.parameters() if p.requires_grad)} seed={seed_value}", flush=True)

    best_valid = -math.inf
    test_at_best = -math.inf
    best_epoch = -1
    pure_best_test = -math.inf
    pure_best_test_epoch = -1
    final_valid = -math.inf
    final_test = -math.inf
    best_pred_cpu = None
    last_epoch = 0
    start = time.time()
    per_seed_rows = []
    for epoch in range(1, args.n_epochs + 1):
        last_epoch = epoch
        lr = adjust_learning_rate(optimizer, args.lr, epoch)
        loss, train_acc_for_loss_subset = train_epoch(args, model, loaded, optimizer)
        should_eval = epoch == args.n_epochs or epoch % max(1, args.eval_every) == 0
        if not should_eval:
            print(
                f"RESULT experiment={experiment_name} seed={seed_value} epoch={epoch} "
                f"loss={loss:.6f} train_acc_subset={train_acc_for_loss_subset:.4f}",
                flush=True,
            )
            continue
        train_acc, valid_acc, test_acc, _, _, _, pred = evaluate(args, model, loaded)
        final_valid = valid_acc
        final_test = test_acc
        if test_acc > pure_best_test:
            pure_best_test = test_acc
            pure_best_test_epoch = epoch
        if valid_acc > best_valid:
            best_valid = valid_acc
            test_at_best = test_acc
            best_epoch = epoch
            best_pred_cpu = pred.detach().cpu()
        elapsed = time.time() - start
        row = make_epoch_row(
            args,
            experiment_name,
            seed_value,
            epoch,
            train_acc,
            valid_acc,
            test_acc,
            best_valid,
            test_at_best,
            best_epoch,
            pure_best_test,
            pure_best_test_epoch,
            loss,
            lr,
            elapsed,
            device,
        )
        per_seed_rows.append(row)
        append_csv(csv_dir / "per_epoch.csv", [row], CSV_FIELDS)
        append_csv(component_dir / "per_epoch.csv", [row], CSV_FIELDS)
        print(
            f"RESULT experiment={experiment_name} seed={seed_value} epoch={epoch} "
            f"train={train_acc:.4f} valid={valid_acc:.4f} test={test_acc:.4f} "
            f"best_valid={best_valid:.4f} test_at_best={test_at_best:.4f} loss={loss:.6f}",
            flush=True,
        )
        if (
            args.early_stop_patience > 0
            and epoch >= args.early_stop_min_epochs
            and best_epoch > 0
            and epoch - best_epoch >= args.early_stop_patience
        ):
            print(
                f"EARLY_STOP experiment={experiment_name} seed={seed_value} "
                f"epoch={epoch} best_epoch={best_epoch} patience={args.early_stop_patience}",
                flush=True,
            )
            break

    if args.save_pred and best_pred_cpu is not None:
        logits_dir = component_dir / "cached_embs"
        logits_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_pred_cpu, logits_dir / f"logits_seed{seed_value}.pt")
        print(f"Saved logits: {logits_dir / f'logits_seed{seed_value}.pt'}", flush=True)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": experiment_name,
        "model": "LinearRevGATGSMP" if args.model_variant == "gsmp" else "LinearRevGAT",
        "seed": seed_value,
        "dataset": args.dataset,
        "feature_source": args.feature_source,
        "gsmp_enabled": args.model_variant == "gsmp",
        "gsmp_norm": args.gsmp_norm if args.model_variant == "gsmp" else "none",
        "use_self_loops": not args.no_self_loops,
        "edge_direction": args.edge_direction,
        "best_valid_acc": best_valid,
        "test_acc_at_best_valid": test_at_best,
        "best_valid_epoch": best_epoch,
        "pure_best_test_acc": pure_best_test,
        "pure_best_test_epoch": pure_best_test_epoch,
        "final_valid_acc": final_valid,
        "final_test_acc": final_test,
        "epochs_run": last_epoch,
        "elapsed_seconds": time.time() - start,
        "gpu_name": gpu_name(device),
        "peak_gpu_mem_mb": peak_gpu_mem_mb(device),
        "git_commit_hash": git_commit_hash(REPO_DIR),
    }
    write_json(component_dir / f"summary_seed{seed_value}.json", summary)
    print(
        f"SUMMARY experiment={experiment_name} seed={seed_value} best_epoch={best_epoch} "
        f"best_valid={best_valid:.4f} test_at_best={test_at_best:.4f} "
        f"pure_best_test={pure_best_test:.4f}",
        flush=True,
    )
    return summary


def train_epoch(
    args: argparse.Namespace,
    model: LinearRevGAT,
    loaded: LoadedData,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    feat = loaded.graph.ndata["feat"]
    labels = loaded.labels
    if args.use_labels:
        mask = torch.rand(loaded.train_idx.shape, device=loaded.train_idx.device) < args.mask_rate
        train_labels_idx = loaded.train_idx[mask]
        train_pred_idx = loaded.train_idx[~mask]
        feat = add_labels(feat, labels, train_labels_idx, loaded.n_classes)
    else:
        mask = torch.rand(loaded.train_idx.shape, device=loaded.train_idx.device) < args.mask_rate
        train_pred_idx = loaded.train_idx[mask]

    optimizer.zero_grad(set_to_none=True)
    if args.n_label_iters > 0:
        with torch.no_grad():
            pred = model(loaded.graph, feat)
    else:
        pred = model(loaded.graph, feat)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([train_pred_idx, loaded.valid_idx, loaded.test_idx])
        for _ in range(args.n_label_iters):
            pred = pred.detach()
            if pred.is_cuda:
                torch.cuda.empty_cache()
            feat[unlabel_idx, -loaded.n_classes :] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(loaded.graph, feat)

    loss = custom_loss(pred[train_pred_idx], labels[train_pred_idx], args.label_smoothing_factor)
    loss.backward()
    optimizer.step()
    train_acc_subset = accuracy(pred[train_pred_idx], labels[train_pred_idx])
    return float(loss.detach().item()), float(train_acc_subset)


@torch.no_grad()
def evaluate(args: argparse.Namespace, model: LinearRevGAT, loaded: LoadedData):
    model.eval()
    feat = loaded.graph.ndata["feat"]
    labels = loaded.labels
    if args.use_labels:
        feat = add_labels(feat, labels, loaded.train_idx, loaded.n_classes)
    pred = model(loaded.graph, feat)
    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([loaded.valid_idx, loaded.test_idx])
        for _ in range(args.n_label_iters):
            feat[unlabel_idx, -loaded.n_classes :] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(loaded.graph, feat)
    train_loss = custom_loss(pred[loaded.train_idx], labels[loaded.train_idx], 0)
    valid_loss = custom_loss(pred[loaded.valid_idx], labels[loaded.valid_idx], 0)
    test_loss = custom_loss(pred[loaded.test_idx], labels[loaded.test_idx], 0)
    return (
        accuracy(pred[loaded.train_idx], labels[loaded.train_idx]),
        accuracy(pred[loaded.valid_idx], labels[loaded.valid_idx]),
        accuracy(pred[loaded.test_idx], labels[loaded.test_idx]),
        float(train_loss.item()),
        float(valid_loss.item()),
        float(test_loss.item()),
        pred,
    )


def custom_loss(logits: torch.Tensor, labels: torch.Tensor, label_smoothing_factor: float) -> torch.Tensor:
    loss = F.cross_entropy(logits, labels[:, 0], reduction="none", label_smoothing=label_smoothing_factor)
    loss = torch.log(EPSILON + loss) - math.log(EPSILON)
    return torch.mean(loss)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1, keepdim=True)
    return float((pred == labels).float().mean().item())


def add_labels(feat: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor, n_classes: int) -> torch.Tensor:
    onehot = torch.zeros((feat.shape[0], n_classes), device=feat.device, dtype=torch.float32)
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer: torch.optim.Optimizer, lr: float, epoch: int) -> float:
    current_lr = lr * epoch / 50 if epoch <= 50 else lr
    for group in optimizer.param_groups:
        group["lr"] = current_lr
    return current_lr


def load_gpt_preds(path: Path) -> torch.Tensor:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            rows.append([int(value) for value in row if value != ""])
    out = torch.zeros((len(rows), 5), dtype=torch.long)
    for idx, row in enumerate(rows):
        if row:
            out[idx, : min(5, len(row))] = torch.tensor(row[:5], dtype=torch.long) + 1
    return out


def make_epoch_row(
    args: argparse.Namespace,
    experiment_name: str,
    seed_value: int,
    epoch: int,
    train_acc: float,
    valid_acc: float,
    test_acc: float,
    best_valid: float,
    test_at_best: float,
    best_epoch: int,
    pure_best_test: float,
    pure_best_test_epoch: int,
    loss: float,
    lr: float,
    elapsed: float,
    device: torch.device,
) -> dict[str, object]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": experiment_name,
        "model": "LinearRevGATGSMP" if args.model_variant == "gsmp" else "LinearRevGAT",
        "seed": seed_value,
        "dataset": args.dataset,
        "feature_source": args.feature_source,
        "gsmp_enabled": args.model_variant == "gsmp",
        "gsmp_norm": args.gsmp_norm if args.model_variant == "gsmp" else "none",
        "use_self_loops": not args.no_self_loops,
        "edge_direction": args.edge_direction,
        "epoch": epoch,
        "train_acc": train_acc,
        "valid_acc": valid_acc,
        "test_acc": test_acc,
        "best_valid_acc_so_far": best_valid,
        "test_acc_at_best_valid_so_far": test_at_best,
        "best_valid_epoch": best_epoch,
        "pure_best_test_acc_so_far": pure_best_test,
        "pure_best_test_epoch": pure_best_test_epoch,
        "loss": loss,
        "lr": lr,
        "elapsed_seconds": elapsed,
        "gpu_name": gpu_name(device),
        "peak_gpu_mem_mb": peak_gpu_mem_mb(device),
        "git_commit_hash": git_commit_hash(REPO_DIR),
    }


def aggregate_summary(args: argparse.Namespace, experiment_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    best_valid = [float(row["best_valid_acc"]) for row in rows]
    test_at_best = [float(row["test_acc_at_best_valid"]) for row in rows]
    pure_best = [float(row["pure_best_test_acc"]) for row in rows]
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": experiment_name,
        "model": "LinearRevGATGSMP" if args.model_variant == "gsmp" else "LinearRevGAT",
        "dataset": args.dataset,
        "feature_source": args.feature_source,
        "gsmp_enabled": args.model_variant == "gsmp",
        "gsmp_norm": args.gsmp_norm if args.model_variant == "gsmp" else "none",
        "num_seeds": len(rows),
        "best_valid_acc_mean": mean(best_valid),
        "best_valid_acc_std": std(best_valid),
        "test_acc_at_best_valid_mean": mean(test_at_best),
        "test_acc_at_best_valid_std": std(test_at_best),
        "pure_best_test_acc_mean_diagnostic_only": mean(pure_best),
        "pure_best_test_acc_std_diagnostic_only": std(pure_best),
        "seeds": " ".join(str(row["seed"]) for row in rows),
    }


def append_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def resolve_path(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("Cannot resolve None path.")
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


def parse_seeds(value: str) -> list[int]:
    return [int(part) for part in value.replace(",", " ").split() if part.strip()]


def default_experiment_name(args: argparse.Namespace) -> str:
    variant = "linear_gsmp" if args.model_variant == "gsmp" else "linear_no_gsmp"
    return f"{variant}_{args.feature_source}"


def set_seed(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed_value)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def target_weight_sum_mean(dst: torch.Tensor, weights: torch.Tensor, num_nodes: int) -> float:
    sums = torch.zeros(num_nodes, dtype=torch.float32)
    sums.scatter_add_(0, dst, weights.to(torch.float32))
    active = sums > 0
    return sums[active].mean().item() if active.any() else 0.0


def gpu_name(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(device)


def peak_gpu_mem_mb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def git_commit_hash(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def print_header(args: argparse.Namespace, experiment_name: str, component_dir: Path, device: torch.device) -> None:
    print("=== 260610 Linearized RevGAT GSMP experiment ===", flush=True)
    print(f"date={datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"project_dir={PROJECT_DIR}", flush=True)
    print(f"experiment_name={experiment_name}", flush=True)
    print(f"component_dir={component_dir}", flush=True)
    print(f"device={device}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"dgl={dgl.__version__}", flush=True)
    print("args=" + json.dumps(vars(args), sort_keys=True), flush=True)


def preflight_cuda_runtime(device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Use CPU=1 for a CPU smoke test or run on a GPU node."
        )
    try:
        tiny = dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=1)
        tiny.ndata["feat"] = torch.zeros(1, 1)
        tiny.edata["mp_weight"] = torch.ones(1)
        tiny.to(device)
    except Exception as exc:
        raise RuntimeError(
            "CUDA was requested, but this DGL installation cannot move graphs to CUDA. "
            "The current environment appears to have CPU-only DGL. Install a CUDA-enabled "
            "DGL build for GPU RevGAT runs, or use CPU=1 for a no-GPU smoke test."
        ) from exc


if __name__ == "__main__":
    main()
