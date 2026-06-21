#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
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

from gsmp_utils import (
    apply_pgsmp_preprocess,
    compute_year_balanced_edge_weight,
    edge_index_from_src_dst,
    fingerprint_path,
    make_cache_name,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_SIMTEG_DIR = REPO_DIR / "SimTeG"
if DEFAULT_SIMTEG_DIR.is_dir():
    sys.path.insert(0, str(DEFAULT_SIMTEG_DIR))

with contextlib.redirect_stdout(io.StringIO()):
    from src.misc.revgat.model_rev import ElementWiseLinear  # noqa: E402
    from src.misc.revgat.rev import memgcn  # noqa: E402
    from src.misc.revgat.rev.rev_layer import SharedDropout  # noqa: E402


METHOD_BASE = "SimTeG+TAPE+linearRevGAT"
EPSILON = 1 - math.log(2)
EXPERIMENT_MODES = ("baseline", "gsmp_first_layer", "pgsmp", "pgsmp_plus_gsmp_first_layer")
CSV_FIELDS = [
    "method",
    "experiment_mode",
    "seed",
    "epoch",
    "val_acc",
    "test_acc",
    "best_val",
    "test_at_best_val",
    "best_epoch",
    "best_raw_test",
    "lr",
    "feature_source",
    "use_gsmp_first_layer",
    "gsmp_layer",
    "gsmp_norm",
    "use_pgsmp",
    "pgsmp_norm",
    "pgsmp_alpha",
    "pgsmp_depth",
    "pgsmp_self_mode",
    "gpu_mem_mb",
    "epoch_time",
    "train_acc",
    "loss",
    "elapsed_seconds",
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
    preprocessing_time: float
    cache_reused: bool


class LinearMeanConv(nn.Module):
    """RevGAT-compatible fixed linear aggregation.

    DGL stores edges as source -> destination. With no GSMP weight this layer is
    ordinary per-destination mean aggregation. If a GSMP multiplier is supplied,
    it is applied before the same mean aggregation, which makes
    `scale_preserve` weights preserve the ordinary mean scale.
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

    def forward(
        self,
        graph: dgl.DGLGraph,
        feat: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        edge_weight: torch.Tensor | None = None,
        perm: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

            if "mean_weight" not in graph.edata:
                raise KeyError('Missing graph.edata["mean_weight"]. Run attach_graph_weights first.')
            mean_weight = graph.edata["mean_weight"].to(device=feat_src.device, dtype=feat_src.dtype).view(-1)
            if edge_weight is None:
                msg_weight = mean_weight
            else:
                msg_weight = mean_weight * edge_weight.to(device=feat_src.device, dtype=feat_src.dtype).view(-1)
            msg_weight = self._apply_edge_dropout(graph, msg_weight, perm)

            graph.srcdata["ft"] = feat_src
            graph.edata["w"] = msg_weight.view(-1, 1, 1)
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
        self.conv = LinearMeanConv(
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
        return self.conv(graph, out, edge_weight=None, perm=perm).flatten(1, -1)


class LinearRevGAT(nn.Module):
    """RevGAT shell with graph attention replaced by fixed linear mean aggregation."""

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
        group: int = 1,
        use_gpt_preds: bool = False,
        input_norm: bool = True,
    ):
        super().__init__()
        if n_layers < 2:
            raise ValueError("LinearRevGAT requires at least two layers, matching the official RevGAT recipe.")
        self.in_feats = int(in_feats)
        self.n_hidden = int(n_hidden)
        self.n_classes = int(n_classes)
        self.n_layers = int(n_layers)
        self.num_heads = int(n_heads)
        self.group = int(group)
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
                    LinearMeanConv(
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
        self.last_gsmp_layers_used: list[int] = []

    def _edge_weight_for_layer(self, layer_idx: int, gsmp_edge_weight: torch.Tensor | None, gsmp_layer: int = 0):
        if gsmp_edge_weight is not None and int(layer_idx) == int(gsmp_layer):
            return gsmp_edge_weight
        return None

    def forward(
        self,
        graph: dgl.DGLGraph,
        feat: torch.Tensor,
        gsmp_edge_weight: torch.Tensor | None = None,
        gsmp_layer: int = 0,
    ) -> torch.Tensor:
        x = feat
        self.last_gsmp_layers_used = []
        if hasattr(self, "encoder"):
            embs = self.encoder(x[:, :5].to(torch.long))
            x = torch.cat([torch.flatten(embs, start_dim=1), x[:, 5:]], dim=1)
        if hasattr(self, "input_norm"):
            x = self.input_norm(x)
        x = self.input_drop(x)

        perms = [torch.randperm(graph.number_of_edges(), device=graph.device) for _ in range(self.n_layers)]
        layer_weight = self._edge_weight_for_layer(0, gsmp_edge_weight, gsmp_layer)
        if layer_weight is not None:
            self.last_gsmp_layers_used.append(0)
        x = self.convs[0](graph, x, edge_weight=layer_weight, perm=perms[0]).flatten(1, -1)

        mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
        mask = mask.requires_grad_(False) / max(torch.finfo(x.dtype).eps, 1 - self.dropout)
        for layer_idx in range(1, self.n_layers - 1):
            graph.requires_grad = False
            perm = torch.stack([perms[layer_idx]] * self.group, dim=1)
            x = self.convs[layer_idx](x, graph, mask, perm)

        x = self.norm(x)
        x = self.activation(x, inplace=True)
        x = self.dp_last(x)
        layer_weight = self._edge_weight_for_layer(self.n_layers - 1, gsmp_edge_weight, gsmp_layer)
        if layer_weight is not None:
            self.last_gsmp_layers_used.append(self.n_layers - 1)
        x = self.convs[-1](graph, x, edge_weight=layer_weight, perm=perms[-1])
        x = x.mean(1)
        return self.bias_last(x)


def copy_module(module: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("SimTeG/TAPE LinearRevGAT with GSMP-first-layer and P-GSMP.")
    parser.add_argument("--experiment-mode", choices=EXPERIMENT_MODES, default="baseline")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset", default="ogbn-arxiv")
    parser.add_argument("--feature-source", default="unknown")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-epochs", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dgl-data-root", default="../dgl_data")
    parser.add_argument("--output-root", default="results/simteg_tape_linearrevgat_gsmp")
    parser.add_argument("--cache-root", default="cache")
    parser.add_argument("--use-bert-x", action="store_true")
    parser.add_argument("--bert-x-dir", default=None)
    parser.add_argument("--use-gpt-preds", action="store_true")
    parser.add_argument("--gpt-preds-path", default="../260609/resources/ogbn-arxiv-gpt-preds.csv")
    parser.add_argument("--use-labels", action="store_true")
    parser.add_argument("--n-label-iters", type=int, default=2)
    parser.add_argument("--mask-rate", type=float, default=0.5)
    parser.add_argument("--use-norm", action="store_true", default=True)
    parser.add_argument("--no-use-norm", action="store_false", dest="use_norm")
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.58)
    parser.add_argument("--input-drop", type=float, default=0.37)
    parser.add_argument("--edge-drop", type=float, default=0.0)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--label-smoothing-factor", type=float, default=0.02)
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--save-pred", action="store_true")
    parser.add_argument("--edge-direction", choices=["bidirected", "original", "reverse"], default="bidirected")
    parser.add_argument("--no-self-loops", action="store_true")
    parser.add_argument("--use-gsmp-first-layer", action="store_true")
    parser.add_argument("--gsmp-layer", type=int, default=0)
    parser.add_argument("--gsmp-norm", choices=["scale_preserve", "strict_observed"], default="scale_preserve")
    parser.add_argument("--gsmp-cache-dir", default=None)
    parser.add_argument("--gsmp-force-recompute", action="store_true")
    parser.add_argument("--use-pgsmp", action="store_true")
    parser.add_argument("--pgsmp-alpha", type=float, default=0.5)
    parser.add_argument("--pgsmp-depth", type=int, default=1)
    parser.add_argument("--pgsmp-norm", choices=["strict_observed", "scale_preserve"], default="strict_observed")
    parser.add_argument("--pgsmp-self-mode", choices=["neighbor_only", "include_self"], default="neighbor_only")
    parser.add_argument("--pgsmp-cache-dir", default=None)
    parser.add_argument("--pgsmp-force-recompute", action="store_true")
    parser.add_argument("--pgsmp-chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.use_bert_x == args.use_gpt_preds:
        raise ValueError("Select exactly one feature source: --use-bert-x or --use-gpt-preds.")
    if args.use_bert_x and not args.bert_x_dir:
        raise ValueError("--bert-x-dir is required with --use-bert-x.")
    if args.n_epochs < 1:
        raise ValueError("--n-epochs must be >= 1.")
    if args.gsmp_layer != 0:
        raise ValueError("This experiment intentionally supports first-layer-only GSMP: --gsmp-layer must be 0.")
    apply_mode_defaults(args)
    if args.use_pgsmp and args.use_gpt_preds:
        raise ValueError("P-GSMP expects real-valued features and is disabled for GPT-prediction label features.")
    return args


def apply_mode_defaults(args: argparse.Namespace) -> None:
    if args.experiment_mode == "baseline":
        args.use_gsmp_first_layer = False
        args.use_pgsmp = False
    elif args.experiment_mode == "gsmp_first_layer":
        args.use_gsmp_first_layer = True
        args.use_pgsmp = False
    elif args.experiment_mode == "pgsmp":
        args.use_gsmp_first_layer = False
        args.use_pgsmp = True
    elif args.experiment_mode == "pgsmp_plus_gsmp_first_layer":
        args.use_gsmp_first_layer = True
        args.use_pgsmp = True


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config_dict(args, seeds, run_dir))

    device = torch.device("cpu" if args.cpu else f"cuda:{args.gpu}")
    if device.type == "cuda":
        preflight_cuda_runtime(device)
    print_header(args, run_id, run_dir, device)
    loaded = load_data(args)
    loaded.graph = preprocess_graph(args, loaded.graph, loaded.node_year)
    loaded = maybe_apply_pgsmp(args, loaded)
    loaded.graph, loaded.labels, loaded.train_idx, loaded.valid_idx, loaded.test_idx = [
        item.to(device) for item in [loaded.graph, loaded.labels, loaded.train_idx, loaded.valid_idx, loaded.test_idx]
    ]

    summaries = []
    for seed_value in seeds:
        summary = run_seed(args, run_dir, loaded, seed_value, device)
        summaries.append(summary)
        append_csv(run_dir / "seed_summary.csv", [summary], list(summary.keys()))

    final = final_summary(args, run_id, summaries)
    write_json(run_dir / "final_summary.json", final)
    print_final(final)


def load_data(args: argparse.Namespace) -> LoadedData:
    dataset = DglNodePropPredDataset(name=args.dataset, root=str(resolve_path(args.dgl_data_root)))
    evaluator = Evaluator(name=args.dataset)
    split = dataset.get_idx_split()
    graph, labels = dataset[0]
    node_year = get_node_year(graph).view(-1).long()
    if args.use_bert_x:
        feature_path = resolve_path(args.bert_x_dir)
        feat = torch.load(feature_path, map_location="cpu", weights_only=False).float()
        graph.ndata["feat"] = feat
        n_node_feats = int(feat.shape[1])
        print(f"[FEATURE] source=cached_embedding path={feature_path} shape={tuple(feat.shape)}", flush=True)
    else:
        feature_path = resolve_path(args.gpt_preds_path)
        feat = load_gpt_preds(feature_path)
        graph.ndata["feat"] = feat
        n_node_feats = args.n_hidden * 5
        print(f"[FEATURE] source=gpt_preds path={feature_path} shape={tuple(feat.shape)}", flush=True)
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
        preprocessing_time=0.0,
        cache_reused=False,
    )


def get_node_year(graph: dgl.DGLGraph) -> torch.Tensor:
    for key in ("year", "node_year", "node_years", "years"):
        if key in graph.ndata:
            return graph.ndata[key]
    raise KeyError('Expected OGBN-Arxiv DGL graph to expose node years in graph.ndata["year"].')


def preprocess_graph(args: argparse.Namespace, graph: dgl.DGLGraph, node_year: torch.Tensor) -> dgl.DGLGraph:
    ndata = {key: value for key, value in graph.ndata.items()}
    before_edges = graph.number_of_edges()
    if args.edge_direction == "bidirected":
        graph = dgl.to_bidirected(graph, copy_ndata=False)
    elif args.edge_direction == "reverse":
        graph = dgl.reverse(graph, copy_ndata=False)
    for key, value in ndata.items():
        graph.ndata[key] = value
    if args.no_self_loops:
        graph = graph.remove_self_loop()
    else:
        graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    attach_graph_weights(args, graph, node_year)
    print(
        "[GRAPH] "
        f"edge_direction={args.edge_direction} edges_before={before_edges} edges_after={graph.number_of_edges()} "
        f"self_loops={not args.no_self_loops} message_direction=src_to_dst",
        flush=True,
    )
    return graph


def attach_graph_weights(args: argparse.Namespace, graph: dgl.DGLGraph, node_year: torch.Tensor) -> None:
    src, dst = graph.edges(order="eid")
    src = src.cpu().long()
    dst = dst.cpu().long()
    num_nodes = graph.num_nodes()
    deg = torch.bincount(dst, minlength=num_nodes).float().clamp_min(1.0)
    graph.edata["mean_weight"] = (1.0 / deg[dst]).float()
    if not args.use_gsmp_first_layer:
        return
    edge_index = edge_index_from_src_dst(src, dst)
    cache_dir = resolve_path(args.gsmp_cache_dir or (Path(args.cache_root) / "gsmp"))
    cache_name = make_cache_name(
        "gsmp",
        {
            "dataset": args.dataset,
            "direction": "src_to_dst",
            "undirected": args.edge_direction == "bidirected",
            "self_loop": not args.no_self_loops,
            "norm": args.gsmp_norm,
            "num_edges": int(edge_index.size(1)),
            "num_nodes": int(num_nodes),
        },
    )
    weight = compute_year_balanced_edge_weight(
        edge_index,
        node_year,
        num_nodes,
        mode=args.gsmp_norm,
        cache_path=cache_dir / cache_name,
        force_recompute=args.gsmp_force_recompute,
    )
    graph.edata["gsmp_weight"] = weight.float()


def maybe_apply_pgsmp(args: argparse.Namespace, loaded: LoadedData) -> LoadedData:
    if not args.use_pgsmp:
        return loaded
    start = time.time()
    src, dst = loaded.graph.edges(order="eid")
    edge_index = edge_index_from_src_dst(src, dst)
    x = loaded.graph.ndata["feat"]
    cache_dir = resolve_path(args.pgsmp_cache_dir or (Path(args.cache_root) / "pgsmp"))
    feature_fingerprint = fingerprint_path(args.bert_x_dir if args.use_bert_x else args.gpt_preds_path)
    cache_name = make_cache_name(
        "pgsmp",
        {
            "dataset": args.dataset,
            "direction": "src_to_dst",
            "undirected": args.edge_direction == "bidirected",
            "self_loop": not args.no_self_loops,
            "norm": args.pgsmp_norm,
            "alpha": args.pgsmp_alpha,
            "depth": args.pgsmp_depth,
            "self_mode": args.pgsmp_self_mode,
            "xshape": f"{x.size(0)}x{x.size(1)}",
            "feature": f"{args.feature_source}-{feature_fingerprint}",
            "num_edges": int(edge_index.size(1)),
            "num_nodes": int(loaded.graph.num_nodes()),
        },
    )
    cache_path = cache_dir / cache_name
    cache_reused = cache_path.is_file() and not args.pgsmp_force_recompute
    loaded.graph.ndata["feat"] = apply_pgsmp_preprocess(
        x,
        edge_index,
        loaded.node_year,
        loaded.graph.num_nodes(),
        alpha=args.pgsmp_alpha,
        depth=args.pgsmp_depth,
        norm=args.pgsmp_norm,
        self_mode=args.pgsmp_self_mode,
        cache_path=cache_path,
        chunk_size=args.pgsmp_chunk_size,
        force_recompute=args.pgsmp_force_recompute,
    )
    loaded.preprocessing_time = time.time() - start
    loaded.cache_reused = cache_reused
    return loaded


def run_seed(
    args: argparse.Namespace,
    run_dir: Path,
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
    print(f"[MODEL] params={sum(p.numel() for p in model.parameters() if p.requires_grad)} seed={seed_value}", flush=True)

    best_val = -math.inf
    test_at_best_val = -math.inf
    best_epoch = -1
    best_raw_test = -math.inf
    best_raw_test_epoch = -1
    best_pred_cpu = None
    last_epoch = 0
    start = time.time()

    for epoch in range(1, args.n_epochs + 1):
        last_epoch = epoch
        tic = time.time()
        lr = adjust_learning_rate(optimizer, args.lr, epoch)
        loss, train_acc_subset = train_epoch(args, model, loaded, optimizer)
        epoch_time = time.time() - tic
        should_eval = epoch == args.n_epochs or epoch % max(1, args.eval_every) == 0
        if not should_eval:
            continue

        train_acc, val_acc, test_acc, pred = evaluate(args, model, loaded)
        if test_acc > best_raw_test:
            best_raw_test = test_acc
            best_raw_test_epoch = epoch
        if val_acc > best_val:
            best_val = val_acc
            test_at_best_val = test_acc
            best_epoch = epoch
            best_pred_cpu = pred.detach().cpu()
        row = make_epoch_row(
            args,
            seed_value,
            epoch,
            train_acc,
            val_acc,
            test_acc,
            best_val,
            test_at_best_val,
            best_epoch,
            best_raw_test,
            lr,
            train_acc_subset,
            loss,
            time.time() - start,
            epoch_time,
            device,
        )
        append_csv(run_dir / "epoch_logs.csv", [row], CSV_FIELDS)
        if should_print_epoch(args, epoch):
            print_result(args, row)
        if (
            args.early_stop_patience > 0
            and epoch >= args.early_stop_min_epochs
            and best_epoch > 0
            and epoch - best_epoch >= args.early_stop_patience
        ):
            print(
                f"[EARLY_STOP] mode={args.experiment_mode} seed={seed_value} "
                f"epoch={epoch} best_epoch={best_epoch} patience={args.early_stop_patience}",
                flush=True,
            )
            break

    if args.save_pred and best_pred_cpu is not None:
        logits_dir = run_dir / "cached_embs"
        logits_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_pred_cpu, logits_dir / f"logits_seed{seed_value}.pt")
        print(f"[LOGITS] saved={logits_dir / f'logits_seed{seed_value}.pt'}", flush=True)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": method_name(args),
        "experiment_mode": args.experiment_mode,
        "seed": seed_value,
        "dataset": args.dataset,
        "feature_source": args.feature_source,
        "use_gsmp_first_layer": bool(args.use_gsmp_first_layer),
        "gsmp_layer": int(args.gsmp_layer),
        "gsmp_norm": args.gsmp_norm if args.use_gsmp_first_layer else "none",
        "use_pgsmp": bool(args.use_pgsmp),
        "pgsmp_norm": args.pgsmp_norm if args.use_pgsmp else "none",
        "pgsmp_alpha": float(args.pgsmp_alpha) if args.use_pgsmp else 0.0,
        "pgsmp_depth": int(args.pgsmp_depth) if args.use_pgsmp else 0,
        "pgsmp_self_mode": args.pgsmp_self_mode if args.use_pgsmp else "none",
        "best_val": best_val,
        "test_at_best_val": test_at_best_val,
        "best_epoch": best_epoch,
        "best_raw_test": best_raw_test,
        "best_raw_test_epoch": best_raw_test_epoch,
        "epochs_run": last_epoch,
        "runtime": time.time() - start,
        "peak_gpu_mem_mb": peak_gpu_mem_mb(device),
        "preprocessing_time": loaded.preprocessing_time,
        "cache_reused": loaded.cache_reused,
        "gpu_name": gpu_name(device),
    }
    print_seed_summary(summary)
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
    gsmp_weight = get_graph_gsmp_weight(args, loaded.graph)
    if args.n_label_iters > 0:
        with torch.no_grad():
            pred = model(loaded.graph, feat, gsmp_edge_weight=gsmp_weight, gsmp_layer=args.gsmp_layer)
    else:
        pred = model(loaded.graph, feat, gsmp_edge_weight=gsmp_weight, gsmp_layer=args.gsmp_layer)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([train_pred_idx, loaded.valid_idx, loaded.test_idx])
        for _ in range(args.n_label_iters):
            pred = pred.detach()
            if pred.is_cuda:
                torch.cuda.empty_cache()
            feat[unlabel_idx, -loaded.n_classes :] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(loaded.graph, feat, gsmp_edge_weight=gsmp_weight, gsmp_layer=args.gsmp_layer)

    loss = custom_loss(pred[train_pred_idx], labels[train_pred_idx], args.label_smoothing_factor)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item()), accuracy(pred[train_pred_idx], labels[train_pred_idx])


@torch.no_grad()
def evaluate(args: argparse.Namespace, model: LinearRevGAT, loaded: LoadedData):
    model.eval()
    feat = loaded.graph.ndata["feat"]
    labels = loaded.labels
    if args.use_labels:
        feat = add_labels(feat, labels, loaded.train_idx, loaded.n_classes)
    gsmp_weight = get_graph_gsmp_weight(args, loaded.graph)
    pred = model(loaded.graph, feat, gsmp_edge_weight=gsmp_weight, gsmp_layer=args.gsmp_layer)
    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([loaded.valid_idx, loaded.test_idx])
        for _ in range(args.n_label_iters):
            feat[unlabel_idx, -loaded.n_classes :] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(loaded.graph, feat, gsmp_edge_weight=gsmp_weight, gsmp_layer=args.gsmp_layer)
    return (
        accuracy(pred[loaded.train_idx], labels[loaded.train_idx]),
        accuracy(pred[loaded.valid_idx], labels[loaded.valid_idx]),
        accuracy(pred[loaded.test_idx], labels[loaded.test_idx]),
        pred,
    )


def get_graph_gsmp_weight(args: argparse.Namespace, graph: dgl.DGLGraph) -> torch.Tensor | None:
    if not args.use_gsmp_first_layer:
        return None
    if "gsmp_weight" not in graph.edata:
        raise KeyError('Missing graph.edata["gsmp_weight"] for GSMP-first-layer mode.')
    return graph.edata["gsmp_weight"]


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
    seed_value: int,
    epoch: int,
    train_acc: float,
    val_acc: float,
    test_acc: float,
    best_val: float,
    test_at_best_val: float,
    best_epoch: int,
    best_raw_test: float,
    lr: float,
    train_acc_subset: float,
    loss: float,
    elapsed: float,
    epoch_time: float,
    device: torch.device,
) -> dict[str, object]:
    return {
        "method": method_name(args),
        "experiment_mode": args.experiment_mode,
        "seed": seed_value,
        "epoch": epoch,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "best_val": best_val,
        "test_at_best_val": test_at_best_val,
        "best_epoch": best_epoch,
        "best_raw_test": best_raw_test,
        "lr": lr,
        "feature_source": args.feature_source,
        "use_gsmp_first_layer": bool(args.use_gsmp_first_layer),
        "gsmp_layer": int(args.gsmp_layer),
        "gsmp_norm": args.gsmp_norm if args.use_gsmp_first_layer else "none",
        "use_pgsmp": bool(args.use_pgsmp),
        "pgsmp_norm": args.pgsmp_norm if args.use_pgsmp else "none",
        "pgsmp_alpha": float(args.pgsmp_alpha) if args.use_pgsmp else 0.0,
        "pgsmp_depth": int(args.pgsmp_depth) if args.use_pgsmp else 0,
        "pgsmp_self_mode": args.pgsmp_self_mode if args.use_pgsmp else "none",
        "gpu_mem_mb": peak_gpu_mem_mb(device),
        "epoch_time": epoch_time,
        "train_acc": train_acc,
        "loss": loss,
        "elapsed_seconds": elapsed,
    }


def final_summary(args: argparse.Namespace, run_id: str, rows: list[dict[str, object]]) -> dict[str, object]:
    vals = [float(row["best_val"]) for row in rows]
    tests = [float(row["test_at_best_val"]) for row in rows]
    raw = [float(row["best_raw_test"]) for row in rows]
    epochs = [float(row["best_epoch"]) for row in rows]
    runtimes = [float(row["runtime"]) for row in rows]
    mem = [float(row["peak_gpu_mem_mb"]) for row in rows]
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "method": method_name(args),
        "experiment_mode": args.experiment_mode,
        "feature_source": args.feature_source,
        "num_seeds": len(rows),
        "seeds": ",".join(str(row["seed"]) for row in rows),
        "val_mean": mean(vals),
        "val_std": std(vals),
        "test_at_best_val_mean": mean(tests),
        "test_at_best_val_std": std(tests),
        "best_raw_test_mean_diagnostic_only": mean(raw),
        "best_raw_test_std_diagnostic_only": std(raw),
        "best_epoch_mean": mean(epochs),
        "best_epoch_std": std(epochs),
        "runtime_mean": mean(runtimes),
        "peak_gpu_memory_mean": mean(mem),
        "preprocessing_time": max((float(row["preprocessing_time"]) for row in rows), default=0.0),
        "cache_reused": all(bool(row["cache_reused"]) for row in rows) if rows else False,
        "main_metric": "test_at_best_val_mean",
        "anchor_warning": (
            "SimTeG+TAPE+RevGAT leaderboard anchor is not reproduced by this LinearRevGAT run. "
            "Interpret linearRevGAT GSMP comparisons as internal ablations only."
        ),
    }


def method_name(args: argparse.Namespace) -> str:
    if args.experiment_mode == "gsmp_first_layer":
        return f"{METHOD_BASE}+GSMP1"
    if args.experiment_mode == "pgsmp":
        return f"{METHOD_BASE}+P-GSMP"
    if args.experiment_mode == "pgsmp_plus_gsmp_first_layer":
        return f"{METHOD_BASE}+P-GSMP+GSMP1"
    return METHOD_BASE


def should_print_epoch(args: argparse.Namespace, epoch: int) -> bool:
    return epoch == 1 or epoch == args.n_epochs or epoch % max(1, args.log_every) == 0 or args.n_epochs <= 5


def print_result(args: argparse.Namespace, row: dict[str, object]) -> None:
    prefix = (
        f"[RESULT] method={row['method']} mode={row['experiment_mode']} "
        f"seed={row['seed']} epoch={int(row['epoch']):03d} "
    )
    if args.experiment_mode == "gsmp_first_layer":
        prefix += f"gsmp_layer={args.gsmp_layer} gsmp_norm={args.gsmp_norm} "
    elif args.experiment_mode.startswith("pgsmp"):
        prefix += (
            f"pgsmp_norm={args.pgsmp_norm} pgsmp_alpha={args.pgsmp_alpha} "
            f"pgsmp_depth={args.pgsmp_depth} pgsmp_self_mode={args.pgsmp_self_mode} "
        )
    print(
        prefix
        + f"val_acc={float(row['val_acc']):.4f} test_acc={float(row['test_acc']):.4f} "
        + f"best_val={float(row['best_val']):.4f} "
        + f"test_at_best_val={float(row['test_at_best_val']):.4f} "
        + f"best_epoch={row['best_epoch']} lr={float(row['lr']):.6g} "
        + f"epoch_time={float(row['epoch_time']):.2f} gpu_mem_mb={float(row['gpu_mem_mb']):.1f}",
        flush=True,
    )


def print_seed_summary(summary: dict[str, object]) -> None:
    print(
        f"[SEED_SUMMARY] method={summary['method']} mode={summary['experiment_mode']} "
        f"seed={summary['seed']} best_val={float(summary['best_val']):.4f} "
        f"test_at_best_val={float(summary['test_at_best_val']):.4f} "
        f"best_epoch={summary['best_epoch']} best_raw_test={float(summary['best_raw_test']):.4f} "
        f"total_time={float(summary['runtime']):.2f} "
        f"peak_gpu_mem_mb={float(summary['peak_gpu_mem_mb']):.1f}",
        flush=True,
    )


def print_final(summary: dict[str, object]) -> None:
    print(
        f"[FINAL] method={summary['method']} mode={summary['experiment_mode']} "
        f"seeds={summary['seeds']} val_mean={float(summary['val_mean']):.4f} "
        f"val_std={float(summary['val_std']):.4f} "
        f"test_at_best_val_mean={float(summary['test_at_best_val_mean']):.4f} "
        f"test_at_best_val_std={float(summary['test_at_best_val_std']):.4f} "
        f"best_raw_test_mean={float(summary['best_raw_test_mean_diagnostic_only']):.4f} "
        f"best_raw_test_std={float(summary['best_raw_test_std_diagnostic_only']):.4f} "
        f"runtime_mean={float(summary['runtime_mean']):.2f}",
        flush=True,
    )


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


def config_dict(args: argparse.Namespace, seeds: list[int], run_dir: Path) -> dict[str, object]:
    return {
        "args": vars(args),
        "seeds": seeds,
        "run_dir": str(run_dir),
        "git_commit_hash": git_commit_hash(REPO_DIR),
        "simteg_dir": str(DEFAULT_SIMTEG_DIR),
        "leaderboard_anchor": {
            "method": "SimTeG+TAPE+RevGAT",
            "validation_accuracy": "0.7846 +/- 0.0004",
            "test_accuracy": "0.7803 +/- 0.0007",
        },
    }


def resolve_path(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("Cannot resolve None path.")
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


def parse_seeds(value: str) -> list[int]:
    return [int(part) for part in value.replace(",", " ").split() if part.strip()]


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


def print_header(args: argparse.Namespace, run_id: str, run_dir: Path, device: torch.device) -> None:
    print("=== 260612_3 SimTeG/TAPE LinearRevGAT GSMP-first-layer/P-GSMP experiment ===", flush=True)
    print(f"date={datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"project_dir={PROJECT_DIR}", flush=True)
    print(f"run_id={run_id}", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    print(f"device={device}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"dgl={dgl.__version__}", flush=True)
    print("args=" + json.dumps(vars(args), sort_keys=True), flush=True)
    print("[WARNING] SimTeG+TAPE+RevGAT leaderboard anchor was not reproduced in this run.", flush=True)
    print("[WARNING] Interpret linearRevGAT GSMP comparisons as internal ablations only.", flush=True)


def preflight_cuda_runtime(device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Use CPU=1 for a CPU smoke test or run on a GPU node."
        )
    try:
        tiny = dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=1)
        tiny.ndata["feat"] = torch.zeros(1, 1)
        tiny.edata["mean_weight"] = torch.ones(1)
        tiny.to(device)
    except Exception as exc:
        raise RuntimeError(
            "CUDA was requested, but this DGL installation cannot move graphs to CUDA. "
            "Install/load CUDA-enabled DGL or use CPU=1 for a no-GPU smoke test."
        ) from exc


if __name__ == "__main__":
    main()

