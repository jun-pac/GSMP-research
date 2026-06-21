import argparse
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import LINKX, MLP

from gsmp import compute_gsmp_edge_weights
from pokec_raw_linkx_smp import (
    PROFILES_URL,
    RELATIONSHIPS_URL,
    build_temporal_split,
    compute_smp_edge_weight,
    download,
    existing_path,
    load_profiles,
    load_relationships,
    make_linkx_sparse_adj,
    normalize_features,
    set_seed,
)


class GCNEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("GCN encoder needs at least one layer.")
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, hidden_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))

    def forward(self, x: Tensor, adj_t: Tensor) -> Tensor:
        for conv in self.convs:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class LINKXGCNHybrid(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        linkx_num_layers: int,
        linkx_num_edge_layers: int,
        linkx_num_node_layers: int,
        gcn_num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.linkx = LINKX(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=linkx_num_layers,
            num_edge_layers=linkx_num_edge_layers,
            num_node_layers=linkx_num_node_layers,
            dropout=dropout,
        )
        self.gcn = GCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=gcn_num_layers,
            dropout=dropout,
        )
        self.classifier = MLP(
            [2 * hidden_channels, hidden_channels, out_channels],
            dropout=dropout,
            act_first=True,
        )

    def forward(self, x: Tensor, linkx_adj_t: Tensor, gcn_adj_t: Tensor) -> Tensor:
        linkx_h = self.linkx(x, linkx_adj_t)
        gcn_h = self.gcn(x, gcn_adj_t)
        return self.classifier(torch.cat([linkx_h, gcn_h], dim=-1))


def accuracy(logits: Tensor, y: Tensor) -> float:
    return float((logits.argmax(dim=-1) == y).float().mean().item())


@torch.no_grad()
def evaluate(
    model: LINKXGCNHybrid,
    data: Data,
    split: Dict[str, Tensor],
    linkx_adj_t: Tensor,
    gcn_adj_t: Tensor,
):
    model.eval()
    logits = model(data.x, linkx_adj_t, gcn_adj_t)
    return {name: accuracy(logits[idx], data.y[idx]) for name, idx in split.items()}


def train_one(
    data: Data,
    split: Dict[str, Tensor],
    linkx_adj_t: Tensor,
    gcn_adj_t: Tensor,
    args: argparse.Namespace,
    tag: str,
) -> Dict[str, float]:
    set_seed(args.seed)
    model = LINKXGCNHybrid(
        num_nodes=data.num_nodes,
        in_channels=data.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=2,
        linkx_num_layers=args.linkx_num_layers,
        linkx_num_edge_layers=args.linkx_num_edge_layers,
        linkx_num_node_layers=args.linkx_num_node_layers,
        gcn_num_layers=args.gcn_num_layers,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    best_test = -1.0
    best_epoch = 0
    final = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, linkx_adj_t, gcn_adj_t)
        loss = F.cross_entropy(logits[split["train"]], data.y[split["train"]])
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, data, split, linkx_adj_t, gcn_adj_t)
            final = metrics
            if metrics["valid"] > best_val:
                best_val = metrics["valid"]
                best_test = metrics["test"]
                best_epoch = epoch
            print(
                f"{tag} Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                f"Train {metrics['train']:.4f} | Val {metrics['valid']:.4f} | "
                f"Test {metrics['test']:.4f} | BestTest {best_test:.4f}",
                flush=True,
            )

    return {
        "best_epoch": float(best_epoch),
        "best_val_acc": best_val,
        "best_test_acc": best_test,
        "final_train_acc": final.get("train", float("nan")),
        "final_val_acc": final.get("valid", float("nan")),
        "final_test_acc": final.get("test", float("nan")),
    }


def build_weights(
    mode: str,
    edge_index: Tensor,
    node_time: Tensor,
    node_year: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, str]:
    if mode == "none":
        return torch.ones(edge_index.size(1), dtype=torch.float32), "none_changed_edges: 0"
    if mode == "smp":
        weights, smp_edges, valid_time_edges = compute_smp_edge_weight(edge_index, node_time)
        return weights, f"smp_weighted_edges: {smp_edges}\nvalid_time_edges: {valid_time_edges}"
    if mode == "gsmp":
        weights = compute_gsmp_edge_weights(edge_index, node_year, num_nodes=num_nodes)
        changed = int((weights != 1).sum().item())
        return (
            weights,
            f"gsmp_changed_edges: {changed}\ngsmp_min_weight: {weights.min().item():.6f}\n"
            f"gsmp_max_weight: {weights.max().item():.6f}",
        )
    raise ValueError(f"Unknown weighting mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw Pokec LINKX+GCN hybrid with optional SMP/GSMP.")
    parser.add_argument("--root", default="data/pokec")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--train-until-year", type=int, default=2009)
    parser.add_argument("--val-year", type=int, default=2010)
    parser.add_argument("--test-from-year", type=int, default=2011)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--linkx-num-layers", type=int, default=2)
    parser.add_argument("--linkx-num-edge-layers", type=int, default=1)
    parser.add_argument("--linkx-num-node-layers", type=int, default=1)
    parser.add_argument("--gcn-num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--mode", choices=("gsmp", "smp"), default="gsmp",
                        help="Compare unweighted hybrid against this weighted hybrid.")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--tag", default="pokec_raw_linkx_gcn_hybrid")
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)
    profiles_path = existing_path(args.root, "soc-pokec-profiles.txt.gz")
    relationships_path = existing_path(args.root, "soc-pokec-relationships.txt.gz")
    if not args.no_download:
        download(PROFILES_URL, profiles_path)
        download(RELATIONSHIPS_URL, relationships_path)

    x, y, node_time, node_year = load_profiles(profiles_path)
    edge_index_directed = load_relationships(relationships_path, num_nodes=x.size(0))
    edge_index_undirected = torch.cat([edge_index_directed, edge_index_directed.flip(0)], dim=1)
    split = build_temporal_split(
        y,
        node_year,
        train_until_year=args.train_until_year,
        val_year=args.val_year,
        test_from_year=args.test_from_year,
    )

    linkx_base_weight, linkx_base_summary = build_weights(
        "none", edge_index_directed, node_time, node_year, x.size(0)
    )
    gcn_base_weight, gcn_base_summary = build_weights(
        "none", edge_index_undirected, node_time, node_year, x.size(0)
    )
    linkx_weight, linkx_summary = build_weights(
        args.mode, edge_index_directed, node_time, node_year, x.size(0)
    )
    gcn_weight, gcn_summary = build_weights(
        args.mode, edge_index_undirected, node_time, node_year, x.size(0)
    )

    linkx_adj_t = make_linkx_sparse_adj(edge_index_directed, x.size(0), edge_weight=linkx_base_weight)
    gcn_adj_t = make_linkx_sparse_adj(edge_index_undirected, x.size(0), edge_weight=gcn_base_weight)
    linkx_adj_t_weighted = make_linkx_sparse_adj(edge_index_directed, x.size(0), edge_weight=linkx_weight)
    gcn_adj_t_weighted = make_linkx_sparse_adj(edge_index_undirected, x.size(0), edge_weight=gcn_weight)

    x = normalize_features(x)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = Data(x=x, y=y, edge_index=edge_index_directed, num_nodes=x.size(0)).to(args.device)
    split = {key: value.to(args.device) for key, value in split.items()}
    linkx_adj_t = linkx_adj_t.to(args.device)
    gcn_adj_t = gcn_adj_t.to(args.device)
    linkx_adj_t_weighted = linkx_adj_t_weighted.to(args.device)
    gcn_adj_t_weighted = gcn_adj_t_weighted.to(args.device)

    print(data, flush=True)
    print(f"Split sizes: train={split['train'].numel()} valid={split['valid'].numel()} test={split['test'].numel()}", flush=True)
    print(f"Directed LINKX edges: {edge_index_directed.size(1)}", flush=True)
    print(f"Undirected GCN edges: {edge_index_undirected.size(1)}", flush=True)
    print(f"Baseline LINKX weights:\n{linkx_base_summary}", flush=True)
    print(f"Baseline GCN weights:\n{gcn_base_summary}", flush=True)
    print(f"Weighted LINKX mode={args.mode}:\n{linkx_summary}", flush=True)
    print(f"Weighted GCN mode={args.mode}:\n{gcn_summary}", flush=True)
    print("No-label-leakage check:", flush=True)
    print("  label target: gender column", flush=True)
    print("  loss labels: train split only", flush=True)
    print("  model selection: validation accuracy only", flush=True)
    print("  feature columns exclude gender, registration, last_login, and user_id", flush=True)
    print("  registration year/time is used only for temporal split and edge weights", flush=True)

    baseline_name = "LINKX+GCN"
    weighted_name = f"LINKX+GCN+{args.mode.upper()}"
    baseline = train_one(data, split, linkx_adj_t, gcn_adj_t, args, baseline_name)
    weighted = train_one(
        data,
        split,
        linkx_adj_t_weighted,
        gcn_adj_t_weighted,
        args,
        weighted_name,
    )

    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, f"{args.tag}.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Raw Pokec {baseline_name} vs {weighted_name}\n")
        f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"root: {args.root}\n")
        f.write(f"mode: {args.mode}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"num_nodes: {data.num_nodes}\n")
        f.write(f"directed_linkx_edges: {edge_index_directed.size(1)}\n")
        f.write(f"undirected_gcn_edges: {edge_index_undirected.size(1)}\n")
        f.write(f"num_features: {data.num_features}\n")
        f.write(f"hidden_channels: {args.hidden_channels}\n")
        f.write(f"linkx_num_layers: {args.linkx_num_layers}\n")
        f.write(f"gcn_num_layers: {args.gcn_num_layers}\n")
        f.write(f"train_until_year: {args.train_until_year}\n")
        f.write(f"val_year: {args.val_year}\n")
        f.write(f"test_from_year: {args.test_from_year}\n")
        f.write(f"train_nodes: {split['train'].numel()}\n")
        f.write(f"valid_nodes: {split['valid'].numel()}\n")
        f.write(f"test_nodes: {split['test'].numel()}\n")
        f.write(linkx_summary + "\n")
        f.write(gcn_summary + "\n")
        f.write("no_label_leakage: gender/registration/last_login/user_id excluded from features; only train labels used for loss\n")
        for prefix, metrics in (("linkx_gcn", baseline), (f"linkx_gcn_{args.mode}", weighted)):
            for key, value in metrics.items():
                f.write(f"{prefix}_{key}: {value:.6f}\n")
    print(f"Saved result txt to {result_path}", flush=True)


if __name__ == "__main__":
    main()
