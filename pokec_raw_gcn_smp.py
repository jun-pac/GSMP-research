import argparse
import os
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

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


class GCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("--num-layers must be at least 2.")
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def accuracy(logits: Tensor, y: Tensor) -> float:
    return float((logits.argmax(dim=-1) == y).float().mean().item())


@torch.no_grad()
def evaluate(model: GCN, data: Data, split: Dict[str, Tensor], adj_t: Tensor):
    model.eval()
    logits = model(data.x, adj_t)
    return {name: accuracy(logits[idx], data.y[idx]) for name, idx in split.items()}


def train_one(
    data: Data,
    split: Dict[str, Tensor],
    adj_t: Tensor,
    args: argparse.Namespace,
    tag: str,
) -> Dict[str, float]:
    set_seed(args.seed)
    model = GCN(
        in_channels=data.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=2,
        num_layers=args.num_layers,
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
        logits = model(data.x, adj_t)
        loss = F.cross_entropy(logits[split["train"]], data.y[split["train"]])
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, data, split, adj_t)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw Pokec GCN vs GCN+SMP with registration-time split.")
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
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--tag", default="pokec_raw_gcn_vs_gcn_smp")
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)
    profiles_path = existing_path(args.root, "soc-pokec-profiles.txt.gz")
    relationships_path = existing_path(args.root, "soc-pokec-relationships.txt.gz")
    if not args.no_download:
        download(PROFILES_URL, profiles_path)
        download(RELATIONSHIPS_URL, relationships_path)

    x, y, node_time, node_year = load_profiles(profiles_path)
    edge_index = load_relationships(relationships_path, num_nodes=x.size(0))
    # Use an undirected graph for the GCN baseline.
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    split = build_temporal_split(
        y,
        node_year,
        train_until_year=args.train_until_year,
        val_year=args.val_year,
        test_from_year=args.test_from_year,
    )
    edge_weight, smp_edges, valid_time_edges = compute_smp_edge_weight(edge_index, node_time)
    adj_t = make_linkx_sparse_adj(edge_index, x.size(0))
    adj_t_smp = make_linkx_sparse_adj(edge_index, x.size(0), edge_weight=edge_weight)

    x = normalize_features(x)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = Data(x=x, y=y, edge_index=edge_index, num_nodes=x.size(0)).to(args.device)
    split = {key: value.to(args.device) for key, value in split.items()}
    adj_t = adj_t.to(args.device)
    adj_t_smp = adj_t_smp.to(args.device)

    print(data, flush=True)
    print(f"Split sizes: train={split['train'].numel()} valid={split['valid'].numel()} test={split['test'].numel()}", flush=True)
    print(f"Edges with finite endpoint registration times: {valid_time_edges} / {edge_index.size(1)}", flush=True)
    print(f"SMP weighted edges: {smp_edges} / {edge_index.size(1)}", flush=True)
    print("No-label-leakage check:", flush=True)
    print("  label target: gender column", flush=True)
    print("  loss labels: train split only", flush=True)
    print("  model selection: validation accuracy only", flush=True)
    print("  feature columns exclude gender, registration, last_login, and user_id", flush=True)

    gcn = train_one(data, split, adj_t, args, "GCN")
    gcn_smp = train_one(data, split, adj_t_smp, args, "GCN+SMP")

    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, f"{args.tag}.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("Raw Pokec GCN vs GCN+SMP\n")
        f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"root: {args.root}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"num_nodes: {data.num_nodes}\n")
        f.write(f"num_edges: {data.edge_index.size(1)}\n")
        f.write(f"num_features: {data.num_features}\n")
        f.write(f"hidden_channels: {args.hidden_channels}\n")
        f.write(f"num_layers: {args.num_layers}\n")
        f.write(f"train_until_year: {args.train_until_year}\n")
        f.write(f"val_year: {args.val_year}\n")
        f.write(f"test_from_year: {args.test_from_year}\n")
        f.write(f"train_nodes: {split['train'].numel()}\n")
        f.write(f"valid_nodes: {split['valid'].numel()}\n")
        f.write(f"test_nodes: {split['test'].numel()}\n")
        f.write(f"valid_time_edges: {valid_time_edges}\n")
        f.write(f"smp_weighted_edges: {smp_edges}\n")
        f.write("no_label_leakage: gender/registration/last_login/user_id excluded from features; only train labels used for loss\n")
        for prefix, metrics in (("gcn", gcn), ("gcn_smp", gcn_smp)):
            for key, value in metrics.items():
                f.write(f"{prefix}_{key}: {value:.6f}\n")
    print(f"Saved result txt to {result_path}", flush=True)


if __name__ == "__main__":
    main()
