import argparse
import os
import time
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def compute_smp_edge_mask(
    edge_index: Tensor,
    node_time: Tensor,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> Tensor:
    """Return True for directed edges whose SMP contribution should be doubled."""
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    node_time = node_time.view(-1).to(edge_index.device)
    time = node_time.to(torch.float32)
    if t_min is None:
        t_min = float(time.min().item())
    if t_max is None:
        t_max = float(time.max().item())

    src, dst = edge_index
    delta = (time[src] - time[dst]).abs()
    radius = torch.minimum(time[dst] - t_min, t_max - time[dst])
    return delta > radius


def preprocess_graph_structure_with_smp(
    edge_index: Tensor,
    node_time: Tensor,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> Tensor:
    """
    Apply SMP without changing the GCN operation.

    SMP weight 2 is represented by duplicating the directed edge once. Edges
    that do not satisfy the SMP condition are left unchanged.
    """
    single_mask = compute_smp_edge_mask(edge_index, node_time, t_min=t_min, t_max=t_max)
    doubled_edges = edge_index[:, single_mask]
    return torch.cat([edge_index, doubled_edges], dim=1)


def compute_smp_edge_weight(
    edge_index: Tensor,
    node_time: Tensor,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Return SMP weights for each directed edge u -> v.

    The raw SMP rule assigns weight 2 to temporal "single" neighbors and
    weight 1 otherwise. PyG's GCNConv then applies its usual GCN
    normalization to this weighted adjacency.
    """
    single_mask = compute_smp_edge_mask(edge_index, node_time, t_min=t_min, t_max=t_max)
    edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)
    edge_weight[single_mask] = 2.0
    return edge_weight


def add_reverse_edges(edge_index: Tensor) -> Tensor:
    src, dst = edge_index
    return torch.cat([edge_index, torch.stack([dst, src], dim=0)], dim=1)


def make_masks_from_time(
    node_time: Tensor,
    train_until: int = 34,
    val_until: Optional[int] = 41,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Default temporal split for Elliptic: train early steps, test later steps."""
    node_time = node_time.view(-1)
    train_mask = node_time <= train_until
    if val_until is None or val_until <= train_until:
        val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
        test_mask = node_time > train_until
    else:
        val_mask = (node_time > train_until) & (node_time <= val_until)
        test_mask = node_time > val_until
    return train_mask, val_mask, test_mask


def load_elliptic_bitcoin(
    root: str,
    use_unknown_as_unlabeled: bool = True,
    make_undirected: bool = True,
    apply_smp: bool = False,
    smp_mode: str = "edge_weight",
    train_until: int = 34,
    val_until: Optional[int] = 41,
) -> Data:
    """
    Load the standard Elliptic Bitcoin transaction dataset as a PyG Data object.

    Expected files:
      - elliptic_txs_features.csv
      - elliptic_txs_classes.csv
      - elliptic_txs_edgelist.csv
    """
    features_path = os.path.join(root, "elliptic_txs_features.csv")
    classes_path = os.path.join(root, "elliptic_txs_classes.csv")
    edges_path = os.path.join(root, "elliptic_txs_edgelist.csv")

    features = pd.read_csv(features_path, header=None)
    classes = pd.read_csv(classes_path)
    edges = pd.read_csv(edges_path)

    tx_ids = features.iloc[:, 0].to_numpy()
    tx_id_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}

    node_time = torch.tensor(features.iloc[:, 1].to_numpy(), dtype=torch.long)
    x = torch.tensor(features.iloc[:, 2:].to_numpy(), dtype=torch.float32)

    class_map = dict(zip(classes["txId"], classes["class"]))
    y = torch.full((len(tx_ids),), -1, dtype=torch.long)
    for tx_id, idx in tx_id_to_idx.items():
        raw_label = class_map.get(tx_id, "unknown")
        if raw_label == "1":
            y[idx] = 1
        elif raw_label == "2":
            y[idx] = 0
        elif not use_unknown_as_unlabeled:
            y[idx] = 2

    edge_src = edges["txId1"].map(tx_id_to_idx)
    edge_dst = edges["txId2"].map(tx_id_to_idx)
    valid_edges = edge_src.notna() & edge_dst.notna()
    src = torch.tensor(edge_src[valid_edges].to_numpy(), dtype=torch.long)
    dst = torch.tensor(edge_dst[valid_edges].to_numpy(), dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    if make_undirected:
        edge_index = add_reverse_edges(edge_index)
    original_edge_count = edge_index.size(1)
    smp_edge_count = 0
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
    if apply_smp:
        smp_edge_mask = compute_smp_edge_mask(edge_index, node_time)
        smp_edge_count = int(smp_edge_mask.sum().item())
        if smp_mode == "duplicate":
            edge_index = preprocess_graph_structure_with_smp(edge_index, node_time)
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
        elif smp_mode == "edge_weight":
            edge_weight = compute_smp_edge_weight(edge_index, node_time)
        else:
            raise ValueError(f"Unknown smp_mode: {smp_mode}")

    train_mask, val_mask, test_mask = make_masks_from_time(
        node_time, train_until=train_until, val_until=val_until
    )
    labeled_mask = y >= 0
    train_mask &= labeled_mask
    val_mask &= labeled_mask
    test_mask &= labeled_mask

    return Data(
        x=x,
        y=y,
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_time=node_time,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        original_edge_count=original_edge_count,
        smp_duplicated_edges=smp_edge_count,
    )


class GCN(torch.nn.Module):
    """Plain GCN. SMP is supplied through preprocessed edge_index or edge_weight."""

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
            raise ValueError("num_layers must be at least 2.")
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index, edge_weight=edge_weight)


def accuracy(logits: Tensor, y: Tensor) -> float:
    if y.numel() == 0:
        return float("nan")
    return float((logits.argmax(dim=-1) == y).sum().item() / y.numel())


def binary_metrics(logits: Tensor, y: Tensor, positive_label: int = 1) -> Dict[str, float]:
    if y.numel() == 0:
        return {
            "acc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "balanced_acc": float("nan"),
            "tp": 0.0,
            "fp": 0.0,
            "tn": 0.0,
            "fn": 0.0,
        }

    pred = logits.argmax(dim=-1)
    positive = y == positive_label
    predicted_positive = pred == positive_label
    negative = ~positive
    predicted_negative = ~predicted_positive

    tp = int((predicted_positive & positive).sum().item())
    fp = int((predicted_positive & negative).sum().item())
    tn = int((predicted_negative & negative).sum().item())
    fn = int((predicted_negative & positive).sum().item())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
    return {
        "acc": (tp + tn) / y.numel(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_acc": 0.5 * (recall + specificity),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def train(model: torch.nn.Module, data: Data, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, getattr(data, "edge_weight", None))
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: Data) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    model.eval()
    out = model(data.x, data.edge_index, getattr(data, "edge_weight", None))
    return (
        binary_metrics(out[data.train_mask], data.y[data.train_mask]),
        binary_metrics(out[data.val_mask], data.y[data.val_mask]),
        binary_metrics(out[data.test_mask], data.y[data.test_mask]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Elliptic Bitcoin GCN with SMP graph preprocessing")
    parser.add_argument("--root", type=str, required=True, help="Directory containing Elliptic CSV files.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-smp", action="store_true", help="Apply SMP to graph message passing.")
    parser.add_argument("--smp-mode", choices=("edge_weight", "duplicate"),
                        default="edge_weight",
                        help="Use weighted GCN adjacency or duplicate SMP edges.")
    parser.add_argument("--directed", action="store_true", help="Keep original directed edges only.")
    parser.add_argument("--train-until", type=int, default=34)
    parser.add_argument("--val-until", type=int, default=41)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--select-metric", choices=("acc", "f1", "balanced_acc"),
                        default="f1",
                        help="Validation metric used to choose the best epoch.")
    parser.add_argument("--result-dir", type=str, default="results")
    parser.add_argument("--tag", type=str, default=None,
                        help="Suffix tag for the result txt file.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = load_elliptic_bitcoin(
        args.root,
        make_undirected=not args.directed,
        apply_smp=args.use_smp,
        smp_mode=args.smp_mode,
        train_until=args.train_until,
        val_until=args.val_until,
    ).to(device)

    model = GCN(
        in_channels=data.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=2,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(data)
    print(f"SMP graph preprocessing: {args.use_smp}")
    print(f"SMP mode: {args.smp_mode}")
    print(f"Original edges: {int(data.original_edge_count)}")
    print(f"SMP duplicated edges: {int(data.smp_duplicated_edges)}")
    print(f"SMP weighted edges: {int((data.edge_weight != 1).sum().item())}")
    if args.use_smp and int(data.smp_duplicated_edges) == 0:
        print("WARNING: SMP selected zero cross-time edges; this graph is unchanged by SMP.")
    print(f"Train/Val/Test labels: {int(data.train_mask.sum())}/"
          f"{int(data.val_mask.sum())}/{int(data.test_mask.sum())}")
    for split_name in ("train", "val", "test"):
        split_y = data.y[getattr(data, f"{split_name}_mask")]
        licit = int((split_y == 0).sum().item())
        illicit = int((split_y == 1).sum().item())
        majority_acc = max(licit, illicit) / split_y.numel()
        print(
            f"{split_name.capitalize()} labels: licit={licit} illicit={illicit} "
            f"illicit_rate={illicit / split_y.numel():.4f} "
            f"majority_acc={majority_acc:.4f}"
        )

    history = []
    best_val = -1.0
    best_test_metric = {}
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer)
        train_metrics, val_metrics, test_metrics = evaluate(model, data)
        history.append((epoch, loss, train_metrics, val_metrics, test_metrics))
        if val_metrics[args.select_metric] > best_val:
            best_val = val_metrics[args.select_metric]
            best_test_metric = test_metrics
            best_epoch = epoch
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                f"TrainAcc {train_metrics['acc']:.4f} | ValAcc {val_metrics['acc']:.4f} | "
                f"TestAcc {test_metrics['acc']:.4f} | "
                f"ValF1 {val_metrics['f1']:.4f} | TestF1 {test_metrics['f1']:.4f} | "
                f"BestTestF1 {best_test_metric['f1']:.4f}"
            )

    tag = args.tag
    if tag is None:
        graph_tag = "smp" if args.use_smp else "vanilla"
        direction_tag = "directed" if args.directed else "undirected"
        tag = (
            f"{graph_tag}_{direction_tag}_"
            f"L{args.num_layers}_H{args.hidden_channels}_"
            f"lr{args.lr:g}_wd{args.weight_decay:g}"
        )
    safe_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in tag)
    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, f"elliptic_bitcoin_{safe_tag}.txt")

    final_epoch, final_loss, final_train, final_val, final_test = history[-1]
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("Elliptic Bitcoin SMP GCN result\n")
        f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"tag: {safe_tag}\n")
        f.write(f"root: {args.root}\n")
        f.write(f"use_smp: {args.use_smp}\n")
        f.write(f"smp_mode: {args.smp_mode}\n")
        f.write(f"directed: {args.directed}\n")
        f.write(f"num_nodes: {data.num_nodes}\n")
        f.write(f"original_edges: {int(data.original_edge_count)}\n")
        f.write(f"smp_duplicated_edges: {int(data.smp_duplicated_edges)}\n")
        f.write(f"smp_weighted_edges: {int((data.edge_weight != 1).sum().item())}\n")
        f.write(f"num_edges: {data.edge_index.size(1)}\n")
        f.write(f"num_features: {data.num_features}\n")
        f.write(f"train_labels: {int(data.train_mask.sum())}\n")
        f.write(f"val_labels: {int(data.val_mask.sum())}\n")
        f.write(f"test_labels: {int(data.test_mask.sum())}\n")
        f.write(f"hidden_channels: {args.hidden_channels}\n")
        f.write(f"num_layers: {args.num_layers}\n")
        f.write(f"dropout: {args.dropout}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"weight_decay: {args.weight_decay}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"select_metric: {args.select_metric}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_{args.select_metric}: {best_val:.6f}\n")
        for metric_name, metric_value in best_test_metric.items():
            f.write(f"best_test_{metric_name}: {metric_value:.6f}\n")
        f.write(f"final_epoch: {final_epoch}\n")
        f.write(f"final_loss: {final_loss:.6f}\n")
        for split_name, split_metrics in (
            ("train", final_train),
            ("val", final_val),
            ("test", final_test),
        ):
            for metric_name, metric_value in split_metrics.items():
                f.write(f"final_{split_name}_{metric_name}: {metric_value:.6f}\n")

    print(f"Saved result txt to {result_path}")


if __name__ == "__main__":
    main()
