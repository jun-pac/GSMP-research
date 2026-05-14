import argparse
import os
from typing import Optional, Tuple

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
    if apply_smp:
        edge_index = preprocess_graph_structure_with_smp(edge_index, node_time)

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
        node_time=node_time,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


class GCN(torch.nn.Module):
    """Plain GCN. SMP is supplied only through the preprocessed edge_index."""

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

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def accuracy(logits: Tensor, y: Tensor) -> float:
    if y.numel() == 0:
        return float("nan")
    return float((logits.argmax(dim=-1) == y).sum().item() / y.numel())


def train(model: torch.nn.Module, data: Data, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: Data) -> Tuple[float, float, float]:
    model.eval()
    out = model(data.x, data.edge_index)
    return (
        accuracy(out[data.train_mask], data.y[data.train_mask]),
        accuracy(out[data.val_mask], data.y[data.val_mask]),
        accuracy(out[data.test_mask], data.y[data.test_mask]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Elliptic Bitcoin GCN with SMP graph preprocessing")
    parser.add_argument("--root", type=str, required=True, help="Directory containing Elliptic CSV files.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-smp", action="store_true", help="Duplicate SMP single-neighbor edges.")
    parser.add_argument("--directed", action="store_true", help="Keep original directed edges only.")
    parser.add_argument("--train-until", type=int, default=34)
    parser.add_argument("--val-until", type=int, default=41)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = load_elliptic_bitcoin(
        args.root,
        make_undirected=not args.directed,
        apply_smp=args.use_smp,
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
    print(f"Train/Val/Test labels: {int(data.train_mask.sum())}/"
          f"{int(data.val_mask.sum())}/{int(data.test_mask.sum())}")

    best_val = -1.0
    best_test = -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = evaluate(model, data)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                f"Train {train_acc:.4f} | Val {val_acc:.4f} | "
                f"Test {test_acc:.4f} | BestTest {best_test:.4f}"
            )


if __name__ == "__main__":
    main()
