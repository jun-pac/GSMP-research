import argparse
import gzip
import os
import random
import time
from collections import Counter
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.models import LINKX


PROFILES_URL = "https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz"
RELATIONSHIPS_URL = "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz"

USER_ID_COL = 0
PUBLIC_COL = 1
COMPLETION_COL = 2
GENDER_COL = 3
LAST_LOGIN_COL = 5
REGISTRATION_COL = 6
AGE_COL = 7


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download(url: str, path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import urllib.request
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def existing_path(root: str, gz_name: str) -> str:
    gz_path = os.path.join(root, gz_name)
    plain_path = gz_path[:-3] if gz_path.endswith(".gz") else gz_path
    if os.path.exists(gz_path):
        return gz_path
    if os.path.exists(plain_path):
        return plain_path
    return gz_path


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_datetime(value: str) -> Optional[datetime]:
    value = value.strip()
    if not value or value == "null":
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def parse_float(value: str, default: float = 0.0) -> float:
    if value == "null" or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_int_label(value: str) -> int:
    if value == "null" or value == "":
        return -1
    try:
        return int(value)
    except ValueError:
        return -1


def timestamp_days(dt: datetime) -> float:
    return dt.timestamp() / 86400.0


def load_profiles(path: str, num_nodes: Optional[int] = None):
    rows = []
    max_user_id = 0
    with open_text(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= AGE_COL:
                continue
            try:
                user_id = int(parts[USER_ID_COL])
            except ValueError:
                continue
            max_user_id = max(max_user_id, user_id)
            rows.append((user_id, parts))

    n = max(num_nodes or 0, max_user_id)
    y = torch.full((n,), -1, dtype=torch.long)
    node_time = torch.full((n,), float("nan"), dtype=torch.float32)
    node_year = torch.full((n,), -1, dtype=torch.long)
    x = torch.zeros((n, 5), dtype=torch.float32)

    valid_registration = 0
    valid_gender = 0
    for user_id, parts in rows:
        idx = user_id - 1
        gender = parse_int_label(parts[GENDER_COL])
        if gender in (0, 1):
            y[idx] = gender
            valid_gender += 1

        registered_at = parse_datetime(parts[REGISTRATION_COL])
        if registered_at is not None:
            node_time[idx] = timestamp_days(registered_at)
            node_year[idx] = registered_at.year
            valid_registration += 1

        public = parse_float(parts[PUBLIC_COL])
        completion = parse_float(parts[COMPLETION_COL]) / 100.0
        age_raw = parse_float(parts[AGE_COL])
        age_known = 1.0 if age_raw > 0 else 0.0
        age = age_raw / 100.0 if age_raw > 0 else 0.0

        # No label leakage: excluded columns are user_id, gender label,
        # last_login, and registration time. Registration is used only for
        # temporal split construction and SMP edge weights.
        x[idx] = torch.tensor([public, completion, age, age_known, 1.0])

    print(f"profile rows: {len(rows)}")
    print(f"valid gender labels: {valid_gender}")
    print(f"valid registration times: {valid_registration}")
    print("features: public, completion_percentage, age, age_known, bias")
    print("excluded from features: user_id, gender(label), last_login, registration")
    return x, y, node_time, node_year


def load_relationships(path: str, num_nodes: int) -> Tensor:
    def flat_edges():
        with open_text(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                u = int(parts[0]) - 1
                v = int(parts[1]) - 1
                if 0 <= u < num_nodes and 0 <= v < num_nodes:
                    yield u
                    yield v

    flat = np.fromiter(flat_edges(), dtype=np.int64)
    if flat.size % 2 != 0:
        raise ValueError("Relationship file produced an odd number of endpoints.")
    edges = flat.reshape(-1, 2).T
    return torch.from_numpy(edges).long()


def build_temporal_split(
    y: Tensor,
    node_year: Tensor,
    train_until_year: int,
    val_year: int,
    test_from_year: int,
) -> Dict[str, Tensor]:
    train = []
    val = []
    test = []
    years = []
    for idx in torch.where((y >= 0) & (node_year >= 0))[0].tolist():
        year = int(node_year[idx].item())
        years.append(year)
        if year <= train_until_year:
            train.append(idx)
        elif year == val_year:
            val.append(idx)
        elif year >= test_from_year:
            test.append(idx)
    print(f"labeled temporal year counts: {dict(sorted(Counter(years).items()))}")
    return {
        "train": torch.tensor(train, dtype=torch.long),
        "valid": torch.tensor(val, dtype=torch.long),
        "test": torch.tensor(test, dtype=torch.long),
    }


def compute_smp_edge_weight(edge_index: Tensor, node_time: Tensor) -> Tuple[Tensor, int, int]:
    finite_time = node_time[torch.isfinite(node_time)]
    if finite_time.numel() == 0:
        raise ValueError("No finite node_time values available for SMP.")
    t_min = float(finite_time.min().item())
    t_max = float(finite_time.max().item())
    src, dst = edge_index
    valid = torch.isfinite(node_time[src]) & torch.isfinite(node_time[dst])
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
    delta = (node_time[src[valid]] - node_time[dst[valid]]).abs()
    radius = torch.minimum(node_time[dst[valid]] - t_min, t_max - node_time[dst[valid]])
    single_valid = delta > radius
    valid_positions = torch.where(valid)[0]
    edge_weight[valid_positions[single_valid]] = 2.0
    return edge_weight, int(single_valid.sum().item()), int(valid.sum().item())


def make_linkx_sparse_adj(edge_index: Tensor, num_nodes: int, edge_weight: Optional[Tensor] = None) -> Tensor:
    """
    Build the transposed sparse adjacency expected by PyG MessagePassing.

    Passing raw edge_index to LINKX materializes [num_edges, hidden_channels]
    messages. For raw Pokec that is too large, so we pass a sparse adjacency
    and trigger LINKX's sparse-matrix multiply path instead.
    """
    src, dst = edge_index
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
    indices = torch.stack([dst, src], dim=0)
    adj_t = torch.sparse_coo_tensor(indices, edge_weight, (num_nodes, num_nodes))
    return adj_t.coalesce().to_sparse_csr()


def normalize_features(x: Tensor) -> Tensor:
    out = x.clone()
    mean = out.mean(dim=0, keepdim=True)
    std = out.std(dim=0, keepdim=True)
    std[std == 0] = 1
    return (out - mean) / std


def accuracy(logits: Tensor, y: Tensor) -> float:
    return float((logits.argmax(dim=-1) == y).float().mean().item())


@torch.no_grad()
def evaluate(model: LINKX, data: Data, split: Dict[str, Tensor], adj_t: Tensor):
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
    model = LINKX(
        num_nodes=data.num_nodes,
        in_channels=data.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=2,
        num_layers=args.num_layers,
        num_edge_layers=args.num_edge_layers,
        num_node_layers=args.num_node_layers,
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
                f"Test {metrics['test']:.4f} | BestTest {best_test:.4f}"
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
    parser = argparse.ArgumentParser(description="Raw Pokec LINKX vs LINKX+SMP with registration-time split.")
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
    parser.add_argument("--num-edge-layers", type=int, default=1)
    parser.add_argument("--num-node-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--tag", default="pokec_raw_linkx_vs_linkx_smp")
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)
    profiles_path = existing_path(args.root, "soc-pokec-profiles.txt.gz")
    relationships_path = existing_path(args.root, "soc-pokec-relationships.txt.gz")
    if not args.no_download:
        download(PROFILES_URL, profiles_path)
        download(RELATIONSHIPS_URL, relationships_path)

    x, y, node_time, node_year = load_profiles(profiles_path)
    edge_index = load_relationships(relationships_path, num_nodes=x.size(0))
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

    print(data)
    print(f"Split sizes: train={split['train'].numel()} valid={split['valid'].numel()} test={split['test'].numel()}")
    print(f"Edges with finite endpoint registration times: {valid_time_edges} / {edge_index.size(1)}")
    print(f"SMP weighted edges: {smp_edges} / {edge_index.size(1)}")
    print("No-label-leakage check:")
    print("  label target: gender column")
    print("  loss labels: train split only")
    print("  model selection: validation accuracy only")
    print("  feature columns exclude gender, registration, last_login, and user_id")

    linkx = train_one(data, split, adj_t, args, "LINKX")
    linkx_smp = train_one(data, split, adj_t_smp, args, "LINKX+SMP")

    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, f"{args.tag}.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("Raw Pokec LINKX vs LINKX+SMP\n")
        f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"root: {args.root}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"num_nodes: {data.num_nodes}\n")
        f.write(f"num_edges: {data.edge_index.size(1)}\n")
        f.write(f"num_features: {data.num_features}\n")
        f.write(f"hidden_channels: {args.hidden_channels}\n")
        f.write(f"train_until_year: {args.train_until_year}\n")
        f.write(f"val_year: {args.val_year}\n")
        f.write(f"test_from_year: {args.test_from_year}\n")
        f.write(f"train_nodes: {split['train'].numel()}\n")
        f.write(f"valid_nodes: {split['valid'].numel()}\n")
        f.write(f"test_nodes: {split['test'].numel()}\n")
        f.write(f"valid_time_edges: {valid_time_edges}\n")
        f.write(f"smp_weighted_edges: {smp_edges}\n")
        f.write("no_label_leakage: gender/registration/last_login/user_id excluded from features; only train labels used for loss\n")
        for prefix, metrics in (("linkx", linkx), ("linkx_smp", linkx_smp)):
            for key, value in metrics.items():
                f.write(f"{prefix}_{key}: {value:.6f}\n")
    print(f"Saved result txt to {result_path}")


if __name__ == "__main__":
    main()
