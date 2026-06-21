import argparse
import os
import random
import time
from typing import Dict, Optional, Tuple

import gdown
import numpy as np
import scipy.io
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.models import LINKX


POKEC_FILE_ID = "1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y"
POKEC_SPLITS_FILE_ID = "1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_google_drive_file(file_id: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        return
    print(f"Downloading {output_path}")
    gdown.download(id=file_id, output=output_path, quiet=False)
    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"Could not download {output_path}. If Google Drive blocks gdown, "
            "download it manually from the original LINKX repository link."
        )


def load_mat_array(value):
    if sp.issparse(value):
        return value.todense()
    return value


def load_pokec(root: str, download: bool = True) -> Tuple[Data, Dict[str, object]]:
    mat_path = os.path.join(root, "pokec.mat")
    if download:
        download_google_drive_file(POKEC_FILE_ID, mat_path)
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Missing {mat_path}")

    mat = scipy.io.loadmat(mat_path)
    if "edge_index" not in mat or "node_feat" not in mat or "label" not in mat:
        keys = ", ".join(sorted(k for k in mat if not k.startswith("__")))
        raise ValueError(f"{mat_path} does not look like LINKX pokec.mat. Keys: {keys}")

    edge_index_np = np.asarray(mat["edge_index"])
    if edge_index_np.shape[0] != 2 and edge_index_np.shape[1] == 2:
        edge_index_np = edge_index_np.T
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    node_feat_np = load_mat_array(mat["node_feat"])
    x = torch.tensor(np.asarray(node_feat_np), dtype=torch.float32)

    y = torch.tensor(np.asarray(mat["label"]).reshape(-1), dtype=torch.long)
    num_nodes = int(np.asarray(mat.get("num_nodes", [[x.size(0)]])).reshape(-1)[0])
    if x.size(0) != num_nodes:
        raise ValueError(f"node_feat has {x.size(0)} rows but num_nodes={num_nodes}")

    data = Data(x=x, y=y, edge_index=edge_index, num_nodes=num_nodes)
    return data, mat


def random_split(
    y: Tensor,
    train_prop: float = 0.5,
    val_prop: float = 0.25,
    seed: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    labeled = torch.where(y >= 0)[0]
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = labeled[torch.randperm(labeled.numel(), generator=generator)]
    train_end = int(labeled.numel() * train_prop)
    val_end = train_end + int(labeled.numel() * val_prop)
    return perm[:train_end], perm[train_end:val_end], perm[val_end:]


def load_fixed_splits(root: str, download: bool = True, split_id: int = 0):
    split_path = os.path.join(root, "pokec-splits.npy")
    if download:
        download_google_drive_file(POKEC_SPLITS_FILE_ID, split_path)
    if not os.path.exists(split_path):
        return None
    splits = np.load(split_path, allow_pickle=True)
    split = splits[split_id]
    return {key: torch.as_tensor(split[key], dtype=torch.long) for key in split}


def get_node_time(
    data: Data,
    mat: Dict[str, object],
    time_key: Optional[str],
    time_feature_col: Optional[int],
) -> Tensor:
    candidate_keys = []
    if time_key:
        candidate_keys.append(time_key)
    candidate_keys.extend(["node_time", "time", "year", "years", "age"])

    for key in candidate_keys:
        if key in mat:
            values = np.asarray(load_mat_array(mat[key])).reshape(-1)
            if values.shape[0] == data.num_nodes:
                print(f"Using node time from pokec.mat key: {key}")
                return torch.tensor(values, dtype=torch.float32)

    if time_feature_col is None:
        raise ValueError(
            "SMP needs a node timestamp/time-like variable, but pokec.mat did not "
            "contain one of node_time/time/year/years/age. Pass "
            "--time-feature-col COL if one node feature column should be treated "
            "as the SMP time variable."
        )
    if time_feature_col < 0 or time_feature_col >= data.x.size(1):
        raise ValueError(f"--time-feature-col must be in [0, {data.x.size(1) - 1}]")
    print(f"Using node feature column {time_feature_col} as SMP time.")
    return data.x[:, time_feature_col].to(torch.float32)


def compute_smp_edge_weight(
    edge_index: Tensor,
    node_time: Tensor,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> Tuple[Tensor, int]:
    node_time = node_time.view(-1).to(edge_index.device).to(torch.float32)
    if t_min is None:
        t_min = float(node_time.min().item())
    if t_max is None:
        t_max = float(node_time.max().item())
    src, dst = edge_index
    delta = (node_time[src] - node_time[dst]).abs()
    radius = torch.minimum(node_time[dst] - t_min, t_max - node_time[dst])
    single = delta > radius
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
    edge_weight[single.cpu()] = 2.0
    return edge_weight, int(single.sum().item())


def accuracy(logits: Tensor, y: Tensor) -> float:
    return float((logits.argmax(dim=-1) == y).float().mean().item())


@torch.no_grad()
def evaluate(model: LINKX, data: Data, split: Dict[str, Tensor], edge_weight: Optional[Tensor]):
    model.eval()
    logits = model(data.x, data.edge_index, edge_weight)
    return {
        name: accuracy(logits[idx], data.y[idx])
        for name, idx in split.items()
    }, logits


def train_one(
    data: Data,
    split: Dict[str, Tensor],
    edge_weight: Optional[Tensor],
    args: argparse.Namespace,
    tag: str,
) -> Dict[str, float]:
    set_seed(args.seed)
    model = LINKX(
        num_nodes=data.num_nodes,
        in_channels=data.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=int(data.y.max().item()) + 1,
        num_layers=args.num_layers,
        num_edge_layers=args.num_edge_layers,
        num_node_layers=args.num_node_layers,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val = -1.0
    best_test = -1.0
    best_epoch = 0
    final_metrics = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, edge_weight)
        loss = F.cross_entropy(logits[split["train"]], data.y[split["train"]])
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            metrics, _ = evaluate(model, data, split, edge_weight)
            final_metrics = metrics
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
        "final_train_acc": final_metrics.get("train", float("nan")),
        "final_val_acc": final_metrics.get("valid", float("nan")),
        "final_test_acc": final_metrics.get("test", float("nan")),
    }


def inspect_data(data: Data, mat: Dict[str, object]) -> None:
    keys = sorted(k for k in mat if not k.startswith("__"))
    print(f"mat keys: {keys}")
    print(data)
    print(f"label counts: {torch.bincount(data.y[data.y >= 0]).tolist()}")
    print("feature column summary:")
    for col in range(data.x.size(1)):
        values = data.x[:, col]
        unique_count = int(torch.unique(values).numel())
        if unique_count <= 25:
            print(
                f"  col={col:02d} min={values.min().item():.4g} "
                f"max={values.max().item():.4g} unique={unique_count}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LINKX vs LINKX+SMP on Pokec.")
    parser.add_argument("--root", default="data/pokec")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--time-key", default=None)
    parser.add_argument("--time-feature-col", type=int, default=None)
    parser.add_argument("--use-fixed-split", action="store_true")
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-edge-layers", type=int, default=1)
    parser.add_argument("--num-node-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--tag", default="pokec_linkx_compare")
    args = parser.parse_args()

    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data, mat = load_pokec(args.root, download=not args.no_download)
    if args.inspect:
        inspect_data(data, mat)

    if args.use_fixed_split:
        split = load_fixed_splits(args.root, download=not args.no_download, split_id=args.split_id)
        if split is None:
            raise FileNotFoundError("Fixed split file missing. Retry without --no-download.")
    else:
        train, valid, test = random_split(data.y, seed=args.seed)
        split = {"train": train, "valid": valid, "test": test}

    node_time = get_node_time(data, mat, args.time_key, args.time_feature_col)
    edge_weight, smp_weighted_edges = compute_smp_edge_weight(data.edge_index, node_time)

    data = data.to(args.device)
    split = {key: value.to(args.device) for key, value in split.items()}
    edge_weight = edge_weight.to(args.device)

    print(data)
    print(f"Split sizes: train={split['train'].numel()} valid={split['valid'].numel()} test={split['test'].numel()}")
    print(f"SMP weighted edges: {smp_weighted_edges} / {data.edge_index.size(1)}")
    if smp_weighted_edges == 0:
        print("WARNING: SMP selected zero edges. LINKX+SMP will match LINKX except for randomness.")

    linkx = train_one(data, split, None, args, "LINKX")
    linkx_smp = train_one(data, split, edge_weight, args, "LINKX+SMP")

    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, f"{args.tag}.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("Pokec LINKX vs LINKX+SMP\n")
        f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"root: {args.root}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"num_nodes: {data.num_nodes}\n")
        f.write(f"num_edges: {data.edge_index.size(1)}\n")
        f.write(f"num_features: {data.num_features}\n")
        f.write(f"time_key: {args.time_key}\n")
        f.write(f"time_feature_col: {args.time_feature_col}\n")
        f.write(f"smp_weighted_edges: {smp_weighted_edges}\n")
        for prefix, metrics in (("linkx", linkx), ("linkx_smp", linkx_smp)):
            for key, value in metrics.items():
                f.write(f"{prefix}_{key}: {value:.6f}\n")
    print(f"Saved result txt to {result_path}")


if __name__ == "__main__":
    main()
