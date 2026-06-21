from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


def parse_int_list(text: str) -> List[int]:
    if text.strip() == "":
        return []
    return [int(part) for part in text.split(",") if part.strip() != ""]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def balanced_timestamps(num_nodes: int, num_times: int) -> torch.Tensor:
    base = torch.arange(num_times, dtype=torch.long).repeat_interleave(math.ceil(num_nodes / num_times))
    return base[:num_nodes][torch.randperm(num_nodes)]


def make_class_centers(num_classes: int, feat_dim: int, feature_scale: float) -> torch.Tensor:
    centers = torch.randn(num_classes, feat_dim)
    centers = torch.nn.functional.normalize(centers, p=2, dim=1)
    return centers * float(feature_scale)


def make_masks(
    node_time: torch.Tensor,
    train_times: Iterable[int],
    val_times: Iterable[int],
    test_times: Iterable[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_set = set(int(t) for t in train_times)
    val_set = set(int(t) for t in val_times)
    test_set = set(int(t) for t in test_times)
    train_mask = torch.tensor([int(t) in train_set for t in node_time.tolist()], dtype=torch.bool)
    val_mask = torch.tensor([int(t) in val_set for t in node_time.tolist()], dtype=torch.bool)
    test_mask = torch.tensor([int(t) in test_set for t in node_time.tolist()], dtype=torch.bool)
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0 or int(test_mask.sum()) == 0:
        raise ValueError("train/val/test masks must all contain at least one node.")
    return train_mask, val_mask, test_mask


def make_default_splits(num_times: int) -> Tuple[List[int], List[int], List[int]]:
    train_end = max(1, int(round(num_times * 0.55)))
    val_end = max(train_end + 1, int(round(num_times * 0.75)))
    train_times = list(range(0, train_end))
    val_times = list(range(train_end, min(val_end, num_times)))
    test_times = list(range(min(val_end, num_times - 1), num_times))
    return train_times, val_times, test_times


def make_base_and_gamma(args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    base = torch.full((args.num_classes, args.num_classes), float(args.cross_p0), dtype=torch.float32)
    gamma = torch.full((args.num_classes, args.num_classes), float(args.cross_gamma), dtype=torch.float32)
    diag = torch.arange(args.num_classes)
    base[diag, diag] = float(args.same_p0)
    gamma[diag, diag] = float(args.same_gamma)
    return base.clamp(0.0, 1.0), gamma.clamp(0.0, 1.0)


def generate_edges(
    labels: torch.Tensor,
    node_time: torch.Tensor,
    base_prob: torch.Tensor,
    gamma: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    num_nodes = int(labels.numel())
    src_labels = labels.view(1, num_nodes)
    src_times = node_time.view(1, num_nodes)
    src_chunks: List[torch.Tensor] = []
    dst_chunks: List[torch.Tensor] = []

    for start in range(0, num_nodes, int(chunk_size)):
        end = min(start + int(chunk_size), num_nodes)
        dst_labels = labels[start:end].view(end - start, 1)
        dst_times = node_time[start:end].view(end - start, 1)
        delta_t = (dst_times - src_times).abs()
        p0 = base_prob[dst_labels, src_labels]
        g = gamma[dst_labels, src_labels]
        prob = p0 * torch.pow(g, delta_t.float())
        local_dst = torch.arange(start, end, dtype=torch.long).view(end - start, 1)
        prob[local_dst == torch.arange(num_nodes, dtype=torch.long).view(1, num_nodes)] = 0.0
        sampled = torch.rand_like(prob) < prob
        dst_local, src = sampled.nonzero(as_tuple=True)
        if src.numel() > 0:
            src_chunks.append(src.long())
            dst_chunks.append((dst_local + start).long())

    if not src_chunks:
        raise RuntimeError("Generated graph has no edges; increase probabilities or node count.")
    return torch.stack([torch.cat(src_chunks), torch.cat(dst_chunks)], dim=0).contiguous()


def edge_summary(edge_index: torch.Tensor, labels: torch.Tensor, node_time: torch.Tensor) -> Dict[str, float]:
    src, dst = edge_index
    deg = torch.bincount(dst, minlength=int(labels.numel())).float()
    same_label = (labels[src] == labels[dst]).float().mean().item()
    same_time = (node_time[src] == node_time[dst]).float().mean().item()
    delta = (node_time[src] - node_time[dst]).abs().float()
    return {
        "num_edges": int(edge_index.size(1)),
        "avg_in_degree": float(deg.mean().item()),
        "min_in_degree": float(deg.min().item()),
        "max_in_degree": float(deg.max().item()),
        "same_label_edge_fraction": float(same_label),
        "same_time_edge_fraction": float(same_time),
        "mean_abs_time_delta": float(delta.mean().item()),
    }


def label_time_table(labels: torch.Tensor, node_time: torch.Tensor, num_classes: int, num_times: int) -> List[List[int]]:
    table = torch.zeros(num_times, num_classes, dtype=torch.long)
    for t, y in zip(node_time.tolist(), labels.tolist()):
        table[int(t), int(y)] += 1
    return table.tolist()


def build_dataset(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    labels = torch.randint(low=0, high=args.num_classes, size=(args.num_nodes,), dtype=torch.long)
    node_time = balanced_timestamps(args.num_nodes, args.num_times)
    centers = make_class_centers(args.num_classes, args.feat_dim, args.feature_scale)
    class_noise = torch.linspace(args.feature_noise_min, args.feature_noise_max, args.num_classes)
    noise = torch.randn(args.num_nodes, args.feat_dim) * class_noise[labels].view(-1, 1)
    x = centers[labels] + noise

    base_prob, gamma = make_base_and_gamma(args)
    edge_index = generate_edges(labels, node_time, base_prob, gamma, args.chunk_size)

    train_times, val_times, test_times = make_default_splits(args.num_times)
    if args.train_times:
        train_times = parse_int_list(args.train_times)
    if args.val_times:
        val_times = parse_int_list(args.val_times)
    if args.test_times:
        test_times = parse_int_list(args.test_times)
    train_mask, val_mask, test_mask = make_masks(node_time, train_times, val_times, test_times)

    stats = edge_summary(edge_index, labels, node_time)
    stats.update(
        {
            "num_nodes": int(args.num_nodes),
            "num_classes": int(args.num_classes),
            "num_times": int(args.num_times),
            "feat_dim": int(args.feat_dim),
            "train_nodes": int(train_mask.sum().item()),
            "val_nodes": int(val_mask.sum().item()),
            "test_nodes": int(test_mask.sum().item()),
            "label_time_table": label_time_table(labels, node_time, args.num_classes, args.num_times),
        }
    )
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["train_times_resolved"] = train_times
    config["val_times_resolved"] = val_times
    config["test_times_resolved"] = test_times
    config["base_prob"] = base_prob.tolist()
    config["gamma"] = gamma.tolist()
    return {
        "x": x.float().contiguous(),
        "y": labels.contiguous(),
        "node_time": node_time.contiguous(),
        "edge_index": edge_index,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "class_centers": centers.float().contiguous(),
        "config": config,
        "stats": stats,
    }


def save_dataset(dataset: Dict[str, object], out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pt"
    torch.save(dataset, path)
    with (out_dir / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump({"config": dataset["config"], "stats": dataset["stats"]}, f, indent=2, sort_keys=True)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Temporal Stochastic Block Model dataset. Labels and timestamps are sampled "
            "independently; x_i = mu(y_i) + k_y Z_i; edge probabilities follow "
            "P[t,t~,y,y~] = P0[y,y~] * gamma[y,y~]^|t-t~|."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./data_tsbm"))
    parser.add_argument("--dataset", type=str, default="tsbm_extreme")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-nodes", type=int, default=3000)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--num-times", type=int, default=12)
    parser.add_argument("--feat-dim", type=int, default=32)
    parser.add_argument("--feature-scale", type=float, default=0.55)
    parser.add_argument("--feature-noise-min", type=float, default=1.2)
    parser.add_argument("--feature-noise-max", type=float, default=2.0)
    parser.add_argument("--same-p0", type=float, default=0.018)
    parser.add_argument("--cross-p0", type=float, default=0.22)
    parser.add_argument("--same-gamma", type=float, default=0.94)
    parser.add_argument("--cross-gamma", type=float, default=0.025)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--train-times", type=str, default="")
    parser.add_argument("--val-times", type=str, default="")
    parser.add_argument("--test-times", type=str, default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset = build_dataset(args)
    path = save_dataset(dataset, args.out_dir, args.dataset)
    print(f"Saved TSBM dataset: {path}")
    for key, value in dataset["stats"].items():
        if key == "label_time_table":
            continue
        print(f"{key}: {value}")
    print(f"train_times: {dataset['config']['train_times_resolved']}")
    print(f"val_times: {dataset['config']['val_times_resolved']}")
    print(f"test_times: {dataset['config']['test_times_resolved']}")


if __name__ == "__main__":
    main()
