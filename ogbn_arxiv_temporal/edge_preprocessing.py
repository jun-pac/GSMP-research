from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
from torch_geometric.utils import to_undirected


@dataclass
class EdgePreprocessResult:
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    stats: dict[str, object] = field(default_factory=dict)


def preprocess_edges(
    edge_index: torch.Tensor,
    node_year: torch.Tensor,
    num_nodes: int,
    mode: str,
    make_undirected: bool = True,
) -> EdgePreprocessResult:
    mode = mode.lower()
    if mode not in {"baseline", "smp", "ump", "gsmp"}:
        raise ValueError(f"Unknown mode: {mode}")

    edge_index = edge_index.long().cpu()
    node_year = node_year.view(-1).long().cpu()
    original_edges = int(edge_index.size(1))

    if make_undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    aggregation_edges_before = int(edge_index.size(1))

    stats = {
        "mode": mode,
        "num_nodes": int(num_nodes),
        "original_directed_edges": original_edges,
        "aggregation_edges_before_preprocessing": aggregation_edges_before,
        "make_undirected": bool(make_undirected),
        "year_min": int(node_year.min().item()),
        "year_max": int(node_year.max().item()),
    }

    if mode == "baseline":
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
    elif mode == "smp":
        edge_weight, smp_stats = compute_smp_weights(edge_index, node_year, num_nodes)
        stats.update(smp_stats)
    elif mode == "ump":
        edge_index, edge_weight, ump_stats = apply_ump_filter(edge_index, node_year, num_nodes)
        stats.update(ump_stats)
    else:
        edge_weight, gsmp_stats = compute_gsmp_weights(edge_index, node_year, num_nodes)
        stats.update(gsmp_stats)

    stats["edges_after_preprocessing"] = int(edge_index.size(1))
    stats["weight_stats"] = weight_stats(edge_weight)
    return EdgePreprocessResult(edge_index=edge_index.contiguous(), edge_weight=edge_weight.contiguous(), stats=stats)


def compute_smp_weights(
    edge_index: torch.Tensor,
    node_year: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    src, dst = edge_index
    src_time = node_year[src].float()
    dst_time = node_year[dst].float()
    t_min = float(node_year.min().item())
    t_max = float(node_year.max().item())

    boundary = torch.minimum(
        torch.full_like(dst_time, t_max) - dst_time,
        dst_time - torch.full_like(dst_time, t_min),
    )
    single_mask = (src_time == dst_time) | ((src_time - dst_time).abs() > boundary)
    raw_weight = torch.where(
        single_mask,
        torch.full_like(src_time, 2.0),
        torch.ones_like(src_time),
    )
    edge_weight = normalize_to_target_mean_one(edge_index, raw_weight, num_nodes)

    return edge_weight, {
        "smp_single_edges": int(single_mask.sum().item()),
        "smp_double_edges": int((~single_mask).sum().item()),
        "smp_single_fraction": float(single_mask.float().mean().item()) if single_mask.numel() else 0.0,
        "smp_raw_weight_stats": weight_stats(raw_weight),
    }


def apply_ump_filter(
    edge_index: torch.Tensor,
    node_year: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    src, dst = edge_index
    keep_mask = node_year[src] <= node_year[dst]
    kept_edge_index = edge_index[:, keep_mask]
    edge_weight = torch.ones(kept_edge_index.size(1), dtype=torch.float32)

    before = int(edge_index.size(1))
    after = int(kept_edge_index.size(1))
    dropped = before - after
    incoming_degree = torch.bincount(kept_edge_index[1], minlength=num_nodes)
    zero_nodes = torch.nonzero(incoming_degree == 0, as_tuple=False).view(-1)
    zero_incoming = int(zero_nodes.numel())
    if zero_incoming > 0:
        fallback_loops = torch.stack([zero_nodes, zero_nodes], dim=0)
        kept_edge_index = torch.cat([kept_edge_index, fallback_loops], dim=1)
        edge_weight = torch.ones(kept_edge_index.size(1), dtype=torch.float32)

    return kept_edge_index, edge_weight, {
        "ump_edges_dropped": dropped,
        "ump_edges_dropped_pct": (100.0 * dropped / before) if before else 0.0,
        "ump_zero_incoming_nodes": zero_incoming,
        "ump_self_loop_fallback_edges": zero_incoming,
    }


def compute_gsmp_weights(
    edge_index: torch.Tensor,
    node_year: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    src, dst = edge_index
    unique_years, year_inverse = torch.unique(node_year, sorted=True, return_inverse=True)
    src_year_id = year_inverse[src]
    num_years = int(unique_years.numel())

    key = dst * num_years + src_year_id
    group_counts = torch.bincount(key, minlength=num_nodes * num_years).float()
    base_weight = 1.0 / group_counts[key].clamp_min(1.0)

    degree = torch.bincount(dst, minlength=num_nodes).float()
    sum_base = torch.zeros(num_nodes, dtype=torch.float32)
    sum_base.index_add_(0, dst, base_weight)
    mean_base = sum_base[dst] / degree[dst].clamp_min(1.0)
    edge_weight = base_weight / mean_base.clamp_min(torch.finfo(torch.float32).eps)

    return edge_weight, {
        "gsmp_num_years": num_years,
        "gsmp_nonempty_target_year_groups": int((group_counts > 0).sum().item()),
        "gsmp_base_weight_stats": weight_stats(base_weight),
    }


def normalize_to_target_mean_one(
    edge_index: torch.Tensor,
    raw_weight: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    dst = edge_index[1]
    degree = torch.bincount(dst, minlength=num_nodes).float()
    weight_sum = torch.zeros(num_nodes, dtype=torch.float32)
    weight_sum.index_add_(0, dst, raw_weight.float())
    scale = degree[dst] / weight_sum[dst].clamp_min(torch.finfo(torch.float32).eps)
    return raw_weight.float() * scale


def weight_stats(edge_weight: Optional[torch.Tensor]) -> dict[str, float]:
    if edge_weight is None or edge_weight.numel() == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    edge_weight = edge_weight.float()
    return {
        "min": float(edge_weight.min().item()),
        "mean": float(edge_weight.mean().item()),
        "max": float(edge_weight.max().item()),
    }


def stats_as_log_lines(stats: dict[str, object]) -> list[str]:
    lines = [
        f"nodes={stats['num_nodes']}",
        f"original_directed_edges={stats['original_directed_edges']}",
        f"aggregation_edges_before_preprocessing={stats['aggregation_edges_before_preprocessing']}",
        f"edges_after_preprocessing={stats['edges_after_preprocessing']}",
        f"make_undirected={stats['make_undirected']}",
        f"node_year_range=[{stats['year_min']}, {stats['year_max']}]",
    ]
    if stats.get("mode") == "ump":
        lines.extend(
            [
                f"ump_edges_dropped={stats['ump_edges_dropped']}",
                f"ump_edges_dropped_pct={stats['ump_edges_dropped_pct']:.2f}",
                f"ump_zero_incoming_nodes={stats['ump_zero_incoming_nodes']}",
                f"ump_self_loop_fallback_edges={stats['ump_self_loop_fallback_edges']}",
            ]
        )
    if stats.get("mode") in {"smp", "gsmp"}:
        weight = stats["weight_stats"]
        lines.append(
            "weight_min_mean_max="
            f"{weight['min']:.6f}/{weight['mean']:.6f}/{weight['max']:.6f}"
        )
    return lines
