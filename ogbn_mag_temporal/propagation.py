"""Paper-level HGAMLP propagation and SMP/UMP/GSMP variants for ogbn-mag.

The implementation intentionally starts with the paper citation graph because it
is the minimum robust channel required by the requested ablation and avoids
silently inventing features for node types that do not have raw attributes.

Extension point: add more meta-path channels, such as paper-author-paper, inside
``_build_paper_feature_dict``. The model already accepts any number of feature
channels as long as each channel is a paper-node tensor.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def assign_time_bucket(
    year: int | float | torch.Tensor,
    bucket_style: str = "yearly",
) -> int | torch.Tensor:
    """Assign ogbn-mag publication years to temporal buckets.

    ``coarse`` buckets:
        0 = unknown, 1 = train_past (<=2017), 2 = val_proxy (2018),
        3 = test_future (>=2019)

    ``yearly`` buckets:
        0 = unknown, otherwise the integer publication year.
    """
    if bucket_style not in {"coarse", "yearly"}:
        raise ValueError(f"Unknown bucket_style '{bucket_style}'. Use coarse or yearly.")

    if isinstance(year, torch.Tensor):
        years = year.detach().cpu().view(-1).float()
        finite = torch.isfinite(years) & (years > 0)
        if bucket_style == "yearly":
            buckets = torch.zeros_like(years, dtype=torch.long)
            buckets[finite] = years[finite].round().long()
            return buckets

        buckets = torch.zeros_like(years, dtype=torch.long)
        buckets[finite & (years <= 2017)] = 1
        buckets[finite & (years == 2018)] = 2
        buckets[finite & (years >= 2019)] = 3
        return buckets

    if year is None:
        return 0
    value = float(year)
    if not torch.isfinite(torch.tensor(value)) or value <= 0:
        return 0
    if bucket_style == "yearly":
        return int(round(value))
    if value <= 2017:
        return 1
    if value == 2018:
        return 2
    return 3


def normalize_adj(
    edge_index: torch.Tensor,
    num_src: int,
    num_dst: int,
    edge_weight: Optional[torch.Tensor] = None,
    mode: str = "dst",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize directed edge weights for stable propagation.

    ``edge_index`` uses PyG convention ``[source, destination]``. With
    ``mode='dst'`` each destination row sums to one over incoming edges.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    if num_src <= 0 or num_dst <= 0:
        raise ValueError(f"num_src and num_dst must be positive, got {num_src}, {num_dst}")

    edge_index = edge_index.detach().cpu().long()
    num_edges = edge_index.shape[1]
    if edge_weight is None:
        edge_weight = torch.ones(num_edges, dtype=torch.float32)
    else:
        edge_weight = edge_weight.detach().cpu().float().view(-1)

    if edge_weight.numel() != num_edges:
        raise ValueError(
            f"edge_weight length {edge_weight.numel()} does not match edge count {num_edges}."
        )
    if not torch.isfinite(edge_weight).all():
        raise ValueError("edge_weight contains non-finite values.")

    src = edge_index[0]
    dst = edge_index[1]
    valid = (src >= 0) & (src < num_src) & (dst >= 0) & (dst < num_dst)
    if not valid.all():
        invalid = int((~valid).sum().item())
        raise ValueError(f"edge_index contains {invalid} out-of-range edges.")

    if mode == "none":
        return edge_index, edge_weight
    if mode not in {"dst", "src"}:
        raise ValueError(f"Unknown normalization mode '{mode}'.")

    if mode == "dst":
        denom = torch.zeros(num_dst, dtype=torch.float32)
        denom.index_add_(0, dst, edge_weight)
        norm = edge_weight / denom[dst].clamp_min(1e-12)
    else:
        denom = torch.zeros(num_src, dtype=torch.float32)
        denom.index_add_(0, src, edge_weight)
        norm = edge_weight / denom[src].clamp_min(1e-12)

    if not torch.isfinite(norm).all():
        raise ValueError("Normalized edge weights contain non-finite values.")
    return edge_index, norm


def _propagate_once(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_dst: int,
    chunk_size: int = 250_000,
) -> torch.Tensor:
    """One chunked message-passing step: out[dst] += weight * x[src]."""
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got {tuple(x.shape)}")
    src = edge_index[0]
    dst = edge_index[1]
    out = torch.zeros(num_dst, x.shape[1], dtype=x.dtype)
    for start in range(0, edge_index.shape[1], chunk_size):
        end = min(start + chunk_size, edge_index.shape[1])
        messages = x.index_select(0, src[start:end])
        messages = messages * edge_weight[start:end].unsqueeze(1).to(messages.dtype)
        out.index_add_(0, dst[start:end], messages)
    return out


def propagate_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    num_hops: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> list[torch.Tensor]:
    """Return ``[A X, A^2 X, ... A^K X]`` for a homogeneous paper graph."""
    if num_hops < 0:
        raise ValueError("num_hops must be non-negative.")
    if x.shape[0] != num_nodes:
        raise ValueError(f"x has {x.shape[0]} rows but num_nodes={num_nodes}.")
    if num_hops == 0:
        return []

    norm_edge_index, norm_edge_weight = normalize_adj(
        edge_index=edge_index,
        num_src=num_nodes,
        num_dst=num_nodes,
        edge_weight=edge_weight,
        mode="dst",
    )
    h = x.detach().cpu().float().contiguous()
    hops = []
    for _ in range(num_hops):
        h = _propagate_once(h, norm_edge_index, norm_edge_weight, num_nodes)
        hops.append(h.contiguous())
    return hops


def _get_paper_features(data: Any) -> torch.Tensor:
    if "paper" not in getattr(data, "node_types", []):
        raise ValueError("ogbn-mag data does not contain a 'paper' node type.")
    x = getattr(data["paper"], "x", None)
    if x is None:
        raise ValueError(
            "Missing data['paper'].x. This pipeline uses paper features as the "
            "base HGAMLP input; provide paper features or plug in trainable "
            "embeddings before propagation."
        )
    if x.ndim != 2:
        raise ValueError(f"data['paper'].x must be 2D, got {tuple(x.shape)}.")
    if not torch.isfinite(x.float()).all():
        raise ValueError("data['paper'].x contains NaN or Inf values.")
    return x.detach().cpu().float().contiguous()


def _get_paper_years(data: Any, years: Optional[torch.Tensor] = None) -> torch.Tensor:
    if years is None:
        years = getattr(data["paper"], "year", None)
    if years is None:
        raise ValueError(
            "Missing paper publication years. ogbn-mag temporal variants require "
            "data['paper'].year."
        )
    years = years.detach().cpu().view(-1)
    num_nodes_attr = getattr(data["paper"], "num_nodes", None)
    num_papers = int(num_nodes_attr) if num_nodes_attr is not None else years.numel()
    if years.numel() != num_papers:
        raise ValueError(
            f"Paper year length {years.numel()} does not match num_papers={num_papers}."
        )
    if not torch.isfinite(years.float()).any():
        raise ValueError("Paper years contain no finite values.")
    return years


def _get_paper_citation_edge_index(data: Any) -> torch.Tensor:
    preferred = ("paper", "cites", "paper")
    if preferred in getattr(data, "edge_types", []):
        edge_index = data[preferred].edge_index
    else:
        paper_edges = [
            edge_type
            for edge_type in getattr(data, "edge_types", [])
            if edge_type[0] == "paper" and edge_type[2] == "paper"
        ]
        if not paper_edges:
            raise ValueError(
                "Could not find a paper-paper citation edge type. Expected "
                "('paper', 'cites', 'paper') in ogbn-mag."
            )
        edge_type = paper_edges[0]
        logger.warning("Using paper-paper edge type %s as citation channel.", edge_type)
        edge_index = data[edge_type].edge_index

    if edge_index is None or edge_index.numel() == 0:
        raise ValueError("Paper citation edge_index is missing or empty.")
    return edge_index.detach().cpu().long().contiguous()


def _paper_split(split_idx: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in ("train", "valid", "test"):
        value = split_idx[key]
        if isinstance(value, Mapping):
            value = value["paper"]
        out[key] = value.detach().cpu().long().view(-1)
    return out


def _domain_masks(
    split_idx: Mapping[str, Any],
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    split = _paper_split(split_idx)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[split["train"]] = True
    valid_mask[split["valid"]] = True
    return train_mask, valid_mask


def _transition_ratio_weights(
    edge_index: torch.Tensor,
    years: torch.Tensor,
    split_idx: Mapping[str, Any],
    bucket_style: str,
    eps: float,
    max_ratio: float = 1000.0,
) -> torch.Tensor:
    """Estimate P_valid(bucket_pair) / P_train(bucket_pair) for an edge channel."""
    if eps <= 0:
        raise ValueError("eps must be positive.")
    num_nodes = years.numel()
    train_mask, valid_mask = _domain_masks(split_idx, num_nodes)
    src = edge_index[0]
    dst = edge_index[1]

    buckets = assign_time_bucket(years, bucket_style=bucket_style)
    assert isinstance(buckets, torch.Tensor)
    src_bucket = buckets[src]
    dst_bucket = buckets[dst]
    stride = int(max(src_bucket.max().item(), dst_bucket.max().item()) + 1)
    pair_id = src_bucket * stride + dst_bucket

    source_domain = train_mask[dst]
    target_domain = valid_mask[dst]
    support_mask = source_domain | target_domain
    if source_domain.sum() == 0 or target_domain.sum() == 0 or support_mask.sum() == 0:
        logger.warning(
            "No train/valid destination support for temporal ratios; using unit weights."
        )
        return torch.ones(edge_index.shape[1], dtype=torch.float32)

    support_ids = torch.unique(pair_id[support_mask], sorted=True)
    num_pairs = support_ids.numel()

    source_pos = torch.searchsorted(support_ids, pair_id[source_domain])
    target_pos = torch.searchsorted(support_ids, pair_id[target_domain])
    source_counts = torch.bincount(source_pos, minlength=num_pairs).float()
    target_counts = torch.bincount(target_pos, minlength=num_pairs).float()

    source_total = source_counts.sum()
    target_total = target_counts.sum()
    source_prob = (source_counts + eps) / (source_total + eps * num_pairs)
    target_prob = (target_counts + eps) / (target_total + eps * num_pairs)
    ratio = (target_prob / source_prob).clamp(min=1.0 / max_ratio, max=max_ratio)

    weights = torch.ones(edge_index.shape[1], dtype=torch.float32)
    all_pos = torch.searchsorted(support_ids, pair_id)
    in_range = all_pos < num_pairs
    matched = torch.zeros_like(in_range)
    matched[in_range] = support_ids[all_pos[in_range]] == pair_id[in_range]
    weights[matched] = ratio[all_pos[matched]]
    return weights


def _bucket_balanced_weights(
    edge_index: torch.Tensor,
    years: torch.Tensor,
    bucket_style: str,
) -> torch.Tensor:
    """Uniform message-passing correction by inverse temporal bucket-pair count."""
    src = edge_index[0]
    dst = edge_index[1]
    buckets = assign_time_bucket(years, bucket_style=bucket_style)
    assert isinstance(buckets, torch.Tensor)
    src_bucket = buckets[src]
    dst_bucket = buckets[dst]
    stride = int(max(src_bucket.max().item(), dst_bucket.max().item()) + 1)
    pair_id = src_bucket * stride + dst_bucket
    _, inverse = torch.unique(pair_id, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse).float()
    return 1.0 / counts[inverse].clamp_min(1.0)


def _build_paper_feature_dict(
    data: Any,
    num_hops: int,
    forward_weight: Optional[torch.Tensor] = None,
    reverse_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    x = _get_paper_features(data)
    num_papers = x.shape[0]
    edge_index = _get_paper_citation_edge_index(data)
    reverse_edge_index = edge_index.flip(0).contiguous()

    features: Dict[str, torch.Tensor] = {"paper": x}

    channels = [
        ("paper__cites__paper", edge_index, forward_weight),
        ("paper__cited_by__paper", reverse_edge_index, reverse_weight),
    ]

    for channel_name, channel_edge_index, channel_weight in channels:
        logger.info(
            "Building %s propagation: hops=%d, edges=%d",
            channel_name,
            num_hops,
            channel_edge_index.shape[1],
        )
        hops = propagate_features(
            x=x,
            edge_index=channel_edge_index,
            num_nodes=num_papers,
            num_hops=num_hops,
            edge_weight=channel_weight,
        )
        for hop_idx, hop_x in enumerate(hops, start=1):
            key = f"{channel_name}_hop{hop_idx}"
            features[key] = hop_x

    return features


def build_base_propagation(data: Any, num_hops: int) -> Dict[str, torch.Tensor]:
    """Ordinary HGAMLP-HOPE paper citation propagation."""
    return _build_paper_feature_dict(data=data, num_hops=num_hops)


def build_smp_propagation(
    data: Any,
    split_idx: Mapping[str, Any],
    years: torch.Tensor,
    num_hops: int,
    bucket_style: str = "coarse",
) -> Dict[str, torch.Tensor]:
    """Source-to-target temporal reweighted propagation for HH+SMP."""
    years = _get_paper_years(data, years)
    edge_index = _get_paper_citation_edge_index(data)
    forward_weight = _transition_ratio_weights(
        edge_index=edge_index,
        years=years,
        split_idx=split_idx,
        bucket_style=bucket_style,
        eps=1e-3,
    )
    reverse_edge_index = edge_index.flip(0).contiguous()
    reverse_weight = _transition_ratio_weights(
        edge_index=reverse_edge_index,
        years=years,
        split_idx=split_idx,
        bucket_style=bucket_style,
        eps=1e-3,
    )
    return _build_paper_feature_dict(
        data=data,
        num_hops=num_hops,
        forward_weight=forward_weight,
        reverse_weight=reverse_weight,
    )


def build_ump_propagation(
    data: Any,
    split_idx: Mapping[str, Any],
    years: torch.Tensor,
    num_hops: int,
    bucket_style: str = "coarse",
) -> Dict[str, torch.Tensor]:
    """Uniform/bucket-balanced message-passing correction for HH+UMP."""
    del split_idx
    years = _get_paper_years(data, years)
    edge_index = _get_paper_citation_edge_index(data)
    forward_weight = _bucket_balanced_weights(edge_index, years, bucket_style)
    reverse_edge_index = edge_index.flip(0).contiguous()
    reverse_weight = _bucket_balanced_weights(reverse_edge_index, years, bucket_style)
    return _build_paper_feature_dict(
        data=data,
        num_hops=num_hops,
        forward_weight=forward_weight,
        reverse_weight=reverse_weight,
    )


def build_gsmp_propagation(
    data: Any,
    split_idx: Mapping[str, Any],
    years: torch.Tensor,
    num_hops: int,
    bucket_style: str = "yearly",
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Generalized source message passing for HH+GSMP.

    Ratios are estimated from train-paper destination edges (source domain) and
    validation-paper destination edges (target proxy). Test labels are never read,
    and test destination edges are not counted when estimating ratios.
    """
    years = _get_paper_years(data, years)
    edge_index = _get_paper_citation_edge_index(data)
    forward_weight = _transition_ratio_weights(
        edge_index=edge_index,
        years=years,
        split_idx=split_idx,
        bucket_style=bucket_style,
        eps=eps,
    )
    reverse_edge_index = edge_index.flip(0).contiguous()
    reverse_weight = _transition_ratio_weights(
        edge_index=reverse_edge_index,
        years=years,
        split_idx=split_idx,
        bucket_style=bucket_style,
        eps=eps,
    )
    return _build_paper_feature_dict(
        data=data,
        num_hops=num_hops,
        forward_weight=forward_weight,
        reverse_weight=reverse_weight,
    )
