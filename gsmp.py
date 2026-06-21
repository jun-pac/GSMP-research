from typing import Optional, Tuple

import torch
from torch import Tensor


def _valid_time_mask(time: Tensor) -> Tensor:
    if time.is_floating_point():
        return torch.isfinite(time) & (time >= 0)
    return time >= 0


def compute_gsmp_edge_weights(
    edge_index: Tensor,
    node_time: Tensor,
    num_nodes: Optional[int] = None,
    max_bincount_size: int = 50_000_000,
) -> Tensor:
    """
    Compute General Symmetrized Message Passing edge weights.

    For each source node u, outgoing neighbors v are grouped by timestamp
    time[v]. The edge u -> v receives inverse-frequency weight within the
    source-local timestamp group, then the outgoing weights of u are normalized
    to have average 1.

    This realizes the paper-style time-indexed neighbor-set balancing as
    per-edge weights usable by PyG layers, e.g.:

        out = conv(x, edge_index, edge_weight=edge_weight)

    Args:
        edge_index: LongTensor [2, num_edges], source nodes in row 0 and
            target nodes in row 1.
        node_time: Tensor [num_nodes], timestamp/year for each node. Negative
            or non-finite timestamps are treated as missing and get raw weight 1.
        num_nodes: Optional number of nodes. If omitted, inferred from inputs.
        max_bincount_size: Use a fast bincount path when
            num_nodes * num_unique_times is no larger than this.

    Returns:
        Tensor [num_edges] with GSMP weights in edge order.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    edge_index = edge_index.long()
    src, dst = edge_index
    num_edges = edge_index.size(1)
    if num_nodes is None:
        inferred = int(edge_index.max().item()) + 1 if num_edges > 0 else 0
        num_nodes = max(inferred, int(node_time.numel()))

    edge_b = torch.ones(num_edges, dtype=torch.float32, device=edge_index.device)
    if num_edges == 0:
        return edge_b

    node_time = node_time.to(edge_index.device)
    target_time = node_time[dst]
    valid = _valid_time_mask(target_time)

    if valid.any():
        valid_pos = torch.where(valid)[0]
        src_valid = src[valid_pos]
        target_time_valid = target_time[valid_pos]
        _, time_group = torch.unique(target_time_valid, sorted=True, return_inverse=True)
        num_time_groups = int(time_group.max().item()) + 1

        pair_key = src_valid * num_time_groups + time_group
        key_space = int(num_nodes) * num_time_groups

        if key_space <= max_bincount_size:
            counts = torch.bincount(pair_key, minlength=key_space)
            count_per_edge = counts[pair_key]
        else:
            _, inverse, counts = torch.unique(
                pair_key, sorted=False, return_inverse=True, return_counts=True
            )
            count_per_edge = counts[inverse]

        edge_b[valid_pos] = 1.0 / count_per_edge.to(torch.float32).clamp(min=1)

    outgoing_count = torch.bincount(src, minlength=num_nodes).to(torch.float32)
    b_sum = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    b_sum.scatter_add_(0, src, edge_b)
    mu = torch.zeros_like(b_sum)
    has_outgoing = outgoing_count > 0
    mu[has_outgoing] = b_sum[has_outgoing] / outgoing_count[has_outgoing]

    return edge_b / mu[src].clamp(min=1e-12)


def gsmp_toy_example() -> Tuple[Tensor, Tensor]:
    """Return weights for the prompt's [2020, 2020, 2020, 2021, 2022] example."""
    edge_index = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
    )
    node_time = torch.tensor([-1, 2020, 2020, 2020, 2021, 2022], dtype=torch.long)
    weights = compute_gsmp_edge_weights(edge_index, node_time, num_nodes=6)
    expected = torch.tensor([5 / 9, 5 / 9, 5 / 9, 5 / 3, 5 / 3], dtype=torch.float32)
    return weights, expected


if __name__ == "__main__":
    weights, expected = gsmp_toy_example()
    print("computed:", weights.tolist())
    print("expected:", expected.tolist())
    print("matches:", torch.allclose(weights, expected))
