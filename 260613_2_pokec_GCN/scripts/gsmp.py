#!/usr/bin/env python3
"""Target-wise GSMP weights and sparse GCN normalization."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


def compute_gsmp_edge_weights(
    edge_index: Tensor,
    year_idx: Tensor,
    num_nodes: int,
    num_timestamps: int,
    include_self_loops: bool = False,
) -> Tensor:
    """Compute target-wise inverse timestamp-frequency GSMP weights.

    PyG edge convention is source -> target, so counts are target-wise:
    C_dst(time(src)). By default self-loops are not included in timestamp
    counts and receive weight 1.0.
    """

    if edge_index.device != year_idx.device:
        raise ValueError("edge_index and year_idx must be on the same device.")
    if year_idx.numel() != num_nodes:
        raise ValueError("year_idx length must equal num_nodes.")

    src, dst = edge_index
    src_time = year_idx[src].long()
    is_self_loop = src == dst
    count_mask = src_time >= 0
    if not include_self_loops:
        count_mask = count_mask & (~is_self_loop)

    weights = torch.zeros(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
    if not torch.any(count_mask):
        if not include_self_loops:
            weights[is_self_loop] = 1.0
        return weights

    pair_index = dst[count_mask].long() * int(num_timestamps) + src_time[count_mask]
    counts = torch.bincount(
        pair_index,
        minlength=int(num_nodes) * int(num_timestamps),
    ).to(torch.float32)

    edge_pair_index = dst.long() * int(num_timestamps) + torch.clamp(src_time, min=0)
    pair_counts = counts[edge_pair_index].clamp_min(1.0)
    base = torch.zeros_like(weights)
    base[count_mask] = 1.0 / pair_counts[count_mask]

    target_sum = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    target_count = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    target_sum.index_add_(0, dst[count_mask].long(), base[count_mask])
    target_count.index_add_(
        0,
        dst[count_mask].long(),
        torch.ones(int(count_mask.sum().item()), dtype=torch.float32, device=edge_index.device),
    )

    mean_base = torch.zeros_like(target_sum)
    nonempty_targets = target_count > 0
    mean_base[nonempty_targets] = target_sum[nonempty_targets] / target_count[nonempty_targets]
    valid_mean = count_mask & (mean_base[dst.long()] > 0)
    weights[valid_mean] = base[valid_mean] / mean_base[dst[valid_mean].long()]

    if not include_self_loops:
        weights[is_self_loop] = 1.0

    if not torch.isfinite(weights).all():
        raise FloatingPointError("GSMP edge weights contain NaN or Inf.")
    return weights


def compute_gcn_norm(
    edge_index: Tensor,
    num_nodes: int,
    message_weight: Optional[Tensor] = None,
    degree_weight: Optional[Tensor] = None,
) -> Tensor:
    """Compute source-to-target symmetric GCN normalization.

    Default degree_weight is all ones. For the default GSMP setting, pass
    message_weight=gsmp_weight and leave degree_weight=None, giving
    D_unweighted^{-1/2} A_gsmp D_unweighted^{-1/2}.
    """

    src, dst = edge_index
    device = edge_index.device
    if message_weight is None:
        message_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
    else:
        message_weight = message_weight.to(device=device, dtype=torch.float32)

    if degree_weight is None:
        degree_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
    else:
        degree_weight = degree_weight.to(device=device, dtype=torch.float32)

    deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    deg.index_add_(0, dst.long(), degree_weight)
    deg_inv_sqrt = deg.clamp_min(1.0).pow(-0.5)
    norm = deg_inv_sqrt[src.long()] * message_weight * deg_inv_sqrt[dst.long()]
    if not torch.isfinite(norm).all():
        raise FloatingPointError("GCN edge normalization contains NaN or Inf.")
    return norm


def make_sparse_adj(edge_index: Tensor, edge_norm: Tensor, num_nodes: int) -> Tensor:
    indices = torch.stack([edge_index[1].long(), edge_index[0].long()], dim=0)
    adj = torch.sparse_coo_tensor(
        indices,
        edge_norm.to(dtype=torch.float32),
        (int(num_nodes), int(num_nodes)),
        device=edge_index.device,
    )
    return adj.coalesce()


def gsmp_weight_stats(weights: Tensor) -> Dict[str, float]:
    weights = weights.detach()
    return {
        "min": float(weights.min().item()) if weights.numel() else float("nan"),
        "mean": float(weights.mean().item()) if weights.numel() else float("nan"),
        "max": float(weights.max().item()) if weights.numel() else float("nan"),
    }


def sanity_check_gsmp_identity(device: str = "cpu") -> float:
    """Check our sparse GCN conv matches PyG GCNConv when all weights are one."""

    try:
        from torch_geometric.nn import GCNConv
    except Exception as exc:
        raise ImportError("PyG is required for the GCNConv sanity check.") from exc

    from models import SparseGCNConv

    torch.manual_seed(123)
    dev = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    x = torch.randn(6, 4, device=dev)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 1, 2, 3, 4, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
        device=dev,
    )

    ours = SparseGCNConv(4, 3).to(dev)
    pyg = GCNConv(4, 3, add_self_loops=False, normalize=True).to(dev)
    with torch.no_grad():
        pyg.lin.weight.copy_(ours.lin.weight)
        pyg.bias.copy_(ours.bias)

    edge_norm = compute_gcn_norm(edge_index, x.size(0))
    adj = make_sparse_adj(edge_index, edge_norm, x.size(0))
    out_ours = ours(x, adj)
    out_pyg = pyg(x, edge_index)
    max_abs_diff = float((out_ours - out_pyg).abs().max().item())
    if max_abs_diff > 1e-5:
        raise AssertionError(f"GSMP identity sanity check failed: max_abs_diff={max_abs_diff}")
    return max_abs_diff


if __name__ == "__main__":
    diff = sanity_check_gsmp_identity()
    print(f"sanity_check_gsmp_identity max_abs_diff={diff:.8g}")
