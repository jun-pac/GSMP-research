from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class WeightedMeanSAGEConv(nn.Module):
    """GraphSAGE mean aggregation with optional scalar edge weights."""

    def __init__(self, in_channels: int, out_channels: int, aggregation_chunk_size: int = 200_000):
        super().__init__()
        self.aggregation_chunk_size = int(aggregation_chunk_size)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_root = nn.Linear(in_channels, out_channels, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_neigh.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src, dst = edge_index
        num_nodes = x.size(0)

        if edge_weight is None:
            weights = torch.ones(src.numel(), device=x.device, dtype=x.dtype)
        else:
            weights = edge_weight.to(device=x.device, dtype=x.dtype).view(-1)

        neigh = torch.zeros(num_nodes, x.size(-1), device=x.device, dtype=x.dtype)
        if self.aggregation_chunk_size <= 0 or src.numel() <= self.aggregation_chunk_size:
            neigh.index_add_(0, dst, x[src] * weights.view(-1, 1))
        else:
            for start in range(0, src.numel(), self.aggregation_chunk_size):
                end = min(start + self.aggregation_chunk_size, src.numel())
                chunk_src = src[start:end]
                chunk_dst = dst[start:end]
                chunk_weight = weights[start:end].view(-1, 1)
                neigh.index_add_(0, chunk_dst, x[chunk_src] * chunk_weight)

        denom = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        denom.index_add_(0, dst, weights)
        eps = torch.finfo(denom.dtype).eps
        neigh = neigh / denom.clamp_min(eps).view(-1, 1)

        return self.lin_neigh(neigh) + self.lin_root(x)


class WeightedGraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        aggregation_chunk_size: int = 200_000,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = float(dropout)

        if num_layers == 1:
            self.convs.append(WeightedMeanSAGEConv(in_channels, out_channels, aggregation_chunk_size))
        else:
            self.convs.append(WeightedMeanSAGEConv(in_channels, hidden_channels, aggregation_chunk_size))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(WeightedMeanSAGEConv(hidden_channels, hidden_channels, aggregation_chunk_size))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.convs.append(WeightedMeanSAGEConv(hidden_channels, out_channels, aggregation_chunk_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer_idx, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.bns[layer_idx](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index, edge_weight)
