#!/usr/bin/env python3
"""GCN models for Pokec temporal split experiments."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class SparseGCNConv(nn.Module):
    """GCN layer using a pre-normalized sparse adjacency matrix."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        x = self.lin(x)
        out = torch.sparse.mm(adj, x)
        return out + self.bias


class TemporalGCN(nn.Module):
    """Paper-style residual BatchNorm GCN with optional GSMP first layer.

    The tunedGNN large-graph GCN uses ``local_layers`` hidden GCN layers and a
    separate linear prediction head. ``num_layers`` here follows that meaning.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        in_dropout: float,
        use_bn: bool,
        use_residual: bool,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2.")

        self.dropout = float(dropout)
        self.in_dropout = float(in_dropout)
        self.use_bn = bool(use_bn)
        self.use_residual = bool(use_residual)

        channels = [in_channels] + [hidden_channels] * num_layers
        self.convs = nn.ModuleList(
            SparseGCNConv(channels[layer], channels[layer + 1])
            for layer in range(num_layers)
        )
        self.bns = nn.ModuleList(
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        )
        self.residuals = nn.ModuleList(
            nn.Linear(channels[layer], channels[layer + 1], bias=False)
            for layer in range(num_layers)
        )
        self.pred_linear = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x: Tensor, adj_first: Tensor, adj_rest: Tensor) -> Tensor:
        x = F.dropout(x, p=self.in_dropout, training=self.training)
        for layer, conv in enumerate(self.convs):
            residual = self.residuals[layer](x) if self.use_residual else None
            adj = adj_first if layer == 0 else adj_rest
            h = conv(x, adj)
            if residual is not None:
                h = h + residual

            if self.use_bn:
                h = self.bns[layer](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = h
        return self.pred_linear(x)
