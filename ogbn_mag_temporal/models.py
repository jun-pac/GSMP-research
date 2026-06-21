"""HGAMLP-HOPE-style model used by the HH ablation pipeline.

This is a faithful, modular approximation designed for the new experiment
runner. It consumes precomputed paper-node propagation channels and can be
replaced by the official HGAMLP-HOPE implementation later without changing the
training script contract.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HOPEExpertBlock(nn.Module):
    """Small HOPE-style prototype/expert routing approximation."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        num_experts: int = 4,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        self.num_experts = num_experts
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return (gate.unsqueeze(-1) * expert_outputs).sum(dim=1)


class HHModel(nn.Module):
    """HGAMLP-HOPE-style classifier over propagated paper feature channels.

    Args:
        input_dims: Mapping from feature_dict key to feature dimension.
        hidden_dim: Shared channel projection dimension.
        num_classes: Number of ogbn-mag paper classes.
        dropout: Dropout probability.
        use_hope: Enable the HOPE-style expert block.

    ``feature_dict`` passed to ``forward`` should contain paper-node tensors such
    as ``paper`` and ``paper__cites__paper_hop1``. All tensors must have the same
    first dimension for the current full graph or mini-batch.
    """

    def __init__(
        self,
        input_dims: Mapping[str, int],
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        use_hope: bool = True,
        num_experts: int = 4,
    ) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims must contain at least one feature channel.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1.")

        self.feature_keys = list(input_dims.keys())
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_hope = use_hope

        self.projections = nn.ModuleDict(
            OrderedDict(
                (key, nn.Linear(int(input_dims[key]), hidden_dim))
                for key in self.feature_keys
            )
        )
        self.channel_norms = nn.ModuleDict(
            OrderedDict((key, nn.LayerNorm(hidden_dim)) for key in self.feature_keys)
        )
        self.semantic_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

        # TODO: Replace this approximation with the official HGAMLP-HOPE
        # implementation if available. The interface can stay as
        # forward(feature_dict), because propagation is already materialized.
        self.hope_block: Optional[nn.Module]
        if use_hope:
            self.hope_block = HOPEExpertBlock(
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_experts=num_experts,
            )
        else:
            self.hope_block = None

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feature_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return logits for paper nodes represented by ``feature_dict``."""
        missing = [key for key in self.feature_keys if key not in feature_dict]
        if missing:
            raise KeyError(f"feature_dict is missing required channels: {missing}")

        batch_size = None
        channel_embeddings = []
        for key in self.feature_keys:
            x = feature_dict[key]
            if x.ndim != 2:
                raise ValueError(f"Feature '{key}' must be 2D, got {tuple(x.shape)}.")
            if batch_size is None:
                batch_size = x.shape[0]
            elif x.shape[0] != batch_size:
                raise ValueError(
                    f"Feature '{key}' has {x.shape[0]} rows, expected {batch_size}."
                )

            h = self.projections[key](x)
            h = self.channel_norms[key](h)
            h = F.relu(h)
            h = self.dropout(h)
            channel_embeddings.append(h)

        stacked = torch.stack(channel_embeddings, dim=1)
        attention_scores = self.semantic_attention(stacked)
        attention = F.softmax(attention_scores, dim=1)
        fused = (stacked * attention).sum(dim=1)

        if self.hope_block is not None:
            fused = self.hope_block(fused)

        logits = self.classifier(fused)
        return logits
