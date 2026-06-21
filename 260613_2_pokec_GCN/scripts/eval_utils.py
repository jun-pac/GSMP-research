#!/usr/bin/env python3
"""Evaluation and logging helpers."""

from __future__ import annotations

import time
from typing import Dict

import torch
from torch import Tensor

from data_utils import accuracy, gpu_memory_string


@torch.no_grad()
def evaluate_full_graph(
    model: torch.nn.Module,
    x: Tensor,
    y: Tensor,
    split: Dict[str, Tensor],
    adj_first: Tensor,
    adj_rest: Tensor,
    device: torch.device,
) -> Dict[str, float]:
    was_training = model.training
    original_device = next(model.parameters()).device
    model.eval()
    model.to(device)
    x_dev = x.to(device)
    y_dev = y.to(device)
    split_dev = {name: idx.to(device) for name, idx in split.items()}
    logits = model(x_dev, adj_first, adj_rest)
    metrics = {
        name: accuracy(logits[idx], y_dev[idx])
        for name, idx in split_dev.items()
    }
    if original_device != device:
        model.to(original_device)
    if was_training:
        model.train()
    return metrics


def format_epoch_log(
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    best_valid: float,
    test_at_best_valid: float,
    best_epoch: int,
    device: torch.device,
    start_time: float,
) -> str:
    elapsed = time.time() - start_time
    return (
        f"epoch={epoch:04d} loss={loss:.6f} "
        f"train_acc={metrics['train']:.6f} valid_acc={metrics['valid']:.6f} "
        f"test_acc={metrics['test']:.6f} best_valid={best_valid:.6f} "
        f"test_at_best_valid={test_at_best_valid:.6f} best_epoch={best_epoch} "
        f"gpu_mem={gpu_memory_string(device)} elapsed={elapsed:.1f}s"
    )
