#!/usr/bin/env python3
"""Shared data helpers for the Pokec temporal GCN/GSMP experiment."""

from __future__ import annotations

import gzip
import json
import os
import random
import re
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch


PROFILES_URL = "https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz"
RELATIONSHIPS_URL = "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz"

USER_ID_COL = 0
PUBLIC_COL = 1
COMPLETION_COL = 2
GENDER_COL = 3
LAST_LOGIN_COL = 5
REGISTRATION_COL = 6
AGE_COL = 7


def log(message: str) -> None:
    print(message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def download_if_needed(url: str, path: Path) -> None:
    if path.exists():
        log(f"[download] found {path}; skipping")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    log(f"[download] fetching {url} -> {path}")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def existing_path(root: Path, gz_name: str) -> Path:
    gz_path = root / gz_name
    plain_path = root / gz_name.removesuffix(".gz")
    if gz_path.exists():
        return gz_path
    if plain_path.exists():
        return plain_path
    return gz_path


def parse_registration_year(value: str) -> int:
    value = value.strip()
    if not value or value.lower() == "null":
        return -1
    if len(value) >= 4 and value[:4].isdigit():
        return int(value[:4])
    match = re.search(r"(?:19|20)\d{2}", value)
    return int(match.group(0)) if match else -1


def parse_float(value: str, default: float = 0.0) -> float:
    value = value.strip()
    if not value or value.lower() == "null":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_label(value: str) -> int:
    value = value.strip()
    if not value or value.lower() == "null":
        return -1
    try:
        label = int(value)
    except ValueError:
        return -1
    return label if label in (0, 1) else -1


def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_temporal_tensors(data_dir: Path) -> Dict[str, torch.Tensor]:
    names = (
        "x",
        "y",
        "year_raw",
        "year_idx",
        "split_group",
        "train_idx",
        "valid_idx",
        "test_idx",
        "edge_index_directed",
        "edge_index_undirected_self_loop",
    )
    tensors = {}
    for name in names:
        path = data_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing preprocessed tensor: {path}")
        tensors[name] = safe_torch_load(path)
    return tensors


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    std[std == 0] = 1.0
    return (x - mean) / std


def require_pyg_utils():
    try:
        from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected
    except Exception as exc:
        raise ImportError(
            "torch_geometric is required for graph preprocessing. Install PyG or use "
            "an environment that already has tunedGNN dependencies."
        ) from exc
    return add_self_loops, remove_self_loops, to_undirected


def make_undirected_self_loop(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    add_self_loops, remove_self_loops, to_undirected = require_pyg_utils()
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    return edge_index.contiguous()


def add_self_loops_to_directed(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    add_self_loops, remove_self_loops, _ = require_pyg_utils()
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    return edge_index.contiguous()


def class_distribution(y: torch.Tensor, idx: torch.Tensor, num_classes: int) -> Dict[str, int]:
    labels = y[idx]
    return {str(c): int((labels == c).sum().item()) for c in range(num_classes)}


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    if y.numel() == 0:
        return float("nan")
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


def gpu_memory_string(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "cpu"
    idx = device.index if device.index is not None else torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(idx) / (1024**3)
    reserved = torch.cuda.memory_reserved(idx) / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated(idx) / (1024**3)
    return f"alloc={allocated:.2f}GB reserved={reserved:.2f}GB max={max_allocated:.2f}GB"


def edge_index_diagnostics(edge_index: torch.Tensor, num_nodes: int, name: str) -> Dict[str, int]:
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"{name} must have shape [2, num_edges], got {tuple(edge_index.shape)}")
    min_id = int(edge_index.min().item()) if edge_index.numel() else 0
    max_id = int(edge_index.max().item()) if edge_index.numel() else -1
    if min_id < 0 or max_id >= num_nodes:
        raise ValueError(
            f"{name} contains out-of-range node ids: min={min_id}, max={max_id}, "
            f"num_nodes={num_nodes}"
        )
    return {
        "num_edges": int(edge_index.size(1)),
        "min_node_id": min_id,
        "max_node_id": max_id,
    }
