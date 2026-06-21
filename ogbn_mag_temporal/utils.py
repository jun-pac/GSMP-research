"""Shared utilities for ogbn-mag HH/HGAMLP-HOPE ablation runs."""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


METRIC_COLUMNS = [
    "method",
    "seed",
    "epoch",
    "loss",
    "train_acc",
    "valid_acc",
    "test_acc",
    "best_valid_acc",
    "best_test_acc",
    "elapsed_sec",
]

SUMMARY_COLUMNS = [
    "method",
    "seed",
    "best_valid_acc",
    "best_test_acc",
    "final_train_acc",
    "final_valid_acc",
    "final_test_acc",
]


def set_seed(seed: int) -> None:
    """Set Python, NumPy, PyTorch, and CUDA seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic kernels help reproducibility; benchmark=False avoids
    # nondeterministic autotuning. Some sparse/index_add kernels can still vary
    # slightly across hardware, which is normal for large graph workloads.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_dirs(paths: Iterable[str | os.PathLike[str]]) -> None:
    """Create several directories."""
    for path in paths:
        ensure_dir(path)


def get_device(device_str: str) -> torch.device:
    """Resolve a torch device and fail clearly if CUDA was requested but absent."""
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{device_str}', but torch.cuda.is_available() is False. "
            "Use --device cpu or run on a GPU node."
        )
    return torch.device(device_str)


def resolve_device(device_str: str) -> torch.device:
    """Alias used by the HH ablation trainer."""
    return get_device(device_str)


def move_heterodata_to_device(data: Any, device: torch.device) -> Any:
    """Move a PyG HeteroData-like object to a device."""
    if hasattr(data, "to"):
        return data.to(device)

    for node_type in getattr(data, "node_types", []):
        store = data[node_type]
        for attr in ("x", "y", "year"):
            if getattr(store, attr, None) is not None:
                setattr(store, attr, getattr(store, attr).to(device))

    for edge_type in getattr(data, "edge_types", []):
        store = data[edge_type]
        for attr in ("edge_index", "edge_weight"):
            if getattr(store, attr, None) is not None:
                setattr(store, attr, getattr(store, attr).to(device))

    return data


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute multiclass accuracy from logits and integer labels."""
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape [N, C], got {tuple(logits.shape)}")
    labels = labels.view(-1)
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Logit/label shape mismatch: {logits.shape[0]} predictions vs "
            f"{labels.shape[0]} labels"
        )
    if labels.numel() == 0:
        raise ValueError("Cannot compute accuracy on an empty label tensor.")
    pred = logits.argmax(dim=-1)
    return (pred == labels).float().mean().item()


class Timer:
    """Small wall-clock timer."""

    def __init__(self) -> None:
        self.start_time = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time

    def reset(self) -> None:
        self.start_time = time.perf_counter()


class CSVLogger:
    """Append-only CSV logger that writes and flushes each row immediately."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        fieldnames: Sequence[str],
        overwrite: bool = True,
    ) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        ensure_dir(self.path.parent)
        if overwrite or not self.path.exists():
            with self.path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
                handle.flush()
                os.fsync(handle.fileno())

    def append(self, row: Mapping[str, Any]) -> None:
        clean_row = {field: row.get(field, "") for field in self.fieldnames}
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(clean_row)
            handle.flush()
            os.fsync(handle.fileno())


def append_csv_row(
    path: str | os.PathLike[str],
    fieldnames: Sequence[str],
    row: Mapping[str, Any],
    create_header: bool = True,
) -> None:
    """Append one row to a CSV, creating the header when needed."""
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if create_header and not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())


def upsert_summary_row(
    summary_path: str | os.PathLike[str],
    row: Mapping[str, Any],
    key_fields: Sequence[str] = ("method", "seed"),
) -> None:
    """Insert or replace a row in results/summary.csv."""
    summary_path = Path(summary_path)
    ensure_dir(summary_path.parent)

    rows: List[Dict[str, Any]] = []
    if summary_path.exists() and summary_path.stat().st_size > 0:
        with summary_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for existing in reader:
                rows.append(dict(existing))

    row_key = tuple(str(row.get(field, "")) for field in key_fields)
    replaced = False
    for idx, existing in enumerate(rows):
        existing_key = tuple(str(existing.get(field, "")) for field in key_fields)
        if existing_key == row_key:
            rows[idx] = {field: row.get(field, "") for field in SUMMARY_COLUMNS}
            replaced = True
            break

    if not replaced:
        rows.append({field: row.get(field, "") for field in SUMMARY_COLUMNS})

    tmp_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, summary_path)


def save_results(results: Dict[str, Any], save_path: str) -> None:
    """Save a JSON result dictionary."""
    ensure_dir(Path(save_path).parent)
    with open(save_path, "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    logger.info("Results saved to %s", save_path)


def load_results(save_path: str) -> Dict[str, Any]:
    """Load a JSON result dictionary."""
    with open(save_path, "r") as handle:
        return json.load(handle)


def configure_logger(log_file: Optional[str] = None) -> None:
    """Configure Python logging for older scripts in this directory."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        ensure_dir(Path(log_file).parent)
        handlers.insert(0, logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers)


def format_eval_line(
    method: str,
    seed: int,
    epoch: int,
    train_acc: float,
    valid_acc: float,
    test_acc: float,
    loss: float,
    best_valid: float,
    best_test: float,
    elapsed: float,
) -> str:
    """Format the clean tail-friendly evaluation line requested by the pipeline."""
    return (
        f"[METHOD={method}][SEED={seed}][EPOCH={epoch:03d}] "
        f"train_acc={train_acc:.4f} valid_acc={valid_acc:.4f} "
        f"test_acc={test_acc:.4f} loss={loss:.4f} "
        f"best_valid={best_valid:.4f} best_test={best_test:.4f} "
        f"elapsed={elapsed:.1f}s"
    )


def validate_split(split_idx: Mapping[str, torch.Tensor]) -> None:
    """Check that train/valid/test splits exist and are nonempty."""
    for name in ("train", "valid", "test"):
        if name not in split_idx:
            raise ValueError(f"Missing split_idx['{name}'].")
        if split_idx[name].numel() == 0:
            raise ValueError(f"split_idx['{name}'] is empty.")


def print_heterodata_info(data: Any, timestamp_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
    """Print compact information about a PyG HeteroData object."""
    logger.info("=" * 60)
    logger.info("HeteroData Information")
    logger.info("=" * 60)

    logger.info("Node types:")
    for node_type in getattr(data, "node_types", []):
        store = data[node_type]
        num_nodes = getattr(store, "num_nodes", None)
        has_features = getattr(store, "x", None) is not None
        has_labels = getattr(store, "y", None) is not None
        logger.info(
            "  %s: nodes=%s, features=%s, labels=%s",
            node_type,
            num_nodes,
            has_features,
            has_labels,
        )

    logger.info("Edge types:")
    for edge_type in getattr(data, "edge_types", []):
        edge_index = getattr(data[edge_type], "edge_index", None)
        num_edges = edge_index.shape[1] if edge_index is not None else 0
        logger.info("  %s: edges=%d", edge_type, num_edges)

    if timestamp_dict:
        logger.info("Timestamp information:")
        for node_type, timestamps in timestamp_dict.items():
            valid = timestamps[torch.isfinite(timestamps.float())]
            if valid.numel() > 0:
                logger.info(
                    "  %s: min=%.0f, max=%.0f, valid=%d/%d",
                    node_type,
                    valid.min().item(),
                    valid.max().item(),
                    valid.numel(),
                    timestamps.numel(),
                )


def print_method_info(method: str, data: Any, timestamp_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
    """Print method-specific information for older scripts."""
    del timestamp_dict
    logger.info("=" * 60)
    logger.info("Method: %s", method.upper())
    logger.info("=" * 60)
    for edge_type in getattr(data, "edge_types", []):
        store = data[edge_type]
        edge_weight = getattr(store, "edge_weight", None)
        if edge_weight is not None and edge_weight.numel() > 0:
            logger.info(
                "  %s: weight min=%.4f max=%.4f mean=%.4f",
                edge_type,
                edge_weight.min().item(),
                edge_weight.max().item(),
                edge_weight.float().mean().item(),
            )
