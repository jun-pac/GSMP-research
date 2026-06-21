from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import fcntl
import shutil
import time
import numpy as np
import pandas as pd
import torch
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.data import Data


COMMON_FEATURE_KEYS = (
    "x",
    "features",
    "feat",
    "emb",
    "embs",
    "embedding",
    "embeddings",
    "node_emb",
    "node_embs",
    "node_embeddings",
    "x_embs",
)


@dataclass
class ArxivBundle:
    data: Data
    split_idx: dict[str, torch.Tensor]
    evaluator: Evaluator
    num_classes: int
    node_year: torch.Tensor
    feature_source: str


def load_ogbn_arxiv(
    data_root: Path,
    repo_root: Path,
    features_path: Optional[str],
    allow_auto_features: bool = True,
) -> ArxivBundle:
    data_root.mkdir(parents=True, exist_ok=True)
    with ogb_dataset_lock(data_root):
        quarantine_corrupt_ogbn_arxiv(data_root)
        with torch_load_weights_only_false():
            dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(data_root))
            data = dataset[0]
    data.y = data.y.view(-1).long()

    node_year = load_node_year(data, dataset).view(-1).long()
    if node_year.numel() != data.num_nodes:
        raise ValueError(
            f"node_year has {node_year.numel()} rows, but ogbn-arxiv has {data.num_nodes} nodes."
        )

    feature_source = apply_feature_source(
        data=data,
        features_path=features_path,
        repo_root=repo_root,
        allow_auto_features=allow_auto_features,
    )

    split_idx = {name: idx.long() for name, idx in dataset.get_idx_split().items()}
    evaluator = Evaluator(name="ogbn-arxiv")

    return ArxivBundle(
        data=data,
        split_idx=split_idx,
        evaluator=evaluator,
        num_classes=dataset.num_classes,
        node_year=node_year,
        feature_source=feature_source,
    )


@contextmanager
def torch_load_weights_only_false():
    """OGB/PyG processed Data objects are trusted local files, not tensor-only checkpoints."""
    original_load = torch.load

    def load_with_legacy_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_legacy_default
    try:
        yield
    finally:
        torch.load = original_load


class ogb_dataset_lock:
    def __init__(self, data_root: Path):
        self.path = data_root / ".ogbn_arxiv_download.lock"
        self.handle = None

    def __enter__(self):
        self.handle = self.path.open("w")
        print(f"Waiting for OGB dataset lock: {self.path}", flush=True)
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        print(f"Acquired OGB dataset lock: {self.path}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
        print(f"Released OGB dataset lock: {self.path}", flush=True)
        return False


def quarantine_corrupt_ogbn_arxiv(data_root: Path) -> None:
    dataset_dir = data_root / "ogbn_arxiv"
    raw_node_feat = dataset_dir / "raw" / "node-feat.csv.gz"
    if not raw_node_feat.exists():
        return

    with raw_node_feat.open("rb") as handle:
        magic = handle.read(2)
    if magic == b"\x1f\x8b":
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    quarantine_path = data_root / f"ogbn_arxiv.corrupt_{timestamp}"
    print(
        f"Detected corrupt ogbn-arxiv raw file {raw_node_feat} with magic={magic!r}; "
        f"moving dataset to {quarantine_path}",
        flush=True,
    )
    shutil.move(str(dataset_dir), str(quarantine_path))

    zip_path = data_root / "arxiv.zip"
    if zip_path.exists():
        zip_quarantine = data_root / f"arxiv.corrupt_{timestamp}.zip"
        print(f"Moving existing OGB zip to {zip_quarantine}", flush=True)
        shutil.move(str(zip_path), str(zip_quarantine))


def load_node_year(data: Data, dataset: PygNodePropPredDataset) -> torch.Tensor:
    for attr in ("node_year", "node_years", "year", "years"):
        if hasattr(data, attr):
            value = getattr(data, attr)
            if value is not None:
                return torch.as_tensor(value).view(-1)

    root = Path(dataset.root)
    candidates = [
        root / "raw" / "node_year.csv.gz",
        root / "raw" / "node-year.csv.gz",
        root / "raw" / "node_year.csv",
        root / "raw" / "node-year.csv",
    ]
    candidates.extend(sorted(root.rglob("*year*.csv*")))

    for path in candidates:
        if path.is_file():
            frame = pd.read_csv(path, compression="infer", header=None)
            year = torch.from_numpy(frame.values.reshape(-1))
            if year.numel() == data.num_nodes:
                return year

    searched = "\n  ".join(str(path) for path in candidates[:20])
    raise FileNotFoundError(
        "Could not find ogbn-arxiv node-year metadata on the Data object or in raw CSV files. "
        f"First paths checked:\n  {searched}"
    )


def apply_feature_source(
    data: Data,
    features_path: Optional[str],
    repo_root: Path,
    allow_auto_features: bool,
) -> str:
    selected_path: Optional[Path] = None

    if features_path:
        if features_path == "auto":
            selected_path = discover_simteg_tape_features(repo_root)
            if selected_path is None:
                raise FileNotFoundError(
                    "features_path=auto was requested, but no cached SimTeG/TAPE feature file was found."
                )
        else:
            selected_path = Path(features_path).expanduser().resolve()
            if not selected_path.is_file():
                raise FileNotFoundError(f"features_path does not exist: {selected_path}")
    elif allow_auto_features:
        selected_path = discover_simteg_tape_features(repo_root)

    if selected_path is None:
        if not hasattr(data, "x") or data.x is None:
            raise ValueError("No features_path was provided and OGB data.x is missing.")
        data.x = torch.as_tensor(data.x, dtype=torch.float32)
        print(
            "WARNING: using default ogbn-arxiv features, not SimTeG/TAPE embeddings.",
            flush=True,
        )
        return "ogb_default"

    features = load_feature_matrix(selected_path, num_nodes=data.num_nodes)
    data.x = features
    print(f"Loaded node features from {selected_path} with shape {tuple(features.shape)}", flush=True)
    return str(selected_path)


def discover_simteg_tape_features(repo_root: Path) -> Optional[Path]:
    search_roots = [
        repo_root / "SimTeG" / "out",
        repo_root / "lambda_out",
        repo_root / "out",
        repo_root / "data",
    ]
    patterns = [
        "ogbn-arxiv-tape/**/cached_embs/x_embs.pt",
        "ogbn-arxiv/**/cached_embs/x_embs.pt",
        "ogbn-arxiv-tape/**/*emb*.pt",
        "ogbn-arxiv/**/*emb*.pt",
        "ogbn-arxiv-tape/**/*emb*.npy",
        "ogbn-arxiv/**/*emb*.npy",
    ]

    matches: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches.extend(path for path in root.glob(pattern) if path.is_file())

    if not matches:
        return None

    def rank(path: Path) -> tuple[int, int, str]:
        text = str(path)
        tape_rank = 0 if "ogbn-arxiv-tape" in text else 1
        x_emb_rank = 0 if path.name == "x_embs.pt" else 1
        return (tape_rank, x_emb_rank, text)

    return sorted(set(matches), key=rank)[0]


def load_feature_matrix(path: Path, num_nodes: int) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        tensor = tensor_from_object(obj, num_nodes=num_nodes, source=str(path))
    elif suffix == ".npy":
        tensor = torch.from_numpy(np.asarray(np.load(path))).float()
    elif suffix == ".npz":
        with np.load(path) as obj:
            tensor = tensor_from_object(dict(obj), num_nodes=num_nodes, source=str(path))
    else:
        raise ValueError(
            f"Unsupported features_path extension for {path}. Use .pt, .pth, .npy, or .npz."
        )

    tensor = normalize_feature_shape(tensor, num_nodes=num_nodes, source=str(path))
    return tensor.contiguous().float()


def tensor_from_object(obj: Any, num_nodes: int, source: str) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(np.asarray(obj))

    if isinstance(obj, dict):
        for key in COMMON_FEATURE_KEYS:
            if key in obj:
                return tensor_from_object(obj[key], num_nodes=num_nodes, source=f"{source}:{key}")
        for key, value in obj.items():
            try:
                tensor = tensor_from_object(value, num_nodes=num_nodes, source=f"{source}:{key}")
                return normalize_feature_shape(tensor, num_nodes=num_nodes, source=f"{source}:{key}")
            except (TypeError, ValueError):
                continue

    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            try:
                tensor = tensor_from_object(value, num_nodes=num_nodes, source=f"{source}[{index}]")
                return normalize_feature_shape(tensor, num_nodes=num_nodes, source=f"{source}[{index}]")
            except (TypeError, ValueError):
                continue

    raise TypeError(f"Could not extract a feature tensor from {source}.")


def normalize_feature_shape(tensor: torch.Tensor, num_nodes: int, source: str) -> torch.Tensor:
    tensor = torch.as_tensor(tensor)
    if tensor.ndim == 1:
        tensor = tensor.view(-1, 1)
    if tensor.ndim == 3 and tensor.size(1) == 1:
        tensor = tensor.squeeze(1)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix from {source}, got shape {tuple(tensor.shape)}.")
    if tensor.size(0) != num_nodes:
        raise ValueError(
            f"Feature matrix from {source} has {tensor.size(0)} rows, but ogbn-arxiv has {num_nodes} nodes."
        )
    if not torch.isfinite(tensor.float()).all():
        raise ValueError(f"Feature matrix from {source} contains NaN or Inf values.")
    return tensor
