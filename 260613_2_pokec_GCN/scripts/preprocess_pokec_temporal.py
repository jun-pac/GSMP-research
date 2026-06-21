#!/usr/bin/env python3
"""Preprocess SNAP Pokec for chronological GCN/GSMP experiments."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from data_utils import (
    AGE_COL,
    COMPLETION_COL,
    GENDER_COL,
    PROFILES_URL,
    PUBLIC_COL,
    REGISTRATION_COL,
    RELATIONSHIPS_URL,
    USER_ID_COL,
    class_distribution,
    download_if_needed,
    edge_index_diagnostics,
    existing_path,
    log,
    make_undirected_self_loop,
    open_text,
    parse_float,
    parse_label,
    parse_registration_year,
    safe_torch_load,
    write_json,
)


SPLITS = ("train", "valid", "test")


def write_diagnostic_and_raise(out_dir: Path, message: str, details: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"error": message, "details": details, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(out_dir / "alignment_diagnostic.json", payload)
    raise RuntimeError(f"{message}. Wrote {out_dir / 'alignment_diagnostic.json'}")


def parse_raw_profiles(profile_path: Path) -> Dict[str, torch.Tensor]:
    rows = []
    max_user_id = 0
    profile_rows = 0
    valid_registration = 0
    valid_labels = 0
    log(f"[profiles] reading {profile_path}")
    with open_text(profile_path) as f:
        for line in f:
            profile_rows += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(AGE_COL, REGISTRATION_COL, GENDER_COL):
                continue
            try:
                user_id = int(parts[USER_ID_COL])
            except ValueError:
                continue
            max_user_id = max(max_user_id, user_id)
            year = parse_registration_year(parts[REGISTRATION_COL])
            label = parse_label(parts[GENDER_COL])
            if year >= 0:
                valid_registration += 1
            if label >= 0:
                valid_labels += 1

            public = parse_float(parts[PUBLIC_COL])
            completion = parse_float(parts[COMPLETION_COL]) / 100.0
            age_raw = parse_float(parts[AGE_COL])
            age_known = 1.0 if age_raw > 0 else 0.0
            age = age_raw / 100.0 if age_raw > 0 else 0.0
            rows.append((user_id, year, label, public, completion, age, age_known))

            if profile_rows % 500_000 == 0:
                log(
                    f"[profiles] rows={profile_rows:,}, valid_registration={valid_registration:,}, "
                    f"valid_labels={valid_labels:,}"
                )

    if max_user_id <= 0:
        raise ValueError("No parseable user_id values found in profile file.")

    n = max_user_id
    year_raw = torch.full((n,), -1, dtype=torch.long)
    y = torch.full((n,), -1, dtype=torch.long)
    x = torch.zeros((n, 5), dtype=torch.float32)
    seen = torch.zeros(n, dtype=torch.bool)

    for user_id, year, label, public, completion, age, age_known in rows:
        idx = user_id - 1
        if not (0 <= idx < n):
            continue
        seen[idx] = True
        year_raw[idx] = int(year)
        y[idx] = int(label)
        x[idx] = torch.tensor([public, completion, age, age_known, 1.0], dtype=torch.float32)

    log(
        f"[profiles] done: profile_rows={profile_rows:,}, max_user_id={max_user_id:,}, "
        f"seen_users={int(seen.sum().item()):,}, valid_registration={valid_registration:,}, "
        f"valid_labels={valid_labels:,}"
    )
    return {
        "x_raw_profile": x,
        "y_raw_profile": y,
        "year_raw": year_raw,
        "seen_profile": seen,
        "profile_rows": torch.tensor(profile_rows),
        "valid_registration_rows": torch.tensor(valid_registration),
        "valid_label_rows": torch.tensor(valid_labels),
        "max_user_id": torch.tensor(max_user_id),
    }


def _extract_first(payload: Dict, names) -> Optional[object]:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _as_tensor(value, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value.detach().cpu()
    else:
        out = torch.as_tensor(value)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def load_processed_payload(path: Path) -> Dict[str, torch.Tensor]:
    suffix = path.suffix.lower()
    if suffix in (".pt", ".pth"):
        payload = safe_torch_load(path)
        if hasattr(payload, "x") and hasattr(payload, "y"):
            data = {"x": payload.x, "y": payload.y}
            if hasattr(payload, "edge_index"):
                data["edge_index"] = payload.edge_index
            if hasattr(payload, "num_nodes"):
                data["num_nodes"] = int(payload.num_nodes)
            payload = data
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported torch payload type in {path}: {type(payload)}")
    elif suffix == ".npz":
        payload = dict(np.load(path, allow_pickle=True))
    elif suffix == ".mat":
        try:
            from scipy.io import loadmat
        except Exception as exc:
            raise ImportError("Loading .mat files requires scipy. Install scipy or pass .pt/.npz.") from exc
        payload = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    else:
        raise ValueError(f"Unsupported processed file extension: {path.suffix}")

    x_value = _extract_first(payload, ("x", "node_feat", "features", "feat"))
    y_value = _extract_first(payload, ("y", "label", "labels"))
    if x_value is None or y_value is None:
        raise ValueError(
            f"Processed file {path} must contain node features and labels. "
            "Expected keys like x/node_feat and y/label."
        )

    x = _as_tensor(x_value, torch.float32)
    y = _as_tensor(y_value).long().view(-1)
    if y.numel() == x.size(0) and int(y.min().item()) == 1 and int(y.max().item()) == 2:
        y = y - 1

    result = {"x": x, "y": y}
    edge_value = _extract_first(payload, ("edge_index", "edges", "edge_list"))
    if edge_value is not None:
        edge_index = _as_tensor(edge_value).long()
        if edge_index.ndim == 2 and edge_index.size(0) != 2 and edge_index.size(1) == 2:
            edge_index = edge_index.t().contiguous()
        result["edge_index"] = edge_index
    user_id_value = _extract_first(payload, ("user_id", "user_ids", "node_id", "node_ids"))
    if user_id_value is not None:
        result["user_id"] = _as_tensor(user_id_value).long().view(-1)
    return result


def select_features_and_labels(
    args: argparse.Namespace,
    raw: Dict[str, torch.Tensor],
    out_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, str, Dict]:
    processed_path = args.processed_path
    if processed_path is not None and processed_path.exists():
        processed = load_processed_payload(processed_path)
        x = processed["x"].float()
        y = processed["y"].long()
        n_profile = int(raw["max_user_id"].item())
        details = {
            "processed_path": str(processed_path),
            "processed_num_nodes": int(x.size(0)),
            "profile_max_user_id": n_profile,
            "processed_feature_dim": int(x.size(1)),
        }
        if x.size(0) != y.numel():
            write_diagnostic_and_raise(out_dir, "Processed x/y node counts differ", details)
        if x.size(0) != n_profile:
            write_diagnostic_and_raise(
                out_dir,
                "Cannot verify processed node order because node count differs from raw max user_id",
                details,
            )
        if "user_id" in processed:
            user_id = processed["user_id"]
            expected_one_based = torch.arange(1, x.size(0) + 1, dtype=torch.long)
            expected_zero_based = torch.arange(0, x.size(0), dtype=torch.long)
            aligned = torch.equal(user_id, expected_one_based) or torch.equal(user_id, expected_zero_based)
            details["processed_user_id_present"] = True
            details["processed_user_id_aligned"] = bool(aligned)
            if not aligned:
                write_diagnostic_and_raise(out_dir, "Processed user_id order is not raw SNAP order", details)
        else:
            details["processed_user_id_present"] = False
            details["alignment_assumption"] = "processed row i maps to raw SNAP user_id i+1"

        log(f"[features] using processed benchmark features from {processed_path}")
        return x, y, "processed", details

    if args.feature_source == "processed":
        write_diagnostic_and_raise(
            out_dir,
            "Requested processed features but no readable --processed-path was found",
            {"processed_path": str(processed_path) if processed_path else None},
        )
    if not args.allow_raw_profile_features:
        write_diagnostic_and_raise(
            out_dir,
            "No processed Pokec benchmark file found and raw-profile fallback is disabled",
            {
                "processed_path": str(processed_path) if processed_path else None,
                "hint": "Pass --allow-raw-profile-features for a cheap fallback smoke dataset, "
                "or provide --processed-path pokec.mat for tunedGNN-style features.",
            },
        )

    log("[features] warning: using raw profile fallback features, not tunedGNN 65-dim features")
    details = {
        "feature_columns": ["public", "completion_fraction", "age_scaled", "age_known", "bias"],
        "excluded_columns": ["user_id", "gender(label)", "last_login", "registration"],
        "alignment_assumption": "raw profile user_id maps to node index user_id-1",
    }
    return raw["x_raw_profile"], raw["y_raw_profile"], "raw_profile_fallback", details


def build_year_idx(year_raw: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
    valid_years = torch.unique(year_raw[year_raw >= 0]).cpu().numpy()
    years_sorted = np.sort(valid_years.astype(np.int64))
    if years_sorted.size == 0:
        raise ValueError("No valid registration years found.")
    year_to_idx = {int(year): idx for idx, year in enumerate(years_sorted.tolist())}
    year_idx = torch.full_like(year_raw, -1)
    for year, idx in year_to_idx.items():
        year_idx[year_raw == year] = int(idx)
    return year_idx.long(), years_sorted


def build_split(y: torch.Tensor, x: torch.Tensor, year_raw: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
    feature_ok = torch.isfinite(x).all(dim=1)
    valid = (y >= 0) & (year_raw >= 0) & feature_ok
    if int(valid.sum().item()) == 0:
        raise ValueError("No nodes have valid label, feature vector, and registration year.")
    t_max = int(year_raw[valid].max().item())
    masks = {
        "train": valid & (year_raw <= t_max - 2),
        "valid": valid & (year_raw == t_max - 1),
        "test": valid & (year_raw == t_max),
    }
    split = {name: torch.where(mask)[0].long() for name, mask in masks.items()}
    for name, idx in split.items():
        if idx.numel() == 0:
            raise ValueError(f"Chronological split {name} is empty at t_max={t_max}.")
        if (y[idx] >= 0).sum().item() == 0:
            raise ValueError(f"Labels are missing for split {name}.")
    split_group = torch.full((y.numel(),), -1, dtype=torch.long)
    split_group[split["train"]] = 0
    split_group[split["valid"]] = 1
    split_group[split["test"]] = 2
    return split, split_group, t_max


def load_raw_edges(edge_path: Path, num_nodes: int, chunksize: int) -> torch.Tensor:
    import pandas as pd

    src_chunks = []
    dst_chunks = []
    raw_edges = 0
    kept_edges = 0
    log(f"[edges] reading {edge_path} in chunks of {chunksize:,}")
    reader = pd.read_csv(
        edge_path,
        sep=r"\s+",
        header=None,
        names=["src", "dst"],
        usecols=[0, 1],
        dtype={"src": np.int64, "dst": np.int64},
        compression="infer",
        chunksize=chunksize,
        engine="c",
    )
    for chunk_idx, chunk in enumerate(reader, start=1):
        src = chunk["src"].to_numpy(dtype=np.int64) - 1
        dst = chunk["dst"].to_numpy(dtype=np.int64) - 1
        raw_edges += int(src.size)
        valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)
        src_chunks.append(src[valid].astype(np.int64, copy=False))
        dst_chunks.append(dst[valid].astype(np.int64, copy=False))
        kept_edges += int(valid.sum())
        log(f"[edges] chunk={chunk_idx:,}, raw_edges={raw_edges:,}, kept_edges={kept_edges:,}")

    if not src_chunks:
        raise ValueError("No valid relationship edges found.")
    src_all = np.concatenate(src_chunks)
    dst_all = np.concatenate(dst_chunks)
    edge_index = torch.from_numpy(np.stack([src_all, dst_all], axis=0)).long().contiguous()
    log(f"[edges] loaded directed edge_index with {edge_index.size(1):,} edges")
    return edge_index


def matrix_to_csv(matrix: np.ndarray, labels, path: Path, index_name: str) -> None:
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.index.name = index_name
    df.to_csv(path)
    log(f"[output] wrote {path}")


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float64)
    row_sum = matrix.sum(axis=1, keepdims=True)
    out = np.zeros_like(matrix, dtype=np.float64)
    np.divide(matrix, row_sum, out=out, where=row_sum != 0)
    return out


def write_temporal_stats(
    out_dir: Path,
    edge_index: torch.Tensor,
    year_raw: torch.Tensor,
    year_idx: torch.Tensor,
    years_sorted: np.ndarray,
    split_group: torch.Tensor,
) -> Dict:
    src, dst = edge_index
    valid_year = (year_idx[src] >= 0) & (year_idx[dst] >= 0)
    n_years = int(years_sorted.size)
    flat = year_idx[src[valid_year]].numpy() * n_years + year_idx[dst[valid_year]].numpy()
    directed = np.bincount(flat, minlength=n_years * n_years).reshape(n_years, n_years)
    sym = directed + directed.T
    labels = [int(year) for year in years_sorted.tolist()]
    matrix_to_csv(directed, labels, out_dir / "pokec_edge_counts_by_year_directed.csv", "source_registration_year")
    matrix_to_csv(row_normalize(directed), labels, out_dir / "pokec_edge_probs_by_year_directed.csv", "source_registration_year")
    matrix_to_csv(sym, labels, out_dir / "pokec_edge_counts_by_year_symmetrized.csv", "source_registration_year")
    matrix_to_csv(row_normalize(sym), labels, out_dir / "pokec_edge_probs_by_year_symmetrized.csv", "source_registration_year")

    sg_src = split_group[src]
    sg_dst = split_group[dst]
    valid_split = (sg_src >= 0) & (sg_dst >= 0)
    split_flat = sg_src[valid_split].numpy() * 3 + sg_dst[valid_split].numpy()
    split_directed = np.bincount(split_flat, minlength=9).reshape(3, 3)
    split_sym = split_directed + split_directed.T
    split_labels = list(SPLITS)
    matrix_to_csv(
        split_directed,
        split_labels,
        out_dir / "pokec_edge_counts_by_chronological_split_directed.csv",
        "source_split",
    )
    matrix_to_csv(
        split_sym,
        split_labels,
        out_dir / "pokec_edge_counts_by_chronological_split_symmetrized.csv",
        "source_split",
    )
    if 2010 in labels:
        row_idx = labels.index(2010)
        target = pd.DataFrame(
            {
                "source_year": 2010,
                "destination_year": labels,
                "directed_edge_count": directed[row_idx],
                "directed_row_probability": row_normalize(directed)[row_idx],
                "symmetrized_edge_count": sym[row_idx],
                "symmetrized_row_probability": row_normalize(sym)[row_idx],
            }
        )
        target.to_csv(out_dir / "pokec_neighbors_of_year_2010.csv", index=False)

    return {
        "directed_edges_with_valid_years": int(valid_year.sum().item()),
        "self_loops_directed_raw": int((src == dst).sum().item()),
        "split_directed_counts": split_directed.tolist(),
        "split_symmetrized_counts": split_sym.tolist(),
    }


def write_reports(
    out_dir: Path,
    metadata: Dict,
    split: Dict[str, torch.Tensor],
    y: torch.Tensor,
    year_raw: torch.Tensor,
    years_sorted: np.ndarray,
) -> None:
    year_values, year_counts = torch.unique(year_raw[year_raw >= 0], return_counts=True)
    node_counts = pd.DataFrame(
        {
            "registration_year": year_values.numpy().astype(int),
            "num_nodes": year_counts.numpy().astype(np.int64),
        }
    ).sort_values("registration_year")
    node_counts.to_csv(out_dir / "pokec_node_counts_by_registration_year.csv", index=False)

    summary_rows = []
    for name in SPLITS:
        idx = split[name]
        summary_rows.append(
            {
                "split": name,
                "num_nodes": int(idx.numel()),
                "label_distribution": class_distribution(y, idx, metadata["num_classes"]),
                "min_registration_year": int(year_raw[idx].min().item()),
                "max_registration_year": int(year_raw[idx].max().item()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "pokec_chronological_split_summary.csv", index=False)

    report = f"""# Pokec Temporal GCN/GSMP Dataset

Pokec is originally a directed online social network from Slovakia. Nodes are
users and directed edges are friendship relations. The main GCN training graph
is undirected, following the tunedGNN Pokec GCN baseline convention:
`to_undirected`, remove self-loops, then add self-loops.

This chronological split is constructed from SNAP profile registration year and
is not an official Pokec split. Edge timestamps are unavailable, so temporal
structure is node-level rather than edge-level. The reported tunedGNN GCN*
86.33 +/- 0.17 reference is an architecture/hyperparameter reference, not
directly comparable to this chronological split.

- nodes: {metadata['num_nodes']:,}
- directed raw edges: {metadata['num_directed_edges']:,}
- undirected+self-loop training edges: {metadata['num_undirected_self_loop_edges']:,}
- feature dimension: {metadata['num_features']}
- classes: {metadata['num_classes']}
- registration years: {int(years_sorted[0])}-{int(years_sorted[-1])}
- t_max: {metadata['t_max']}
- feature source: {metadata['feature_source']}
"""
    (out_dir / "pokec_dataset_report.md").write_text(report, encoding="utf-8")

    paragraph = (
        "Pokec is a directed online social network from Slovakia whose nodes are "
        "users and whose edges are friendship relations. We construct a temporal "
        "node split from user registration years because Pokec is not distributed "
        "with an official OGB-style chronological split. Let $t_{\\max}$ denote "
        "the latest valid registration year among nodes with valid features and "
        "labels. Nodes with year $\\leq t_{\\max}-2$ are used for training, nodes "
        "with year $t_{\\max}-1$ for validation, and nodes with year $t_{\\max}$ "
        "for testing. The graph is used transductively for message passing, but "
        "validation and test labels are not used for training. Following the "
        "tunedGNN Pokec GCN setting, the main training graph is symmetrized and "
        "self-loops are added. Since edge timestamps are unavailable, temporal "
        "structure is node-level rather than edge-level. The tunedGNN GCN* "
        "accuracy of 86.33 $\\pm$ 0.17\\% is used only as a reference for the "
        "architecture and hyperparameters and is not directly comparable to this "
        "chronological protocol."
    )
    (out_dir / "pokec_dataset_description.tex").write_text(paragraph + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess SNAP Pokec temporal split.")
    parser.add_argument("--snap-dir", type=Path, default=Path("../data/pokec"))
    parser.add_argument("--profile-path", type=Path, default=None)
    parser.add_argument("--edge-path", type=Path, default=None)
    parser.add_argument("--processed-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/pokec_temporal"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--feature-source", choices=("auto", "processed"), default="auto")
    parser.add_argument("--allow-raw-profile-features", action="store_true")
    parser.add_argument("--edge-chunksize", type=int, default=2_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.profile_path or existing_path(args.snap_dir, "soc-pokec-profiles.txt.gz")
    edge_path = args.edge_path or existing_path(args.snap_dir, "soc-pokec-relationships.txt.gz")
    if args.download:
        download_if_needed(PROFILES_URL, profile_path)
        download_if_needed(RELATIONSHIPS_URL, edge_path)
    if not profile_path.exists() or not edge_path.exists():
        raise FileNotFoundError(
            f"Missing SNAP files: profile={profile_path.exists()} edge={edge_path.exists()}. "
            "Pass --download or correct paths."
        )

    raw = parse_raw_profiles(profile_path)
    x, y, feature_source, feature_details = select_features_and_labels(args, raw, args.out_dir)
    num_nodes = int(x.size(0))
    if num_nodes != int(raw["max_user_id"].item()):
        write_diagnostic_and_raise(
            args.out_dir,
            "Selected feature matrix is not aligned to raw SNAP user_id-1 indexing",
            {"num_nodes": num_nodes, "profile_max_user_id": int(raw["max_user_id"].item())},
        )
    year_raw = raw["year_raw"][:num_nodes].long()
    year_idx, years_sorted = build_year_idx(year_raw)
    split, split_group, t_max = build_split(y, x, year_raw)

    edge_index_directed = load_raw_edges(edge_path, num_nodes, args.edge_chunksize)
    edge_index_diagnostics(edge_index_directed, num_nodes, "edge_index_directed")
    edge_index_undirected_self_loop = make_undirected_self_loop(edge_index_directed, num_nodes)
    edge_index_diagnostics(edge_index_undirected_self_loop, num_nodes, "edge_index_undirected_self_loop")

    temporal_stats = write_temporal_stats(
        args.out_dir,
        edge_index_directed,
        year_raw,
        year_idx,
        years_sorted,
        split_group,
    )

    num_classes = int(y[y >= 0].max().item() + 1) if int((y >= 0).sum().item()) else 0
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile_path": str(profile_path),
        "edge_path": str(edge_path),
        "processed_path": str(args.processed_path) if args.processed_path else None,
        "feature_source": feature_source,
        "feature_details": feature_details,
        "num_nodes": num_nodes,
        "num_features": int(x.size(1)),
        "num_classes": num_classes,
        "num_directed_edges": int(edge_index_directed.size(1)),
        "num_undirected_self_loop_edges": int(edge_index_undirected_self_loop.size(1)),
        "min_registration_year": int(years_sorted[0]),
        "max_registration_year": int(years_sorted[-1]),
        "t_max": int(t_max),
        "split_counts": {name: int(idx.numel()) for name, idx in split.items()},
        "label_distribution": {
            name: class_distribution(y, idx, num_classes) for name, idx in split.items()
        },
        "temporal_stats": temporal_stats,
        "main_training_graph": "undirected_self_loop",
        "directed_graph_available_for_ablation": True,
    }

    tensors = {
        "x": x.float().contiguous(),
        "y": y.long().contiguous(),
        "year_raw": year_raw.long().contiguous(),
        "year_idx": year_idx.long().contiguous(),
        "split_group": split_group.long().contiguous(),
        "train_idx": split["train"],
        "valid_idx": split["valid"],
        "test_idx": split["test"],
        "edge_index_directed": edge_index_directed.long().contiguous(),
        "edge_index_undirected_self_loop": edge_index_undirected_self_loop.long().contiguous(),
    }
    for name, tensor in tensors.items():
        path = args.out_dir / f"{name}.pt"
        torch.save(tensor, path)
        log(f"[output] wrote {path}")
    write_json(args.out_dir / "metadata.json", metadata)
    write_reports(args.out_dir, metadata, split, y, year_raw, years_sorted)
    log(f"[done] preprocessing complete: {args.out_dir}")


if __name__ == "__main__":
    main()
