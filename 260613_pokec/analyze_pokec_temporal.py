#!/usr/bin/env python3
"""
Analyze SNAP Pokec registration-time structure and build an OGB-style split.

Example Colab/Linux command:
python analyze_pokec_temporal.py \
  --download \
  --profile-path soc-pokec-profiles.txt.gz \
  --edge-path soc-pokec-relationships.txt.gz \
  --out-dir pokec_temporal_outputs \
  --target-year-row 2010

Memory-light local command with already downloaded files:
python analyze_pokec_temporal.py \
  --profile-path ../data/pokec/soc-pokec-profiles.txt.gz \
  --edge-path ../data/pokec/soc-pokec-relationships.txt.gz \
  --out-dir pokec_temporal_outputs \
  --target-year-row 2010 \
  --duplicate-check none
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROFILE_URL = "https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz"
EDGE_URL = "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz"
REGISTRATION_COL = 6
SENTINEL_YEAR = -1
SPLITS = ("train", "valid", "test")


@dataclass
class ProfileData:
    profile_rows: int
    valid_registration_rows: int
    filtered_registration_rows: int
    duplicate_user_rows: int
    user_ids: np.ndarray
    years: np.ndarray
    years_sorted: np.ndarray
    node_counts_by_year: pd.DataFrame


@dataclass
class YearLookup:
    kind: str
    sentinel: int
    dtype: np.dtype
    array: Optional[np.ndarray] = None
    mapping: Optional[Dict[int, int]] = None
    max_index: int = -1


@dataclass
class SplitData:
    t_max: int
    split_ids: Dict[str, np.ndarray]
    summary: pd.DataFrame


@dataclass
class EdgeStats:
    raw_edges: int
    valid_year_edges: int
    self_loops: int
    duplicate_directed_edges: Optional[int]
    duplicate_note: str
    directed_year_counts: np.ndarray
    directed_split_counts: np.ndarray


def log(message: str) -> None:
    print(message, flush=True)


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def download_if_needed(profile_path: Path, edge_path: Path) -> None:
    downloads = (
        (PROFILE_URL, profile_path),
        (EDGE_URL, edge_path),
    )
    for url, path in downloads:
        if path.exists():
            log(f"[download] found {path}; skipping")
            continue
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


def extract_registration_year(value: str) -> Optional[int]:
    value = value.strip()
    if not value or value.lower() == "null":
        return None
    if len(value) >= 4 and value[:4].isdigit():
        year = int(value[:4])
        return year if 1 <= year <= 9999 else None
    match = re.search(r"(?:19|20)\d{2}", value)
    if match:
        return int(match.group(0))
    return None


def year_dtype_for(years: np.ndarray) -> np.dtype:
    if years.size == 0:
        return np.dtype(np.int16)
    min_year = int(np.min(years))
    max_year = int(np.max(years))
    if np.iinfo(np.int16).min <= min_year and max_year <= np.iinfo(np.int16).max:
        return np.dtype(np.int16)
    return np.dtype(np.int32)


def load_registration_years(
    profile_path: Path,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    progress_interval: int = 500_000,
) -> ProfileData:
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Profile file not found: {profile_path}. Pass --download to fetch it."
        )

    profile_rows = 0
    valid_registration_rows = 0
    filtered_registration_rows = 0
    user_ids: List[int] = []
    years: List[int] = []

    log(f"[profiles] reading {profile_path}")
    with open_text(profile_path) as f:
        for line in f:
            profile_rows += 1
            parts = line.rstrip("\n").split("\t", REGISTRATION_COL + 1)
            if len(parts) <= REGISTRATION_COL:
                continue
            try:
                user_id = int(parts[0])
            except ValueError:
                continue
            year = extract_registration_year(parts[REGISTRATION_COL])
            if year is None:
                continue
            valid_registration_rows += 1
            if min_year is not None and year < min_year:
                filtered_registration_rows += 1
                continue
            if max_year is not None and year > max_year:
                filtered_registration_rows += 1
                continue
            user_ids.append(user_id)
            years.append(year)
            if profile_rows % progress_interval == 0:
                log(
                    f"[profiles] rows={profile_rows:,}, valid={valid_registration_rows:,}, "
                    f"kept={len(user_ids):,}"
                )

    if not user_ids:
        raise ValueError("No users with parseable registration years after filtering.")

    user_ids_arr = np.asarray(user_ids, dtype=np.int64)
    years_arr = np.asarray(years, dtype=year_dtype_for(np.asarray(years, dtype=np.int32)))

    unique_ids, first_indices = np.unique(user_ids_arr, return_index=True)
    duplicate_user_rows = int(user_ids_arr.size - unique_ids.size)
    if duplicate_user_rows:
        log(
            f"[profiles] warning: found {duplicate_user_rows:,} duplicate user_id rows; "
            "keeping first occurrence"
        )
        first_indices = np.sort(first_indices)
        user_ids_arr = user_ids_arr[first_indices]
        years_arr = years_arr[first_indices]

    year_values, year_counts = np.unique(years_arr.astype(np.int32), return_counts=True)
    node_counts_by_year = pd.DataFrame(
        {
            "registration_year": year_values.astype(int),
            "num_nodes": year_counts.astype(np.int64),
        }
    )
    node_counts_by_year["fraction_of_valid_nodes"] = (
        node_counts_by_year["num_nodes"] / int(user_ids_arr.size)
    )

    log(
        f"[profiles] done: rows={profile_rows:,}, valid_rows={valid_registration_rows:,}, "
        f"kept_unique_users={user_ids_arr.size:,}, years={int(year_values[0])}-{int(year_values[-1])}"
    )
    return ProfileData(
        profile_rows=profile_rows,
        valid_registration_rows=valid_registration_rows,
        filtered_registration_rows=filtered_registration_rows,
        duplicate_user_rows=duplicate_user_rows,
        user_ids=user_ids_arr,
        years=years_arr,
        years_sorted=year_values.astype(np.int32),
        node_counts_by_year=node_counts_by_year,
    )


def build_year_lookup(
    user_ids: np.ndarray,
    years: np.ndarray,
    max_lookup_memory_mb: int = 512,
    min_density_for_array: float = 0.05,
) -> YearLookup:
    if user_ids.size == 0:
        raise ValueError("Cannot build lookup for empty user list.")

    max_user_id = int(np.max(user_ids))
    min_user_id = int(np.min(user_ids))
    dtype = year_dtype_for(years.astype(np.int32))
    if min_user_id >= 0:
        dense_bytes = (max_user_id + 1) * np.dtype(dtype).itemsize
        dense_mb = dense_bytes / (1024 * 1024)
        density = user_ids.size / max(max_user_id + 1, 1)
    else:
        dense_mb = float("inf")
        density = 0.0

    if dense_mb <= max_lookup_memory_mb and density >= min_density_for_array:
        lookup = np.full(max_user_id + 1, SENTINEL_YEAR, dtype=dtype)
        lookup[user_ids] = years.astype(dtype, copy=False)
        log(
            f"[lookup] using dense NumPy array: max_user_id={max_user_id:,}, "
            f"density={density:.2%}, memory={dense_mb:.1f} MiB"
        )
        return YearLookup(
            kind="array",
            sentinel=SENTINEL_YEAR,
            dtype=np.dtype(dtype),
            array=lookup,
            max_index=max_user_id,
        )

    mapping = {int(user_id): int(year) for user_id, year in zip(user_ids, years)}
    log(
        f"[lookup] using dictionary mapping: users={len(mapping):,}, "
        f"dense_estimate={dense_mb:.1f} MiB, density={density:.2%}"
    )
    return YearLookup(
        kind="dict",
        sentinel=SENTINEL_YEAR,
        dtype=np.dtype(dtype),
        mapping=mapping,
    )


def map_ids_to_years(ids: np.ndarray, lookup: YearLookup) -> np.ndarray:
    if lookup.kind == "array":
        assert lookup.array is not None
        out = np.full(ids.shape[0], lookup.sentinel, dtype=lookup.dtype)
        valid = (ids >= 0) & (ids <= lookup.max_index)
        if np.any(valid):
            out[valid] = lookup.array[ids[valid]]
        return out

    assert lookup.mapping is not None
    mapped = pd.Series(ids, copy=False).map(lookup.mapping)
    return mapped.fillna(lookup.sentinel).to_numpy(dtype=lookup.dtype)


def make_chronological_split(
    user_ids: np.ndarray,
    years: np.ndarray,
    out_dir: Path,
) -> SplitData:
    t_max = int(np.max(years))
    split_masks = {
        "train": years <= t_max - 2,
        "valid": years == t_max - 1,
        "test": years == t_max,
    }
    split_ids = {name: user_ids[mask] for name, mask in split_masks.items()}

    rows = []
    total = int(user_ids.size)
    for name in SPLITS:
        ids = split_ids[name]
        split_years = years[split_masks[name]].astype(np.int32)
        path = out_dir / f"pokec_chronological_split_nodes_{name}.txt"
        np.savetxt(path, ids.astype(np.int64), fmt="%d")
        rows.append(
            {
                "split": name,
                "num_nodes": int(ids.size),
                "fraction_of_valid_nodes": float(ids.size / total) if total else 0.0,
                "min_registration_year": int(np.min(split_years)) if split_years.size else "",
                "max_registration_year": int(np.max(split_years)) if split_years.size else "",
                "rule": (
                    f"year <= {t_max - 2}"
                    if name == "train"
                    else f"year == {t_max - 1}"
                    if name == "valid"
                    else f"year == {t_max}"
                ),
            }
        )
        log(f"[split] wrote {path} ({ids.size:,} nodes)")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "pokec_chronological_split_summary.csv", index=False)
    return SplitData(t_max=t_max, split_ids=split_ids, summary=summary)


def years_to_split_indices(years: np.ndarray, t_max: int) -> np.ndarray:
    out = np.full(years.shape[0], -1, dtype=np.int8)
    out[years <= t_max - 2] = 0
    out[years == t_max - 1] = 1
    out[years == t_max] = 2
    return out


class DuplicateTracker:
    def __init__(self, mode: str, partitions: int, out_dir: Path):
        self.mode = mode
        self.partitions = partitions
        self.disabled_note = ""
        self.temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.paths: List[Path] = []
        self.handles = []

        if mode == "none":
            self.disabled_note = "not computed (--duplicate-check none)"
            return
        if partitions <= 0:
            raise ValueError("--duplicate-partitions must be positive.")

        self.temp_dir = tempfile.TemporaryDirectory(prefix="pokec_dup_partitions_", dir=out_dir)
        temp_path = Path(self.temp_dir.name)
        self.paths = [temp_path / f"part_{idx:04d}.bin" for idx in range(partitions)]
        self.handles = [open(path, "ab") for path in self.paths]
        log(
            f"[duplicates] exact duplicate counting enabled with {partitions} disk partitions "
            f"under {temp_path}"
        )

    @property
    def enabled(self) -> bool:
        return self.mode != "none" and not self.disabled_note

    def add(self, src: np.ndarray, dst: np.ndarray) -> None:
        if not self.enabled:
            return
        uint32_max = np.iinfo(np.uint32).max
        if (
            np.any(src < 0)
            or np.any(dst < 0)
            or int(np.max(src)) > uint32_max
            or int(np.max(dst)) > uint32_max
        ):
            self.disabled_note = (
                "not computed: encountered user_id outside uint32 packing range"
            )
            log(f"[duplicates] warning: {self.disabled_note}")
            return

        packed = (src.astype(np.uint64) << np.uint64(32)) | dst.astype(np.uint64)
        partition_ids = np.remainder(packed, self.partitions).astype(np.int32)
        for partition in range(self.partitions):
            values = packed[partition_ids == partition]
            if values.size:
                values.tofile(self.handles[partition])

    def finish(self) -> Tuple[Optional[int], str]:
        for handle in self.handles:
            handle.close()
        self.handles = []

        if self.mode == "none" or self.disabled_note:
            self.cleanup()
            return None, self.disabled_note

        duplicate_count = 0
        max_partition_edges = 0
        for idx, path in enumerate(self.paths):
            values = np.fromfile(path, dtype=np.uint64)
            max_partition_edges = max(max_partition_edges, int(values.size))
            if values.size > 1:
                values.sort()
                duplicate_count += int(np.count_nonzero(values[1:] == values[:-1]))
            if (idx + 1) % max(1, self.partitions // 8) == 0:
                log(f"[duplicates] scanned {idx + 1}/{self.partitions} partitions")

        note = f"exact via partitioned uint64 sort; largest_partition_edges={max_partition_edges:,}"
        self.cleanup()
        return duplicate_count, note

    def cleanup(self) -> None:
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None


def make_year_index_lookup(years_sorted: np.ndarray) -> Tuple[np.ndarray, int]:
    year_min = int(years_sorted[0])
    year_max = int(years_sorted[-1])
    if year_max - year_min > 20_000:
        raise ValueError(
            f"Registration year range {year_min}-{year_max} is unexpectedly large. "
            "Use --min-year/--max-year to restrict invalid timestamps."
        )
    lookup = np.full(year_max - year_min + 1, -1, dtype=np.int32)
    lookup[years_sorted - year_min] = np.arange(years_sorted.size, dtype=np.int32)
    return lookup, year_min


def edge_chunk_iterator(edge_path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    if not edge_path.exists():
        raise FileNotFoundError(
            f"Edge file not found: {edge_path}. Pass --download to fetch it."
        )
    return pd.read_csv(
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


def maybe_tqdm(iterable: Iterable[pd.DataFrame], enabled: bool) -> Iterable[pd.DataFrame]:
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc="edge chunks", unit="chunk")
    except Exception:
        return iterable


def compute_edge_year_matrix(
    edge_path: Path,
    year_lookup: YearLookup,
    years_sorted: np.ndarray,
    t_max: int,
    out_dir: Path,
    chunksize: int = 2_000_000,
    duplicate_check: str = "partition",
    duplicate_partitions: int = 64,
    use_tqdm: bool = True,
) -> EdgeStats:
    n_years = int(years_sorted.size)
    year_index_lookup, year_min = make_year_index_lookup(years_sorted)
    directed_year_counts = np.zeros((n_years, n_years), dtype=np.int64)
    directed_split_counts = np.zeros((len(SPLITS), len(SPLITS)), dtype=np.int64)

    raw_edges = 0
    valid_year_edges = 0
    self_loops = 0
    start = time.time()
    duplicate_tracker = DuplicateTracker(duplicate_check, duplicate_partitions, out_dir)

    log(f"[edges] reading {edge_path} in chunks of {chunksize:,}")
    reader = edge_chunk_iterator(edge_path, chunksize)
    for chunk_idx, chunk in enumerate(maybe_tqdm(reader, use_tqdm), start=1):
        src = chunk["src"].to_numpy(dtype=np.int64, copy=False)
        dst = chunk["dst"].to_numpy(dtype=np.int64, copy=False)
        chunk_edges = int(src.size)
        raw_edges += chunk_edges
        self_loops += int(np.count_nonzero(src == dst))
        duplicate_tracker.add(src, dst)

        src_year = map_ids_to_years(src, year_lookup)
        dst_year = map_ids_to_years(dst, year_lookup)
        valid = (src_year != SENTINEL_YEAR) & (dst_year != SENTINEL_YEAR)

        if np.any(valid):
            src_valid_year = src_year[valid].astype(np.int32, copy=False)
            dst_valid_year = dst_year[valid].astype(np.int32, copy=False)
            valid_count = int(src_valid_year.size)
            valid_year_edges += valid_count

            src_year_idx = year_index_lookup[src_valid_year - year_min]
            dst_year_idx = year_index_lookup[dst_valid_year - year_min]
            flat_year_idx = src_year_idx * n_years + dst_year_idx
            directed_year_counts += np.bincount(
                flat_year_idx, minlength=n_years * n_years
            ).reshape(n_years, n_years)

            src_split_idx = years_to_split_indices(src_valid_year, t_max)
            dst_split_idx = years_to_split_indices(dst_valid_year, t_max)
            split_valid = (src_split_idx >= 0) & (dst_split_idx >= 0)
            if np.any(split_valid):
                flat_split_idx = (
                    src_split_idx[split_valid].astype(np.int64) * len(SPLITS)
                    + dst_split_idx[split_valid].astype(np.int64)
                )
                directed_split_counts += np.bincount(
                    flat_split_idx, minlength=len(SPLITS) * len(SPLITS)
                ).reshape(len(SPLITS), len(SPLITS))

        elapsed = time.time() - start
        log(
            f"[edges] chunk={chunk_idx:,}, processed={raw_edges:,}, "
            f"valid_year_edges={valid_year_edges:,}, self_loops={self_loops:,}, "
            f"elapsed={elapsed:.1f}s"
        )

    duplicate_directed_edges, duplicate_note = duplicate_tracker.finish()
    log(
        f"[edges] done: raw_edges={raw_edges:,}, valid_year_edges={valid_year_edges:,}, "
        f"self_loops={self_loops:,}, duplicate_edges={duplicate_directed_edges}"
    )
    return EdgeStats(
        raw_edges=raw_edges,
        valid_year_edges=valid_year_edges,
        self_loops=self_loops,
        duplicate_directed_edges=duplicate_directed_edges,
        duplicate_note=duplicate_note,
        directed_year_counts=directed_year_counts,
        directed_split_counts=directed_split_counts,
    )


def row_normalize(counts: np.ndarray) -> np.ndarray:
    counts_float = counts.astype(np.float64)
    row_sums = counts_float.sum(axis=1, keepdims=True)
    probs = np.zeros_like(counts_float, dtype=np.float64)
    np.divide(counts_float, row_sums, out=probs, where=row_sums != 0)
    return probs


def save_matrix_csv(
    matrix: np.ndarray,
    row_labels: Sequence,
    col_labels: Sequence,
    path: Path,
    index_name: str,
) -> pd.DataFrame:
    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    df.index.name = index_name
    df.to_csv(path)
    log(f"[output] wrote {path}")
    return df


def save_matrices(
    out_dir: Path,
    years_sorted: np.ndarray,
    directed_year_counts: np.ndarray,
    directed_split_counts: np.ndarray,
) -> Dict[str, pd.DataFrame]:
    year_labels = [int(year) for year in years_sorted]
    split_labels = list(SPLITS)

    year_probs = row_normalize(directed_year_counts)
    sym_year_counts = directed_year_counts + directed_year_counts.T
    sym_year_probs = row_normalize(sym_year_counts)

    split_probs = row_normalize(directed_split_counts)
    sym_split_counts = directed_split_counts + directed_split_counts.T
    sym_split_probs = row_normalize(sym_split_counts)

    outputs = {
        "year_counts_directed": save_matrix_csv(
            directed_year_counts,
            year_labels,
            year_labels,
            out_dir / "pokec_edge_counts_by_year_directed.csv",
            "source_registration_year",
        ),
        "year_probs_directed": save_matrix_csv(
            year_probs,
            year_labels,
            year_labels,
            out_dir / "pokec_edge_probs_by_year_directed.csv",
            "source_registration_year",
        ),
        "year_counts_symmetrized": save_matrix_csv(
            sym_year_counts,
            year_labels,
            year_labels,
            out_dir / "pokec_edge_counts_by_year_symmetrized.csv",
            "source_registration_year",
        ),
        "year_counts_undirected": save_matrix_csv(
            sym_year_counts,
            year_labels,
            year_labels,
            out_dir / "pokec_edge_counts_by_year_undirected.csv",
            "registration_year",
        ),
        "year_probs_symmetrized": save_matrix_csv(
            sym_year_probs,
            year_labels,
            year_labels,
            out_dir / "pokec_edge_probs_by_year_symmetrized.csv",
            "source_registration_year",
        ),
        "year_probs_undirected": save_matrix_csv(
            sym_year_probs,
            year_labels,
            year_labels,
            out_dir / "pokec_edge_probs_by_year_undirected.csv",
            "registration_year",
        ),
        "split_counts_directed": save_matrix_csv(
            directed_split_counts,
            split_labels,
            split_labels,
            out_dir / "pokec_edge_counts_by_chronological_split_directed.csv",
            "source_split",
        ),
        "split_probs_directed": save_matrix_csv(
            split_probs,
            split_labels,
            split_labels,
            out_dir / "pokec_edge_probs_by_chronological_split_directed.csv",
            "source_split",
        ),
        "split_counts_symmetrized": save_matrix_csv(
            sym_split_counts,
            split_labels,
            split_labels,
            out_dir / "pokec_edge_counts_by_chronological_split_symmetrized.csv",
            "source_split",
        ),
        "split_counts_undirected": save_matrix_csv(
            sym_split_counts,
            split_labels,
            split_labels,
            out_dir / "pokec_edge_counts_by_chronological_split_undirected.csv",
            "split",
        ),
        "split_probs_symmetrized": save_matrix_csv(
            sym_split_probs,
            split_labels,
            split_labels,
            out_dir / "pokec_edge_probs_by_chronological_split_symmetrized.csv",
            "source_split",
        ),
        "split_probs_undirected": save_matrix_csv(
            sym_split_probs,
            split_labels,
            split_labels,
            out_dir / "pokec_edge_probs_by_chronological_split_undirected.csv",
            "split",
        ),
    }
    return outputs


def save_target_year_row(
    out_dir: Path,
    target_year: int,
    years_sorted: np.ndarray,
    year_counts: np.ndarray,
    suffix: str,
    count_column: str,
    probability_column: str,
    log_label: str,
) -> pd.DataFrame:
    path = out_dir / f"pokec_neighbors_of_year_{target_year}_{suffix}.csv"
    year_labels = years_sorted.astype(int).tolist()
    if target_year not in set(year_labels):
        log(f"[target-year] warning: source year {target_year} is not present.")
        df = pd.DataFrame(
            columns=[
                "year",
                "neighbor_year",
                count_column,
                probability_column,
            ]
        )
        df.to_csv(path, index=False)
        return df

    row_idx = year_labels.index(target_year)
    row = year_counts[row_idx]
    row_sum = int(row.sum())
    probs = row / row_sum if row_sum else np.zeros_like(row, dtype=np.float64)
    df = pd.DataFrame(
        {
            "year": target_year,
            "neighbor_year": year_labels,
            count_column: row.astype(np.int64),
            probability_column: probs.astype(np.float64),
        }
    )
    df.to_csv(path, index=False)
    log(f"[target-year] wrote {path}")
    log(f"[target-year] {log_label} neighbors for year {target_year}:")
    for _, record in df.iterrows():
        log(
            f"  {target_year} -- {int(record['neighbor_year'])}: "
            f"{int(record[count_column]):,} "
            f"({record[probability_column]:.4%})"
        )
    return df


def format_int(value: object) -> str:
    if value == "" or pd.isna(value):
        return ""
    return f"{int(value):,}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_basic_statistics(
    out_dir: Path,
    profile: ProfileData,
    split: SplitData,
    edges: EdgeStats,
) -> pd.DataFrame:
    stats = [
        ("profile_rows", profile.profile_rows),
        ("users_with_valid_registration_year", profile.valid_registration_rows),
        ("users_after_year_filter_and_dedup", int(profile.user_ids.size)),
        ("profile_rows_filtered_by_min_max_year", profile.filtered_registration_rows),
        ("duplicate_profile_user_rows", profile.duplicate_user_rows),
        ("min_registration_year", int(profile.years_sorted[0])),
        ("max_registration_year", int(profile.years_sorted[-1])),
        ("chronological_split_t_max", split.t_max),
        ("raw_directed_edges", edges.raw_edges),
        ("edges_with_valid_year_at_both_endpoints", edges.valid_year_edges),
        ("raw_self_loops", edges.self_loops),
        (
            "duplicate_directed_edges",
            edges.duplicate_directed_edges
            if edges.duplicate_directed_edges is not None
            else edges.duplicate_note,
        ),
        ("duplicate_directed_edges_note", edges.duplicate_note),
    ]
    df = pd.DataFrame(stats, columns=["statistic", "value"])
    path = out_dir / "pokec_dataset_statistics.csv"
    df.to_csv(path, index=False)
    log(f"[output] wrote {path}")
    return df


def write_markdown_report(
    out_dir: Path,
    profile: ProfileData,
    split: SplitData,
    edges: EdgeStats,
    target_year: int,
) -> None:
    stats_rows = [
        ("Profile rows", format_int(profile.profile_rows)),
        ("Users with valid registration year", format_int(profile.valid_registration_rows)),
        ("Users after year filtering and de-duplication", format_int(profile.user_ids.size)),
        ("Min registration year", int(profile.years_sorted[0])),
        ("Max registration year", int(profile.years_sorted[-1])),
        ("Raw directed edges", format_int(edges.raw_edges)),
        ("Edges with valid endpoint years", format_int(edges.valid_year_edges)),
        ("Raw self-loops", format_int(edges.self_loops)),
        (
            "Duplicate directed edges",
            format_int(edges.duplicate_directed_edges)
            if edges.duplicate_directed_edges is not None
            else edges.duplicate_note,
        ),
    ]
    split_rows = [
        (
            row["split"],
            format_int(row["num_nodes"]),
            f"{float(row['fraction_of_valid_nodes']):.2%}",
            row["min_registration_year"],
            row["max_registration_year"],
            row["rule"],
        )
        for _, row in split.summary.iterrows()
    ]
    year_count_rows = [
        (
            int(row.registration_year),
            format_int(row.num_nodes),
            f"{float(row.fraction_of_valid_nodes):.2%}",
        )
        for row in profile.node_counts_by_year.itertuples(index=False)
    ]

    report = f"""# SNAP Pokec Temporal Split Report

Pokec is a large directed online social network from Slovakia. Nodes are users,
directed edges are friendship relations, and profile attributes include user
metadata. The dataset is not distributed with an official OGB-style
chronological split, so this preprocessing uses the user registration year as a
node timestamp.

Let `t_max = {split.t_max}` be the latest valid registration year. Nodes with
registration year `<= {split.t_max - 2}` are assigned to train, nodes with year
`{split.t_max - 1}` are assigned to validation, and nodes with year
`{split.t_max}` are assigned to test. The graph is treated transductively: all
nodes and edges remain available for message passing, but labels from
validation/test nodes are not used during training. Since edge timestamps are
not available, temporal structure is defined on nodes, not edges.

For directed edges `(u, v)`, the timestamp-to-timestamp edge matrix is
`C[a,b] = |{{(u,v) in E : t_u = a, t_v = b}}|`, where `t_u` is the registration
year of the source node and `t_v` is the registration year of the destination
node. For undirected GNN preprocessing, this script also reports
`C_sym[a,b] = C[a,b] + C[b,a]`; the same matrix is also saved with
`undirected` filenames for direction-agnostic analysis.

Because Pokec has no official leaderboard or official chronological split,
comparisons on Pokec should be interpreted under this preprocessing protocol.

## Basic Statistics

{markdown_table(["Statistic", "Value"], stats_rows)}

## Chronological Node Split

{markdown_table(["Split", "Nodes", "Fraction", "Min year", "Max year", "Rule"], split_rows)}

## Node Counts by Registration Year

{markdown_table(["Registration year", "Nodes", "Fraction"], year_count_rows)}

## Output Files

- `pokec_node_counts_by_registration_year.csv`
- `pokec_chronological_split_nodes_train.txt`
- `pokec_chronological_split_nodes_valid.txt`
- `pokec_chronological_split_nodes_test.txt`
- `pokec_chronological_split_summary.csv`
- `pokec_edge_counts_by_year_directed.csv`
- `pokec_edge_probs_by_year_directed.csv`
- `pokec_edge_counts_by_year_symmetrized.csv`
- `pokec_edge_probs_by_year_symmetrized.csv`
- `pokec_edge_counts_by_year_undirected.csv`
- `pokec_edge_probs_by_year_undirected.csv`
- `pokec_edge_counts_by_chronological_split_directed.csv`
- `pokec_edge_probs_by_chronological_split_directed.csv`
- `pokec_edge_counts_by_chronological_split_symmetrized.csv`
- `pokec_edge_probs_by_chronological_split_symmetrized.csv`
- `pokec_edge_counts_by_chronological_split_undirected.csv`
- `pokec_edge_probs_by_chronological_split_undirected.csv`
- `pokec_neighbors_of_year_{target_year}_directed.csv`
- `pokec_neighbors_of_year_{target_year}_undirected.csv`
- `pokec_dataset_statistics.csv`
- `pokec_dataset_description.tex`
- `pokec_split_table.tex`
- `pokec_year_edge_heatmap_directed.png`
- `pokec_year_edge_heatmap_symmetrized.png`
"""
    path = out_dir / "pokec_dataset_report.md"
    path.write_text(report, encoding="utf-8")
    log(f"[output] wrote {path}")


def write_latex_report(
    out_dir: Path,
    profile: ProfileData,
    split: SplitData,
    edges: EdgeStats,
) -> None:
    n_nodes = int(profile.user_ids.size)
    n_edges = int(edges.raw_edges)
    valid_edges = int(edges.valid_year_edges)
    year_min = int(profile.years_sorted[0])
    year_max = int(profile.years_sorted[-1])

    paragraph = (
        "Pokec is a large directed online social network from Slovakia in which "
        "nodes represent users and directed edges represent friendship relations. "
        "The profile records include user metadata, including a registration "
        "timestamp. Pokec is not distributed with an official OGB-style "
        "chronological split; therefore, we construct one using registration "
        f"year as the node timestamp. After filtering to users with valid "
        f"registration years, the resulting graph contains {n_nodes:,} users "
        f"spanning registration years {year_min}--{year_max}. Let $t_{{\\max}}$ "
        f"be the latest valid registration year. We assign nodes with year "
        f"$\\leq t_{{\\max}}-2$ to training, nodes with year "
        f"$t_{{\\max}}-1$ to validation, and nodes with year $t_{{\\max}}$ "
        "to testing. The graph is treated transductively: all nodes and edges "
        "remain available for message passing, but labels from validation and "
        "test nodes are not used during training. Since edge timestamps are not "
        "available, temporal structure is defined on nodes rather than edges. "
        f"The raw relationship file contains {n_edges:,} directed edges, of "
        f"which {valid_edges:,} have valid registration years at both endpoints. "
        "Because Pokec has no official leaderboard or official chronological "
        "split, comparisons on Pokec should be interpreted under this "
        "preprocessing protocol."
    )
    desc_path = out_dir / "pokec_dataset_description.tex"
    desc_path.write_text(paragraph + "\n", encoding="utf-8")
    log(f"[output] wrote {desc_path}")

    split_counts = edges.directed_split_counts + edges.directed_split_counts.T
    rows = []
    for split_idx, split_name in enumerate(SPLITS):
        node_count = int(split.split_ids[split_name].size)
        edge_counts = [int(split_counts[split_idx, dst_idx]) for dst_idx in range(len(SPLITS))]
        rows.append((split_name, node_count, *edge_counts, sum(edge_counts)))

    table_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Pokec chronological node split and direction-agnostic edge counts between splits.}",
        "\\label{tab:pokec-split}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Split & Nodes & Train & Valid & Test & Total incident \\\\",
        "\\midrule",
    ]
    for split_name, nodes, to_train, to_valid, to_test, total_out in rows:
        table_lines.append(
            f"{split_name} & {nodes:,} & {to_train:,} & {to_valid:,} & "
            f"{to_test:,} & {total_out:,} \\\\"
        )
    table_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    table_path = out_dir / "pokec_split_table.tex"
    table_path.write_text("\n".join(table_lines), encoding="utf-8")
    log(f"[output] wrote {table_path}")


def make_heatmaps(
    out_dir: Path,
    years_sorted: np.ndarray,
    directed_year_counts: np.ndarray,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"[heatmap] warning: matplotlib unavailable; skipping heatmaps ({exc})")
        return

    year_labels = [str(int(year)) for year in years_sorted]
    matrices = [
        (
            directed_year_counts,
            "Directed Pokec edges by registration year",
            out_dir / "pokec_year_edge_heatmap_directed.png",
        ),
        (
            directed_year_counts + directed_year_counts.T,
            "Symmetrized Pokec edges by registration year",
            out_dir / "pokec_year_edge_heatmap_symmetrized.png",
        ),
    ]

    for matrix, title, path in matrices:
        fig_width = max(7.0, 0.45 * len(year_labels))
        fig_height = max(5.5, 0.42 * len(year_labels))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
        image = ax.imshow(np.log1p(matrix), cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Destination / neighbor registration year")
        ax.set_ylabel("Source registration year")
        ax.set_xticks(np.arange(len(year_labels)))
        ax.set_yticks(np.arange(len(year_labels)))
        ax.set_xticklabels(year_labels, rotation=45, ha="right")
        ax.set_yticklabels(year_labels)
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("log(1 + edge count)")
        fig.savefig(path, dpi=220)
        plt.close(fig)
        log(f"[output] wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SNAP Pokec registration years, build an OGB-style chronological "
            "node split, and compute timestamp mixing matrices."
        )
    )
    parser.add_argument("--profile-path", type=Path, required=True)
    parser.add_argument("--edge-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Download missing SNAP files.")
    parser.add_argument("--min-year", type=int, default=None)
    parser.add_argument("--max-year", type=int, default=None)
    parser.add_argument("--target-year-row", type=int, default=2010)
    parser.add_argument("--edge-chunksize", type=int, default=2_000_000)
    parser.add_argument(
        "--max-lookup-memory-mb",
        type=int,
        default=512,
        help="Maximum dense user_id->year lookup array size before falling back to dict.",
    )
    parser.add_argument(
        "--duplicate-check",
        choices=("partition", "none"),
        default="partition",
        help=(
            "Exact duplicate-edge counting mode. 'partition' is exact and memory-safe "
            "but uses temporary disk; 'none' is fastest/lightest."
        ),
    )
    parser.add_argument(
        "--duplicate-partitions",
        type=int,
        default=64,
        help="Number of disk partitions used for exact duplicate counting.",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bars even if tqdm is installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_if_needed(args.profile_path, args.edge_path)
    else:
        missing = [str(path) for path in (args.profile_path, args.edge_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing input file(s): "
                + ", ".join(missing)
                + ". Re-run with --download to fetch missing SNAP files."
            )

    profile = load_registration_years(
        args.profile_path,
        min_year=args.min_year,
        max_year=args.max_year,
    )
    profile.node_counts_by_year.to_csv(
        args.out_dir / "pokec_node_counts_by_registration_year.csv",
        index=False,
    )
    log(f"[output] wrote {args.out_dir / 'pokec_node_counts_by_registration_year.csv'}")

    year_lookup = build_year_lookup(
        profile.user_ids,
        profile.years,
        max_lookup_memory_mb=args.max_lookup_memory_mb,
    )
    split = make_chronological_split(profile.user_ids, profile.years, args.out_dir)

    edges = compute_edge_year_matrix(
        args.edge_path,
        year_lookup,
        profile.years_sorted,
        split.t_max,
        args.out_dir,
        chunksize=args.edge_chunksize,
        duplicate_check=args.duplicate_check,
        duplicate_partitions=args.duplicate_partitions,
        use_tqdm=not args.no_tqdm,
    )

    save_matrices(
        args.out_dir,
        profile.years_sorted,
        edges.directed_year_counts,
        edges.directed_split_counts,
    )
    save_target_year_row(
        args.out_dir,
        args.target_year_row,
        profile.years_sorted,
        edges.directed_year_counts,
        suffix="directed",
        count_column="directed_edge_count",
        probability_column="directed_row_probability",
        log_label="directed",
    )
    save_target_year_row(
        args.out_dir,
        args.target_year_row,
        profile.years_sorted,
        edges.directed_year_counts + edges.directed_year_counts.T,
        suffix="undirected",
        count_column="undirected_edge_count",
        probability_column="undirected_row_probability",
        log_label="undirected/symmetrized",
    )
    write_basic_statistics(args.out_dir, profile, split, edges)
    write_markdown_report(args.out_dir, profile, split, edges, args.target_year_row)
    write_latex_report(args.out_dir, profile, split, edges)
    make_heatmaps(args.out_dir, profile.years_sorted, edges.directed_year_counts)

    log("[done] Pokec temporal analysis complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n[error] interrupted by user")
        sys.exit(130)
