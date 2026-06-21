#!/usr/bin/env python3
"""Build timestamp connectivity from the actual GCN undirected training graph.

The earlier `undirected` CSVs were symmetrized raw directed-edge counts:
`C + C.T`. That is useful as raw edge accounting, but it overcounts reciprocal
Pokec relationships relative to the paper-style GCN preprocessing, which uses
PyG `to_undirected`, removes self-loops, then adds one self-loop per node.

This script reads the coalesced training `edge_index_undirected_self_loop.pt`
from `260613_2_GCN` and writes coalesced undirected message-edge counts.
Self-loops are excluded from the year-connectivity matrix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


def log(message: str) -> None:
    print(message, flush=True)


def row_normalize(counts: torch.Tensor) -> torch.Tensor:
    counts = counts.to(torch.float64)
    totals = counts.sum(dim=1, keepdim=True)
    return torch.where(totals > 0, counts / totals.clamp_min(1.0), torch.zeros_like(counts))


def save_matrix(matrix: torch.Tensor, labels: list[int], path: Path, index_name: str) -> None:
    df = pd.DataFrame(matrix.cpu().numpy(), index=labels, columns=labels)
    df.index.name = index_name
    df.to_csv(path)
    log(f"[write] {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute coalesced GCN-undirected Pokec connectivity.")
    parser.add_argument(
        "--gcn-data-dir",
        type=Path,
        default=Path("../260613_2_GCN/data/pokec_temporal"),
        help="Directory containing edge_index_undirected_self_loop.pt and year tensors.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("pokec_temporal_outputs"),
        help="Output directory for corrected undirected CSVs.",
    )
    parser.add_argument("--chunk-edges", type=int, default=5_000_000)
    parser.add_argument(
        "--overwrite-undirected",
        action="store_true",
        help="Also overwrite pokec_edge_counts_by_year_undirected.csv and probability file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    edge_path = args.gcn_data_dir / "edge_index_undirected_self_loop.pt"
    year_idx_path = args.gcn_data_dir / "year_idx.pt"
    year_raw_path = args.gcn_data_dir / "year_raw.pt"
    for path in (edge_path, year_idx_path, year_raw_path):
        if not path.exists():
            raise FileNotFoundError(path)

    log(f"[load] {edge_path}")
    edge_index = torch.load(edge_path, map_location="cpu")
    year_idx = torch.load(year_idx_path, map_location="cpu").long()
    year_raw = torch.load(year_raw_path, map_location="cpu").long()

    valid_years = torch.unique(year_raw[year_raw >= 0], sorted=True)
    labels = [int(x) for x in valid_years.tolist()]
    num_years = len(labels)
    counts = torch.zeros(num_years * num_years, dtype=torch.long)

    total_edges = int(edge_index.size(1))
    kept_edges = 0
    skipped_self = 0
    skipped_invalid_year = 0

    for start in range(0, total_edges, args.chunk_edges):
        end = min(start + args.chunk_edges, total_edges)
        src = edge_index[0, start:end].long()
        dst = edge_index[1, start:end].long()
        non_self = src != dst
        skipped_self += int((~non_self).sum().item())
        src = src[non_self]
        dst = dst[non_self]
        src_year = year_idx[src]
        dst_year = year_idx[dst]
        valid = (src_year >= 0) & (dst_year >= 0)
        skipped_invalid_year += int((~valid).sum().item())
        pair = src_year[valid] * num_years + dst_year[valid]
        counts += torch.bincount(pair, minlength=num_years * num_years).cpu()
        kept_edges += int(valid.sum().item())
        log(f"[chunk] {end:,}/{total_edges:,} edges processed")

    matrix = counts.view(num_years, num_years)
    probs = row_normalize(matrix)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    save_matrix(
        matrix,
        labels,
        args.out_dir / "pokec_edge_counts_by_year_gcn_undirected.csv",
        "registration_year",
    )
    save_matrix(
        probs,
        labels,
        args.out_dir / "pokec_edge_probs_by_year_gcn_undirected.csv",
        "registration_year",
    )
    if args.overwrite_undirected:
        save_matrix(
            matrix,
            labels,
            args.out_dir / "pokec_edge_counts_by_year_undirected.csv",
            "registration_year",
        )
        save_matrix(
            probs,
            labels,
            args.out_dir / "pokec_edge_probs_by_year_undirected.csv",
            "registration_year",
        )

    log(
        "[summary] "
        f"kept_nonself_message_edges={kept_edges:,}, "
        f"self_loops_excluded={skipped_self:,}, invalid_year_edges={skipped_invalid_year:,}, "
        f"matrix_sum={int(matrix.sum().item()):,}"
    )


if __name__ == "__main__":
    main()
