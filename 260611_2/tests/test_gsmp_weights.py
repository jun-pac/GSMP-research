#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util

import torch


ROOT = Path(__file__).resolve().parents[1]
GSMP_FILE = ROOT / "upstream" / "GLEM" / "src" / "models" / "GNNs" / "LinearRevGAT" / "gsmp.py"
spec = importlib.util.spec_from_file_location("glem_linear_revgat_gsmp", GSMP_FILE)
gsmp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gsmp)

build_linear_message_weight = gsmp.build_linear_message_weight
compute_gsmp_edge_weight = gsmp.compute_gsmp_edge_weight


def assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-6):
        raise AssertionError(f"{name}: got {actual.tolist()}, expected {expected.tolist()}")


def main() -> None:
    # Edges are directed message edges src -> dst.
    # Target 3 receives two year-2000 neighbors and one year-2001 neighbor.
    # Target 4 receives two year-2001 neighbors.
    edge_index = torch.tensor(
        [
            [0, 1, 2, 2, 5],
            [3, 3, 3, 4, 4],
        ],
        dtype=torch.long,
    )
    node_year = torch.tensor([2000, 2000, 2001, 2002, 2003, 2001], dtype=torch.long)

    strict = compute_gsmp_edge_weight(edge_index, node_year, num_nodes=6, mode="strict")
    assert_close(strict, torch.tensor([0.25, 0.25, 0.5, 0.5, 0.5]), "strict")

    scale = compute_gsmp_edge_weight(edge_index, node_year, num_nodes=6, mode="scale_preserve")
    assert_close(scale, torch.tensor([0.75, 0.75, 1.5, 1.0, 1.0]), "scale_preserve")

    mean, _ = build_linear_message_weight(edge_index, node_year, 6, use_gsmp=False)
    assert_close(mean, torch.tensor([1 / 3, 1 / 3, 1 / 3, 1 / 2, 1 / 2]), "mean")

    final_scale, _ = build_linear_message_weight(
        edge_index,
        node_year,
        6,
        use_gsmp=True,
        mode="scale_preserve",
    )
    assert_close(final_scale, mean * scale, "scale_preserve_final_linear_weight")

    final_strict, _ = build_linear_message_weight(
        edge_index,
        node_year,
        6,
        use_gsmp=True,
        mode="strict",
    )
    assert_close(final_strict, strict, "strict_final_linear_weight")

    print("GLEM GSMP tiny tests passed")


if __name__ == "__main__":
    main()
