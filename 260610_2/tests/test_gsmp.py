#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TAPE = ROOT / "upstream" / "TAPE"
sys.path.insert(0, str(TAPE))

from core.gsmp import build_linear_message_weight, compute_gsmp_edge_weight  # noqa: E402


def assert_close(actual, expected, name):
    if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-6):
        raise AssertionError(f"{name}: got {actual.tolist()}, expected {expected.tolist()}")


def main() -> None:
    # Edges are src -> dst.  Target 3 receives two year-2000 neighbors and
    # one year-2001 neighbor.  Target 4 receives two year-2001 neighbors.
    edge_index = torch.tensor(
        [
            [0, 1, 2, 2, 5],
            [3, 3, 3, 4, 4],
        ],
        dtype=torch.long,
    )
    node_year = torch.tensor([2000, 2000, 2001, 2002, 2003, 2001], dtype=torch.long)

    strict = compute_gsmp_edge_weight(edge_index, node_year, num_nodes=6, mode="strict")
    expected_strict = torch.tensor([0.25, 0.25, 0.5, 0.5, 0.5])
    assert_close(strict, expected_strict, "strict")

    scale = compute_gsmp_edge_weight(edge_index, node_year, num_nodes=6, mode="scale_preserve")
    expected_scale = torch.tensor([0.75, 0.75, 1.5, 1.0, 1.0])
    assert_close(scale, expected_scale, "scale_preserve")

    mean, _ = build_linear_message_weight(edge_index, node_year, 6, use_gsmp=False)
    expected_mean = torch.tensor([1 / 3, 1 / 3, 1 / 3, 1 / 2, 1 / 2], dtype=torch.float32)
    assert_close(mean, expected_mean, "mean")

    scale_final, _ = build_linear_message_weight(edge_index, node_year, 6, use_gsmp=True, mode="scale_preserve")
    expected_scale_final = expected_mean * expected_scale
    assert_close(scale_final, expected_scale_final, "scale_final")

    print("GSMP tiny tests passed")


if __name__ == "__main__":
    main()
