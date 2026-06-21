from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsmp_utils import apply_pgsmp_preprocess, compute_year_balanced_edge_weight, make_cache_name  # noqa: E402
from linear_revgat_gsmp_experiment import LinearRevGAT  # noqa: E402


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual, expected, atol=1e-6, rtol=0.0):
        raise AssertionError(f"actual={actual.tolist()} expected={expected.tolist()}")


def tiny_graph():
    edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]], dtype=torch.long)
    node_year = torch.tensor([2015, 2015, 2016, 2017], dtype=torch.long)
    return edge_index, node_year


def test_scale_preserve_weights():
    edge_index, node_year = tiny_graph()
    weight = compute_year_balanced_edge_weight(edge_index, node_year, num_nodes=4, mode="scale_preserve")
    expected = torch.tensor([0.75, 0.75, 1.50], dtype=torch.float32)
    assert_close(weight, expected)


def test_strict_observed_weights():
    edge_index, node_year = tiny_graph()
    weight = compute_year_balanced_edge_weight(edge_index, node_year, num_nodes=4, mode="strict_observed")
    expected = torch.tensor([0.25, 0.25, 0.50], dtype=torch.float32)
    assert_close(weight, expected)


def test_gsmp_cache_roundtrip():
    edge_index, node_year = tiny_graph()
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Path(tmpdir) / "weights.pt"
        first = compute_year_balanced_edge_weight(edge_index, node_year, 4, "scale_preserve", cache_path=cache)
        second = compute_year_balanced_edge_weight(edge_index, node_year, 4, "scale_preserve", cache_path=cache)
    assert_close(first, second)


def test_pgsmp_strict_observed_depth1():
    edge_index, node_year = tiny_graph()
    x = torch.tensor([[1.0], [3.0], [10.0], [0.0]])
    out = apply_pgsmp_preprocess(
        x,
        edge_index,
        node_year,
        num_nodes=4,
        alpha=1.0,
        depth=1,
        norm="strict_observed",
        self_mode="neighbor_only",
    )
    expected = torch.tensor([[0.0], [0.0], [0.0], [6.0]])
    assert_close(out, expected)


def test_pgsmp_cache_roundtrip():
    edge_index, node_year = tiny_graph()
    x = torch.tensor([[1.0], [3.0], [10.0], [0.0]])
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Path(tmpdir) / "pgsmp.pt"
        first = apply_pgsmp_preprocess(x, edge_index, node_year, 4, cache_path=cache)
        second = apply_pgsmp_preprocess(x, edge_index, node_year, 4, cache_path=cache)
    assert_close(first, second)


def test_cache_name_stays_filesystem_safe_and_unique():
    metadata = {
        "dataset": "ogbn-arxiv",
        "direction": "src_to_dst",
        "undirected": True,
        "self_loop": True,
        "norm": "strict_observed",
        "alpha": 0.5,
        "depth": 1,
        "self_mode": "neighbor_only",
        "xshape": "169343x1024",
        "feature": "ogbn-arxiv-tape_all-roberta-large-v1-" + ("abcdef" * 20),
        "num_edges": 2484941,
        "num_nodes": 169343,
    }
    first = make_cache_name("pgsmp", metadata)
    second_metadata = dict(metadata)
    second_metadata["feature"] = metadata["feature"] + "-different"
    second = make_cache_name("pgsmp", second_metadata)
    if len(first) > 180:
        raise AssertionError(f"cache name too long: {len(first)} {first}")
    if first == second:
        raise AssertionError("cache names must change when metadata changes")


def test_first_layer_only_routes_weights_only_to_layer_zero():
    model = LinearRevGAT(
        in_feats=4,
        n_classes=3,
        n_hidden=5,
        n_layers=3,
        n_heads=2,
        activation=torch.relu,
    )
    edge_weight = torch.ones(7)
    if model._edge_weight_for_layer(0, edge_weight, gsmp_layer=0) is not edge_weight:
        raise AssertionError("layer 0 must receive GSMP weights")
    if model._edge_weight_for_layer(1, edge_weight, gsmp_layer=0) is not None:
        raise AssertionError("middle layers must not receive GSMP weights")
    if model._edge_weight_for_layer(2, edge_weight, gsmp_layer=0) is not None:
        raise AssertionError("last layer must not receive GSMP weights")


if __name__ == "__main__":
    test_scale_preserve_weights()
    test_strict_observed_weights()
    test_gsmp_cache_roundtrip()
    test_pgsmp_strict_observed_depth1()
    test_pgsmp_cache_roundtrip()
    test_cache_name_stays_filesystem_safe_and_unique()
    test_first_layer_only_routes_weights_only_to_layer_zero()
    print("GSMP utility tests passed")
