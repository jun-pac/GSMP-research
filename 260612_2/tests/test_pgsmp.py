import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
LARGE_GRAPH = ROOT / "upstream" / "tunedGNN" / "large_graph"
sys.path.insert(0, str(LARGE_GRAPH))

from pgsmp import apply_pgsmp_preprocess, compute_pgsmp_edge_weight


def assert_close(actual, expected):
    if not torch.allclose(actual, expected, atol=1e-5, rtol=0.0):
        raise AssertionError(f"actual={actual.tolist()} expected={expected.tolist()}")


def tiny_graph():
    edge_index = torch.tensor(
        [
            [0, 1, 2],
            [3, 3, 3],
        ],
        dtype=torch.long,
    )
    node_year = torch.tensor([2015, 2015, 2016, 2017], dtype=torch.long)
    x = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [0.0, 10.0],
            [100.0, 100.0],
        ],
        dtype=torch.float32,
    )
    return edge_index, node_year, x


def test_strict_observed_weights():
    edge_index, node_year, _ = tiny_graph()
    weight = compute_pgsmp_edge_weight(
        edge_index,
        node_year=node_year,
        num_nodes=4,
        mode="strict_observed",
    )
    expected = torch.tensor([0.25, 0.25, 0.50], dtype=torch.float32)
    assert_close(weight, expected)


def test_scale_preserve_weights():
    edge_index, node_year, _ = tiny_graph()
    weight = compute_pgsmp_edge_weight(
        edge_index,
        node_year=node_year,
        num_nodes=4,
        mode="scale_preserve",
    )
    expected = torch.tensor([0.75, 0.75, 1.50], dtype=torch.float32)
    assert_close(weight, expected)


def test_apply_alpha_one_neighbor_only():
    edge_index, node_year, x = tiny_graph()
    x_pg = apply_pgsmp_preprocess(
        x,
        edge_index,
        node_year=node_year,
        num_nodes=4,
        alpha=1.0,
        depth=1,
        norm="strict_observed",
        self_mode="neighbor_only",
        chunk_size=2,
    )
    assert_close(x_pg[3], torch.tensor([1.0, 5.0]))


def test_apply_alpha_half_neighbor_only():
    edge_index, node_year, x = tiny_graph()
    x_pg = apply_pgsmp_preprocess(
        x,
        edge_index,
        node_year=node_year,
        num_nodes=4,
        alpha=0.5,
        depth=1,
        norm="strict_observed",
        self_mode="neighbor_only",
        chunk_size=2,
    )
    assert_close(x_pg[3], torch.tensor([50.5, 52.5]))


def test_include_self_adds_one_self_edge():
    edge_index, node_year, x = tiny_graph()
    x_pg = apply_pgsmp_preprocess(
        x,
        edge_index,
        node_year=node_year,
        num_nodes=4,
        alpha=1.0,
        depth=1,
        norm="strict_observed",
        self_mode="include_self",
        chunk_size=2,
    )
    assert_close(x_pg[3], torch.tensor([34.0, 110.0 / 3.0]))


def test_feature_cache_roundtrip():
    edge_index, node_year, x = tiny_graph()
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "xpg.pt"
        first, first_info = apply_pgsmp_preprocess(
            x,
            edge_index,
            node_year=node_year,
            num_nodes=4,
            alpha=0.5,
            depth=1,
            norm="strict_observed",
            self_mode="neighbor_only",
            cache_path=cache_path,
            return_info=True,
        )
        second, second_info = apply_pgsmp_preprocess(
            x,
            edge_index,
            node_year=node_year,
            num_nodes=4,
            alpha=0.5,
            depth=1,
            norm="strict_observed",
            self_mode="neighbor_only",
            cache_path=cache_path,
            return_info=True,
        )
    assert_close(first, second)
    if first_info["loaded_from_cache"]:
        raise AssertionError("first run should compute features")
    if not second_info["loaded_from_cache"]:
        raise AssertionError("second run should reuse feature cache")


if __name__ == "__main__":
    test_strict_observed_weights()
    test_scale_preserve_weights()
    test_apply_alpha_one_neighbor_only()
    test_apply_alpha_half_neighbor_only()
    test_include_self_adds_one_self_edge()
    test_feature_cache_roundtrip()
    print("P-GSMP tiny-graph tests passed")
