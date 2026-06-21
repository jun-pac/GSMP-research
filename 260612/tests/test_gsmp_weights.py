import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
LARGE_GRAPH = ROOT / "upstream" / "tunedGNN" / "large_graph"
sys.path.insert(0, str(LARGE_GRAPH))

from gsmp import compute_gsmp_edge_weight
from lg_model import MPNNs


def assert_close(actual, expected):
    if not torch.allclose(actual, expected, atol=1e-6, rtol=0.0):
        raise AssertionError(f"actual={actual.tolist()} expected={expected.tolist()}")


def test_known_scale_preserve():
    edge_index = torch.tensor(
        [
            [0, 1, 2],
            [3, 3, 3],
        ],
        dtype=torch.long,
    )
    node_year = torch.tensor([2015, 2015, 2016, 2017], dtype=torch.long)
    weight = compute_gsmp_edge_weight(edge_index, node_year, num_nodes=4, mode="scale_preserve")
    expected = torch.tensor([0.75, 0.75, 1.50], dtype=torch.float32)
    assert_close(weight, expected)


def test_known_strict():
    edge_index = torch.tensor(
        [
            [0, 1, 2],
            [3, 3, 3],
        ],
        dtype=torch.long,
    )
    node_year = torch.tensor([2015, 2015, 2016, 2017], dtype=torch.long)
    weight = compute_gsmp_edge_weight(edge_index, node_year, num_nodes=4, mode="strict")
    expected = torch.tensor([0.25, 0.25, 0.50], dtype=torch.float32)
    assert_close(weight, expected)


def test_cache_roundtrip():
    edge_index = torch.tensor(
        [
            [0, 1, 2],
            [3, 3, 3],
        ],
        dtype=torch.long,
    )
    node_year = torch.tensor([2015, 2015, 2016, 2017], dtype=torch.long)
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "tiny_gsmp.pt"
        first = compute_gsmp_edge_weight(
            edge_index,
            node_year,
            num_nodes=4,
            mode="scale_preserve",
            cache_path=cache_path,
        )
        second = compute_gsmp_edge_weight(
            edge_index,
            node_year,
            num_nodes=4,
            mode="scale_preserve",
            cache_path=cache_path,
        )
    assert_close(first, second)


def test_first_layer_only_routes_weights_once():
    model = MPNNs(
        in_channels=3,
        hidden_channels=4,
        out_channels=2,
        local_layers=3,
        use_gsmp=True,
        gsmp_apply="first_layer",
    )
    edge_weight = torch.ones(5)
    if model._edge_weight_for_layer(0, edge_weight) is not edge_weight:
        raise AssertionError("first layer should receive GSMP edge weights")
    if model._edge_weight_for_layer(1, edge_weight) is not None:
        raise AssertionError("later layers should use vanilla GCN weights")
    if model._edge_weight_for_layer(2, edge_weight) is not None:
        raise AssertionError("later layers should use vanilla GCN weights")


def test_first_layer_only_restores_later_gcn_normalization():
    model = MPNNs(
        in_channels=3,
        hidden_channels=4,
        out_channels=2,
        local_layers=3,
        use_gsmp=True,
        gcn_gsmp_mode="post_norm_message_scale",
        gsmp_apply="first_layer",
    )
    if model.local_convs[0].normalize:
        raise AssertionError("pre-normalized GSMP first layer should not re-normalize")
    if not model.local_convs[1].normalize or not model.local_convs[2].normalize:
        raise AssertionError("later first-layer-only GSMP layers should be vanilla GCN")


if __name__ == "__main__":
    test_known_scale_preserve()
    test_known_strict()
    test_cache_roundtrip()
    test_first_layer_only_routes_weights_once()
    test_first_layer_only_restores_later_gcn_normalization()
    print("GSMP tiny-graph tests passed")
