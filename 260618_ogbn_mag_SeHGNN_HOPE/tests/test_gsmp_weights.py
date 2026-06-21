#!/usr/bin/env python
import sys
from pathlib import Path

import dgl
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HOPE"))

from utils import compute_gsmp_edge_weights, compute_smp_edge_weights, should_use_gsmp_update  # noqa: E402


def assert_close(actual, expected, name):
    if not torch.allclose(actual, expected, atol=1e-6, rtol=0.0):
        raise AssertionError(f"{name}: actual={actual.tolist()} expected={expected.tolist()}")


def test_nonempty_weights_sum_to_one():
    graph = dgl.heterograph(
        {("A", "A-P", "P"): (torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0]))},
        num_nodes_dict={"A": 3, "P": 1},
    )
    src_time = torch.tensor([2000, 2000, 2001], dtype=torch.long)
    weights = compute_gsmp_edge_weights(graph, "A-P", src_time, normalizer="nonempty")
    expected = torch.tensor([0.25, 0.25, 0.50], dtype=torch.float32)
    assert_close(weights, expected, "A-P nonempty GSMP weights")

    _, dst = graph.edges(etype="A-P", order="eid")
    totals = torch.zeros(graph.num_nodes("P"))
    totals.index_add_(0, dst.long(), weights)
    assert_close(totals, torch.ones_like(totals), "destination weight sums")


def test_global_weights_use_all_source_bins():
    graph = dgl.heterograph(
        {("F", "F-P", "P"): (torch.tensor([0, 1]), torch.tensor([0, 0]))},
        num_nodes_dict={"F": 3, "P": 1},
    )
    src_time = torch.tensor([1999, 2000, 2001], dtype=torch.long)
    weights = compute_gsmp_edge_weights(graph, "F-P", src_time, normalizer="global")
    expected = torch.tensor([1 / 3, 1 / 3], dtype=torch.float32)
    assert_close(weights, expected, "F-P global GSMP weights")


def test_smp_weights_use_target_normalized_corrected_rule():
    graph = dgl.heterograph(
        {("P", "P-P", "P"): (
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([4, 4, 4, 4]),
        )},
        num_nodes_dict={"P": 5},
    )
    paper_time = torch.tensor([2018, 2017, 2019, 2016, 2018], dtype=torch.long)
    weights = compute_smp_edge_weights(graph, "P-P", paper_time, paper_time)
    expected = torch.tensor([1 / 3, 1 / 6, 1 / 6, 1 / 3], dtype=torch.float32)
    assert_close(weights, expected, "P-P target-normalized SMP weights")

    _, dst = graph.edges(etype="P-P", order="eid")
    totals = torch.zeros(graph.num_nodes("P"))
    totals.index_add_(0, dst.long(), weights)
    expected_totals = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    assert_close(totals, expected_totals, "SMP destination weight sums")


def test_paper_stack_scope_includes_second_hop_into_paper():
    assert should_use_gsmp_update(
        True, "paper-stack", 1, "P", "P", "P",
        is_label_propagation=False, gsmp_apply_label_prop=False)
    assert should_use_gsmp_update(
        True, "paper-stack", 2, "P", "P", "P",
        is_label_propagation=False, gsmp_apply_label_prop=False)
    assert not should_use_gsmp_update(
        True, "paper-stack", 2, "P", "A", "P",
        is_label_propagation=False, gsmp_apply_label_prop=False)
    assert not should_use_gsmp_update(
        True, "paper-stack", 2, "P", "P", "A",
        is_label_propagation=False, gsmp_apply_label_prop=False)
    assert not should_use_gsmp_update(
        True, "paper-stack", 2, "P", "P", "P",
        is_label_propagation=True, gsmp_apply_label_prop=False)
    assert should_use_gsmp_update(
        True, "first-hop", 1, "P", "P", "P",
        is_label_propagation=False, gsmp_apply_label_prop=False)
    assert not should_use_gsmp_update(
        True, "first-hop", 2, "P", "P", "P",
        is_label_propagation=False, gsmp_apply_label_prop=False)


if __name__ == "__main__":
    test_nonempty_weights_sum_to_one()
    test_global_weights_use_all_source_bins()
    test_smp_weights_use_target_normalized_corrected_rule()
    test_paper_stack_scope_includes_second_hop_into_paper()
    print("GSMP/SMP edge-weight tests passed")
