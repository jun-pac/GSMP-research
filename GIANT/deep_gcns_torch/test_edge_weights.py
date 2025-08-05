#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import dgl
import numpy as np
from GSMP_ensemble import compute_local_edge_weights, compute_local_edge_weights_parallel, parallel_compute

def test_edge_weight_computation():
    """Test the local time distribution edge weight computation"""
    
    # Create a simple test graph
    num_nodes = 10
    src = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
    dst = torch.tensor([1, 2, 0, 3, 0, 4, 1, 5, 2, 6, 3, 7, 4, 8, 5, 9, 6, 7, 7, 8])
    
    # Create paper years (some years are more frequent)
    paper_year = torch.tensor([2015, 2015, 2016, 2016, 2015, 2017, 2016, 2018, 2017, 2019, 2018, 2020, 2019, 2021, 2020, 2022, 2021, 2022, 2022, 2023])
    
    # Create DGL graph
    graph = dgl.graph((src, dst), num_nodes=num_nodes)
    graph.ndata['year'] = paper_year
    
    print("Test graph:")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f"Paper years: {paper_year}")
    
    # Test sequential computation
    print("\nTesting sequential computation...")
    weights_seq = compute_local_edge_weights(graph, paper_year)
    print(f"Sequential weights: {weights_seq}")
    print(f"Weight stats - min: {weights_seq.min():.4f}, max: {weights_seq.max():.4f}, mean: {weights_seq.mean():.4f}")
    
    # Test parallel computation using the exact same approach as preprocess_GSMP_papers100m.py
    print("\nTesting parallel computation (matching preprocess_GSMP_papers100m.py)...")
    row = src.cpu().numpy()
    col = dst.cpu().numpy()
    paper_year_np = paper_year.cpu().numpy()
    edge_weight = torch.zeros(len(row))
    
    parallel_compute(edge_weight, paper_year_np, row, col, num_nodes, num_workers=2)
    weights_par = edge_weight.float()
    
    print(f"Parallel weights: {weights_par}")
    print(f"Weight stats - min: {weights_par.min():.4f}, max: {weights_par.max():.4f}, mean: {weights_par.mean():.4f}")
    
    # Test the wrapper function
    print("\nTesting wrapper function...")
    weights_wrapper = compute_local_edge_weights_parallel(graph, paper_year, num_workers=2)
    print(f"Wrapper weights: {weights_wrapper}")
    print(f"Weight stats - min: {weights_wrapper.min():.4f}, max: {weights_wrapper.max():.4f}, mean: {weights_wrapper.mean():.4f}")
    
    # Check if results are similar (allowing for small numerical differences)
    diff1 = torch.abs(weights_seq - weights_par)
    diff2 = torch.abs(weights_par - weights_wrapper)
    print(f"\nDifference between sequential and parallel: max={diff1.max():.6f}, mean={diff1.mean():.6f}")
    print(f"Difference between parallel and wrapper: max={diff2.max():.6f}, mean={diff2.mean():.6f}")
    
    if diff1.max() < 1e-6 and diff2.max() < 1e-6:
        print("✓ All implementations produce consistent results!")
    else:
        print("✗ Implementations produce different results!")
    
    # Test with a more complex case - nodes with different year distributions
    print("\n" + "="*50)
    print("Testing with more complex year distribution...")
    
    # Create a graph where some nodes have many neighbors from the same year
    src2 = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
    dst2 = torch.tensor([1, 2, 3, 4, 0, 5, 6, 0, 7, 8, 0, 9, 0, 5, 1, 4, 1, 7, 2, 6])
    
    # Node 0 has many neighbors from year 2015, others have diverse years
    paper_year2 = torch.tensor([2015, 2015, 2015, 2015, 2015, 2016, 2017, 2018, 2019, 2020, 2015, 2021, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
    
    graph2 = dgl.graph((src2, dst2), num_nodes=10)
    graph2.ndata['year'] = paper_year2
    
    print(f"Complex graph - Node 0 has many 2015 neighbors")
    print(f"Paper years: {paper_year2}")
    
    # Test with parallel computation
    row2 = src2.cpu().numpy()
    col2 = dst2.cpu().numpy()
    paper_year2_np = paper_year2.cpu().numpy()
    edge_weight2 = torch.zeros(len(row2))
    
    parallel_compute(edge_weight2, paper_year2_np, row2, col2, 10, num_workers=2)
    weights2 = edge_weight2.float()
    
    print(f"Edge weights: {weights2}")
    
    # Check that edges from node 0 to 2015 nodes have lower weights (more frequent)
    node0_edges = torch.where(src2 == 0)[0]
    node0_weights = weights2[node0_edges]
    node0_dst_years = paper_year2[dst2[node0_edges]]
    
    print(f"Node 0 edge weights: {node0_weights}")
    print(f"Node 0 destination years: {node0_dst_years}")
    
    # Edges to 2015 should have lower weights than edges to other years
    edges_to_2015 = node0_dst_years == 2015
    edges_to_others = node0_dst_years != 2015
    
    if torch.any(edges_to_2015) and torch.any(edges_to_others):
        avg_weight_2015 = node0_weights[edges_to_2015].mean()
        avg_weight_others = node0_weights[edges_to_others].mean()
        print(f"Average weight to 2015 nodes: {avg_weight_2015:.4f}")
        print(f"Average weight to other years: {avg_weight_others:.4f}")
        
        if avg_weight_2015 < avg_weight_others:
            print("✓ Local time distribution working correctly!")
        else:
            print("✗ Local time distribution not working as expected!")

def test_exact_match_with_original():
    """Test that our implementation exactly matches the original approach"""
    print("\n" + "="*60)
    print("Testing exact match with original preprocess_GSMP_papers100m.py approach")
    print("="*60)
    
    # Create a simple test case that we can manually verify
    num_nodes = 5
    src = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 4])
    dst = torch.tensor([1, 2, 3, 0, 4, 0, 4, 0, 4, 1])
    
    # Node 0 has neighbors from years: [2015, 2015, 2016] (2 from 2015, 1 from 2016)
    # Node 1 has neighbors from years: [2015, 2017] (1 from each)
    # Node 2 has neighbors from years: [2015, 2017] (1 from each)
    # Node 3 has neighbors from years: [2015, 2017] (1 from each)
    # Node 4 has neighbors from years: [2017, 2017, 2017, 2017] (4 from 2017)
    paper_year = torch.tensor([2015, 2015, 2016, 2015, 2017, 2015, 2017, 2015, 2017, 2017])
    
    print(f"Test case:")
    print(f"Node 0 neighbors: years {paper_year[dst[src == 0]]}")
    print(f"Node 1 neighbors: years {paper_year[dst[src == 1]]}")
    print(f"Node 2 neighbors: years {paper_year[dst[src == 2]]}")
    print(f"Node 3 neighbors: years {paper_year[dst[src == 3]]}")
    print(f"Node 4 neighbors: years {paper_year[dst[src == 4]]}")
    
    # Manual calculation for node 0:
    # Neighbors: [2015, 2015, 2016]
    # Year counts: 2015 -> 2, 2016 -> 1
    # Raw weights: 1/2, 1/2, 1/1 = [0.5, 0.5, 1.0]
    # Mean: 0.667
    # Normalized: [0.75, 0.75, 1.5]
    
    # Manual calculation for node 4:
    # Neighbors: [2017, 2017, 2017, 2017]
    # Year counts: 2017 -> 4
    # Raw weights: 1/4, 1/4, 1/4, 1/4 = [0.25, 0.25, 0.25, 0.25]
    # Mean: 0.25
    # Normalized: [1.0, 1.0, 1.0, 1.0]
    
    print(f"\nExpected results:")
    print(f"Node 0 edge weights: [0.75, 0.75, 1.5]")
    print(f"Node 4 edge weights: [1.0, 1.0, 1.0, 1.0]")
    
    # Test our implementation
    row = src.cpu().numpy()
    col = dst.cpu().numpy()
    paper_year_np = paper_year.cpu().numpy()
    edge_weight = torch.zeros(len(row))
    
    parallel_compute(edge_weight, paper_year_np, row, col, num_nodes, num_workers=1)
    weights = edge_weight.float()
    
    print(f"\nActual results:")
    print(f"All edge weights: {weights}")
    
    # Check node 0 edges
    node0_edges = torch.where(src == 0)[0]
    node0_weights = weights[node0_edges]
    print(f"Node 0 edge weights: {node0_weights}")
    
    # Check node 4 edges
    node4_edges = torch.where(src == 4)[0]
    node4_weights = weights[node4_edges]
    print(f"Node 4 edge weights: {node4_weights}")
    
    # Verify the results
    expected_node0 = torch.tensor([0.75, 0.75, 1.5])
    expected_node4 = torch.tensor([1.0, 1.0, 1.0, 1.0])
    
    diff_node0 = torch.abs(node0_weights - expected_node0)
    diff_node4 = torch.abs(node4_weights - expected_node4)
    
    print(f"\nVerification:")
    print(f"Node 0 difference: max={diff_node0.max():.6f}")
    print(f"Node 4 difference: max={diff_node4.max():.6f}")
    
    if diff_node0.max() < 1e-6 and diff_node4.max() < 1e-6:
        print("✓ Implementation exactly matches expected results!")
    else:
        print("✗ Implementation does not match expected results!")

if __name__ == "__main__":
    test_edge_weight_computation()
    test_exact_match_with_original() 