"""
Temporal message passing methods: SMP, UMP, GSMP
"""

import torch
import logging
from torch_geometric.data import HeteroData
from typing import Dict, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


def compute_smp_edge_weights(
    data: HeteroData,
    timestamp_dict: Dict[str, torch.Tensor],
    t_min: float,
    t_max: float
) -> HeteroData:
    """
    Compute Symmetrized Message Passing (SMP) edge weights.
    
    For each directed edge u -> v:
        delta = abs(time(u) - time(v))
        boundary = min(t_max - time(v), time(v) - t_min)
        
        single condition: delta == 0 OR delta > boundary
        double condition: otherwise
        
        weight = 2.0 if single, 1.0 if double
    
    Args:
        data: HeteroData object with edge_index tensors
        timestamp_dict: Dict mapping node_type -> timestamp tensor
        t_min: Global minimum timestamp
        t_max: Global maximum timestamp
    
    Returns:
        Modified data with edge_weight attributes added
    """
    logger.info("Computing SMP edge weights...")
    
    for edge_type in data.edge_types:
        src_type, rel_type, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        
        if edge_index is None or edge_index.shape[1] == 0:
            continue
        
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        # Get timestamps
        src_times = timestamp_dict[src_type][src_nodes].float()
        dst_times = timestamp_dict[dst_type][dst_nodes].float()
        
        # Compute delta
        delta = torch.abs(src_times - dst_times)
        
        # Compute boundary
        dst_boundary = torch.minimum(
            t_max - dst_times,
            dst_times - t_min
        )
        
        # Single condition: delta == 0 OR delta > boundary
        single_mask = (delta == 0) | (delta > dst_boundary)
        
        # Weights: 2.0 for single, 1.0 for double
        edge_weight = torch.where(single_mask, torch.tensor(2.0), torch.tensor(1.0))
        
        # Add to data
        data[edge_type].edge_weight = edge_weight.to(edge_index.device)
        
        # Sanity checks
        assert edge_weight.shape[0] == edge_index.shape[1], \
            f"Edge weight shape mismatch for {edge_type}"
        assert torch.isfinite(edge_weight).all(), \
            f"Non-finite edge weights in {edge_type}"
        assert (edge_weight > 0).all(), \
            f"Non-positive edge weights in {edge_type}"
        
        logger.info(f"  {edge_type}: weight min={edge_weight.min():.4f}, " \
                   f"max={edge_weight.max():.4f}, mean={edge_weight.mean():.4f}")
    
    return data


def apply_ump_edge_filter(
    data: HeteroData,
    timestamp_dict: Dict[str, torch.Tensor]
) -> HeteroData:
    """
    Apply Unsymmetrized Message Passing (UMP) edge filtering.
    
    Keep edges u -> v only if time(u) <= time(v).
    Remove edges where time(u) > time(v).
    
    Args:
        data: HeteroData object with edge_index tensors
        timestamp_dict: Dict mapping node_type -> timestamp tensor
    
    Returns:
        Modified data with filtered edges (future edges removed)
    """
    logger.info("Applying UMP edge filtering...")
    
    for edge_type in data.edge_types:
        src_type, rel_type, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        
        if edge_index is None or edge_index.shape[1] == 0:
            continue
        
        original_num_edges = edge_index.shape[1]
        
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        # Get timestamps
        src_times = timestamp_dict[src_type][src_nodes]
        dst_times = timestamp_dict[dst_type][dst_nodes]
        
        # Keep edges where src_time <= dst_time
        keep_mask = src_times <= dst_times
        
        # Filter edge_index
        filtered_edge_index = edge_index[:, keep_mask]
        
        # Update data
        data[edge_type].edge_index = filtered_edge_index
        
        # Add unit edge weights (all ones)
        num_kept = filtered_edge_index.shape[1]
        edge_weight = torch.ones(num_kept, dtype=torch.float32)
        data[edge_type].edge_weight = edge_weight.to(filtered_edge_index.device)
        
        num_removed = original_num_edges - num_kept
        pct_removed = 100.0 * num_removed / original_num_edges if original_num_edges > 0 else 0
        
        logger.info(f"  {edge_type}: {original_num_edges} -> {num_kept} edges " \
                   f"({num_removed} removed, {pct_removed:.1f}%)")
        
        # Sanity check
        if num_kept > 0:
            src_times_kept = timestamp_dict[src_type][filtered_edge_index[0]]
            dst_times_kept = timestamp_dict[dst_type][filtered_edge_index[1]]
            assert (src_times_kept <= dst_times_kept).all(), \
                f"UMP constraint violated for {edge_type}"
    
    return data


def compute_gsmp_edge_weights(
    data: HeteroData,
    timestamp_dict: Dict[str, torch.Tensor]
) -> HeteroData:
    """
    Compute General Symmetrized Message Passing (GSMP) edge weights.
    
    For each target node v and edge u -> v:
        Count how many incoming neighbors have the same timestamp as u:
            count_v[time(u)] = |{u' : (u', v) in E and time(u') == time(u)}|
        
        Edge weight: w_{u->v} = 1 / count_v[time(u)]
    
    This makes each timestamp group contribute equally to the aggregation.
    
    Args:
        data: HeteroData object with edge_index tensors
        timestamp_dict: Dict mapping node_type -> timestamp tensor
    
    Returns:
        Modified data with edge_weight attributes added
    """
    logger.info("Computing GSMP edge weights...")
    
    for edge_type in data.edge_types:
        src_type, rel_type, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        
        if edge_index is None or edge_index.shape[1] == 0:
            continue
        
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        # Get timestamps
        src_times = timestamp_dict[src_type][src_nodes]
        dst_times = timestamp_dict[dst_type][dst_nodes]
        
        # For each (dst_node, src_time) pair, count how many edges have that combination
        # This is count_v[tau]
        timestamp_counts = defaultdict(int)
        for src_time, dst_node in zip(src_times.tolist(), dst_nodes.tolist()):
            src_time = int(src_time)
            dst_node = int(dst_node)
            key = (dst_node, src_time)
            timestamp_counts[key] += 1
        
        # Compute edge weights: 1 / count_v[time(u)]
        edge_weight = torch.zeros(len(src_nodes), dtype=torch.float32)
        for i, (src_time, dst_node) in enumerate(zip(src_times.tolist(), dst_nodes.tolist())):
            src_time = int(src_time)
            dst_node = int(dst_node)
            key = (dst_node, src_time)
            count = timestamp_counts[key]
            edge_weight[i] = 1.0 / count
        
        # Add to data
        data[edge_type].edge_weight = edge_weight.to(edge_index.device)
        
        # Sanity checks
        assert edge_weight.shape[0] == edge_index.shape[1], \
            f"Edge weight shape mismatch for {edge_type}"
        assert torch.isfinite(edge_weight).all(), \
            f"Non-finite edge weights in {edge_type}"
        assert (edge_weight > 0).all(), \
            f"Non-positive edge weights in {edge_type}"
        
        logger.info(f"  {edge_type}: weight min={edge_weight.min():.4f}, " \
                   f"max={edge_weight.max():.4f}, mean={edge_weight.mean():.4f}")
    
    return data


def add_baseline_edge_weights(data: HeteroData) -> HeteroData:
    """
    Add uniform edge weights (all ones) for baseline method.
    """
    logger.info("Adding baseline (uniform) edge weights...")
    
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        
        if edge_index is None or edge_index.shape[1] == 0:
            continue
        
        num_edges = edge_index.shape[1]
        edge_weight = torch.ones(num_edges, dtype=torch.float32)
        data[edge_type].edge_weight = edge_weight.to(edge_index.device)
    
    return data
