"""
Data loading and preprocessing for ogbn-mag dataset.
"""

import torch
import numpy as np
from torch_geometric.data import HeteroData
from ogb.nodeproppred import PygNodePropPredDataset
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def _torch_load_with_weights_only_false(*args, **kwargs):
    """Compatibility shim for OGB/PyG processed files under PyTorch >= 2.6."""
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


_ORIGINAL_TORCH_LOAD = torch.load


def _legacy_ogb_data_to_heterodata(data) -> HeteroData:
    """Convert OGB's legacy ogbn-mag PyG Data object into HeteroData."""
    if hasattr(data, "node_types") and hasattr(data, "edge_types"):
        return data

    hetero = HeteroData()

    for node_type, num_nodes in data.num_nodes_dict.items():
        hetero[node_type].num_nodes = num_nodes

        if node_type in data.x_dict:
            hetero[node_type].x = data.x_dict[node_type]

        if node_type in data.y_dict:
            hetero[node_type].y = data.y_dict[node_type].view(-1)

        if hasattr(data, "node_year") and node_type in data.node_year:
            hetero[node_type].year = data.node_year[node_type].view(-1)

    for edge_type, edge_index in data.edge_index_dict.items():
        hetero[edge_type].edge_index = edge_index

    return hetero


def load_ogbn_mag(root: str = "./data") -> Tuple[HeteroData, Dict, int, Dict]:
    """
    Load ogbn-mag dataset from OGB.
    
    Returns:
        data: HeteroData object
        split_idx: Dict with 'train', 'valid', 'test' masks/indices for paper nodes
        num_classes: Number of classes for paper node classification
        timestamp_dict: Dict mapping node_type -> timestamp tensor
    """
    logger.info(f"Loading ogbn-mag from {root}...")

    original_torch_load = torch.load
    torch.load = _torch_load_with_weights_only_false
    try:
        dataset = PygNodePropPredDataset(name="ogbn-mag", root=root)
    finally:
        torch.load = original_torch_load

    data = dataset[0]
    data = _legacy_ogb_data_to_heterodata(data)

    split_idx = dataset.get_idx_split()
    if isinstance(split_idx.get("train"), dict):
        split_idx = {key: value["paper"] for key, value in split_idx.items()}
    
    num_classes = dataset.num_classes
    
    logger.info(f"Dataset loaded. Num classes: {num_classes}")
    logger.info(f"Train nodes: {len(split_idx['train'])}")
    logger.info(f"Valid nodes: {len(split_idx['valid'])}")
    logger.info(f"Test nodes: {len(split_idx['test'])}")
    
    return data, split_idx, num_classes, None


def compute_timestamps(
    data: HeteroData,
    strategy: str = "mean"
) -> Dict[str, torch.Tensor]:
    """
    Compute timestamps for all node types.
    
    Paper nodes use their year attribute.
    Other nodes compute timestamps from connected paper nodes using the specified strategy.
    
    Args:
        data: HeteroData object
        strategy: 'mean', 'min', or 'max' for aggregating paper node years
    
    Returns:
        timestamp_dict: Dict mapping node_type -> timestamp tensor (on CPU)
    """
    logger.info(f"Computing timestamps with strategy: {strategy}")
    
    timestamp_dict = {}
    
    # Get paper timestamps from the year attribute
    if "paper" in data.node_types and getattr(data["paper"], "year", None) is not None:
        paper_years = data["paper"].year.float()  # shape: [num_papers]
        timestamp_dict["paper"] = paper_years
        t_min_paper = paper_years.min().item()
        t_max_paper = paper_years.max().item()
        logger.info(f"Paper timestamps: min={t_min_paper:.0f}, max={t_max_paper:.0f}")
    else:
        raise ValueError("Paper node type not found or year attribute missing")
    
    # For other node types, infer from connected papers
    node_types = [nt for nt in data.node_types if nt != "paper"]
    
    for node_type in node_types:
        num_nodes = data[node_type].num_nodes
        timestamps = torch.full((num_nodes,), -1.0, device=paper_years.device)

        if strategy == "mean":
            time_sum = torch.zeros(num_nodes, dtype=paper_years.dtype, device=paper_years.device)
            time_count = torch.zeros(num_nodes, dtype=paper_years.dtype, device=paper_years.device)
        elif strategy == "min":
            time_value = torch.full((num_nodes,), float("inf"), dtype=paper_years.dtype, device=paper_years.device)
        elif strategy == "max":
            time_value = torch.full((num_nodes,), float("-inf"), dtype=paper_years.dtype, device=paper_years.device)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Find all edge types involving this node type and papers
        for edge_type in data.edge_types:
            src_type, rel_type, dst_type = edge_type

            edge_index = data[edge_type].edge_index
            node_ids = None
            connected_times = None
            
            # Case 1: node_type -> ... -> paper
            if src_type == node_type and dst_type == "paper":
                node_ids = edge_index[0]
                connected_times = paper_years[edge_index[1]]
            
            # Case 2: paper -> ... -> node_type (reverse edges)
            elif dst_type == node_type and src_type == "paper":
                node_ids = edge_index[1]
                connected_times = paper_years[edge_index[0]]

            if node_ids is None:
                continue

            if strategy == "mean":
                time_sum.index_add_(0, node_ids, connected_times)
                time_count.index_add_(0, node_ids, torch.ones_like(connected_times))
            elif strategy == "min":
                time_value.scatter_reduce_(0, node_ids, connected_times, reduce="amin", include_self=True)
            elif strategy == "max":
                time_value.scatter_reduce_(0, node_ids, connected_times, reduce="amax", include_self=True)

        if strategy == "mean":
            valid_mask = time_count > 0
            timestamps[valid_mask] = (time_sum[valid_mask] / time_count[valid_mask]).round()
        elif strategy == "min":
            valid_mask = torch.isfinite(time_value)
            timestamps[valid_mask] = time_value[valid_mask]
        elif strategy == "max":
            valid_mask = torch.isfinite(time_value)
            timestamps[valid_mask] = time_value[valid_mask]
        
        timestamp_dict[node_type] = timestamps
        num_valid = (timestamps != -1).sum().item()
        if num_valid > 0:
            valid_ts = timestamps[timestamps != -1]
            logger.info(f"{node_type} timestamps: min={valid_ts.min():.0f}, max={valid_ts.max():.0f}, valid={num_valid}/{num_nodes}")
        else:
            logger.warning(f"No valid timestamps computed for {node_type}")
    
    return timestamp_dict


def get_temporal_bounds(timestamp_dict: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    """
    Get global temporal bounds from all valid timestamps.
    
    Returns:
        (t_min, t_max): global minimum and maximum timestamps
    """
    all_timestamps = []
    for node_type, ts in timestamp_dict.items():
        valid_ts = ts[ts != -1]
        if len(valid_ts) > 0:
            all_timestamps.append(valid_ts)
    
    if not all_timestamps:
        raise ValueError("No valid timestamps found in any node type")
    
    all_ts = torch.cat(all_timestamps)
    t_min = all_ts.min().item()
    t_max = all_ts.max().item()
    
    logger.info(f"Global temporal bounds: t_min={t_min:.0f}, t_max={t_max:.0f}")
    
    return t_min, t_max


def get_node_type_for_nodeid(data: HeteroData, node_id: int) -> str:
    """
    Get node type for a given node ID.
    Assumes node IDs are contiguous per node type (which is true for HeteroData).
    """
    cumsum = 0
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        if node_id < cumsum + num_nodes:
            return node_type
        cumsum += num_nodes
    raise ValueError(f"Node ID {node_id} out of range")


def ensure_undirected_edges(data: HeteroData, reverse_edge_type_names: Optional[Dict] = None) -> HeteroData:
    """
    Ensure undirected edges by adding reverse edges where needed.
    
    For ogbn-mag, this adds reverse edges for citation and other relations.
    
    Args:
        data: HeteroData object
        reverse_edge_type_names: Optional dict mapping original edge types to reverse names
    
    Returns:
        Modified data with reverse edges
    """
    logger.info("Adding reverse edges for undirected message passing...")
    
    # Default reverse edge names for ogbn-mag
    if reverse_edge_type_names is None:
        reverse_edge_type_names = {
            ("paper", "cites", "paper"): ("paper", "cited_by", "paper"),
            ("author", "writes", "paper"): ("paper", "written_by", "author"),
            ("paper", "has_topic", "field_of_study"): ("field_of_study", "has_paper", "paper"),
            ("institution", "affiliated_with", "author"): ("author", "affiliated_institution", "institution"),
        }
    
    # Add reverse edges
    for orig_type, rev_type in reverse_edge_type_names.items():
        if orig_type in data.edge_types:
            edge_index = data[orig_type].edge_index
            src, rel, dst = orig_type
            rev_src, rev_rel, rev_dst = rev_type
            
            # Create reverse edge index
            reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            
            # Add reverse edges if not already present
            if rev_type not in data.edge_types:
                data[rev_type].edge_index = reverse_edge_index
                logger.info(f"Added reverse edge type: {rev_type} with {reverse_edge_index.shape[1]} edges")
    
    return data


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, "/users/PAS1289/jyp531/GSMP-research/ogbn_mag_temporal")
    from utils import configure_logger
    
    configure_logger()
    
    data, split_idx, num_classes, _ = load_ogbn_mag()
    print("\nDataset loaded successfully")
    print(f"Node types: {data.node_types}")
    print(f"Edge types: {data.edge_types}")
    print(f"Number of classes: {num_classes}")
    
    timestamp_dict = compute_timestamps(data, strategy="mean")
    print("\nTimestamps computed successfully")
    
    t_min, t_max = get_temporal_bounds(timestamp_dict)
    print(f"Temporal bounds: [{t_min}, {t_max}]")
