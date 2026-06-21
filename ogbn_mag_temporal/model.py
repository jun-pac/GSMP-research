"""
HGAMLP-HOPE model for heterogeneous graph node classification with temporal message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import logging

logger = logging.getLogger(__name__)


class HGAMLPHOPELayer(nn.Module):
    """
    Single layer of HGAMLP-HOPE with weighted message passing.
    """
    def __init__(self, in_dim: int, out_dim: int, num_edge_types: int, dropout: float = 0.0):
        """
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            num_edge_types: Number of edge types
            dropout: Dropout rate
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_edge_types = num_edge_types
        
        # Per-edge-type linear transformations
        self.edge_transforms = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for _ in range(num_edge_types)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.message_chunk_size = 250_000
    
    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        """
        Forward pass.
        
        Args:
            x_dict: Dict mapping node_type -> features [num_nodes, in_dim]
            edge_index_dict: Dict mapping edge_type -> edge_index [2, num_edges]
            edge_weight_dict: Dict mapping edge_type -> edge_weight [num_edges]
        
        Returns:
            out_dict: Dict mapping node_type -> aggregated features [num_nodes, out_dim]
        """
        out_dict = {}
        
        # Collect all node types that will receive messages
        all_node_types = set()
        for (src, rel, dst) in edge_index_dict.keys():
            all_node_types.add(dst)
        
        for dst_type in all_node_types:
            # Aggregate messages from all edge types targeting this node type
            aggregated = None
            num_nodes = x_dict[dst_type].shape[0] if dst_type in x_dict else None
            
            for edge_idx, ((src_type, rel, dst_t), edge_index) in enumerate(edge_index_dict.items()):
                if dst_t != dst_type:
                    continue
                
                if edge_index.shape[1] == 0:
                    # No edges for this type
                    continue
                
                src_features = x_dict[src_type]  # [num_src_nodes, in_dim]
                
                # Get edge indices and weights
                src_idx = edge_index[0]
                dst_idx = edge_index[1]
                edge_weight = edge_weight_dict[(src_type, rel, dst_t)]  # [num_edges]
                
                sum_src = torch.zeros(num_nodes, self.in_dim,
                                      device=src_features.device,
                                      dtype=src_features.dtype)

                # Linear aggregation can be moved before the relation transform:
                # mean(Wx + b) == W mean(x) + b. This avoids storing transformed
                # features for every source node and edge on ogbn-mag.
                num_edges = edge_index.shape[1]
                for start in range(0, num_edges, self.message_chunk_size):
                    end = min(start + self.message_chunk_size, num_edges)
                    weighted_features = (
                        src_features[src_idx[start:end]]
                        * edge_weight[start:end].unsqueeze(1)
                    )
                    sum_src.index_add_(0, dst_idx[start:end], weighted_features)
                
                weight_sum = torch.zeros(num_nodes, 1,
                                        device=edge_weight.device,
                                        dtype=edge_weight.dtype)
                weight_sum.index_add_(0, dst_idx, edge_weight.unsqueeze(1))
                
                # Avoid division by zero
                has_messages = weight_sum > 0
                mean_src = sum_src / torch.clamp(weight_sum, min=1e-8)
                aggregated_edge = self.edge_transforms[edge_idx](mean_src)
                aggregated_edge = aggregated_edge * has_messages.to(aggregated_edge.dtype)
                
                # Accumulate across edge types
                if aggregated is None:
                    aggregated = aggregated_edge
                else:
                    aggregated = aggregated + aggregated_edge
            
            if aggregated is not None:
                out_dict[dst_type] = self.dropout(aggregated)
            else:
                # No messages for this node type, keep original or zero
                num_nodes_dst = x_dict[dst_type].shape[0] if dst_type in x_dict else 0
                out_dict[dst_type] = torch.zeros(num_nodes_dst, self.out_dim, 
                                                 device=x_dict[dst_type].device,
                                                 dtype=x_dict[dst_type].dtype)

        for node_type, features in x_dict.items():
            if node_type not in out_dict:
                out_dict[node_type] = features
        
        return out_dict


class HGAMLPHOPE(nn.Module):
    """
    HGAMLP-HOPE: Heterogeneous Graph Multi-Layer Perceptron with High-Order Proximity.
    
    This model performs multi-hop message passing on heterogeneous graphs and combines
    representations from different hops using attention.
    """
    
    def __init__(
        self,
        num_node_types: int,
        node_type_names: list,
        input_dims: dict,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        """
        Args:
            num_node_types: Number of node types
            node_type_names: List of node type names
            input_dims: Dict mapping node_type -> input dimension
            hidden_dim: Hidden dimension
            num_classes: Number of output classes (for target node type 'paper')
            num_layers: Number of propagation layers
            dropout: Dropout rate
            use_attention: Whether to use attention for combining hops
        """
        super().__init__()
        
        self.node_type_names = node_type_names
        self.target_node_type = "paper"
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Input projections for each node type
        self.input_projs = nn.ModuleDict()
        for node_type in node_type_names:
            in_dim = input_dims.get(node_type, hidden_dim)
            self.input_projs[node_type] = nn.Linear(in_dim, hidden_dim)
        
        # Propagation layers
        self.prop_layers = nn.ModuleList([
            HGAMLPHOPELayer(hidden_dim, hidden_dim, num_edge_types=100, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Attention for combining hops (for paper nodes)
        if use_attention:
            self.hop_attention = nn.Linear(hidden_dim, 1)
        
        # Final MLP classifier for paper nodes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout_rate = dropout
    
    def forward(self, data: HeteroData, return_embeddings: bool = False):
        """
        Forward pass.
        
        Args:
            data: HeteroData object with:
                - node features: x_dict[node_type] -> [num_nodes, feature_dim]
                - edge indices: edge_index_dict[(src_type, rel, dst_type)] -> [2, num_edges]
                - edge weights: (optional) edge_weight in edge store
            return_embeddings: If True, return paper embeddings instead of logits
        
        Returns:
            logits: Classification logits for paper nodes [num_papers, num_classes]
        """
        # Prepare edge_index_dict and edge_weight_dict
        edge_index_dict = {}
        edge_weight_dict = {}
        
        for edge_type in data.edge_types:
            edge_index_dict[edge_type] = data[edge_type].edge_index
            
            # Get edge weights (default to ones if not provided)
            if hasattr(data[edge_type], 'edge_weight') and data[edge_type].edge_weight is not None:
                edge_weight_dict[edge_type] = data[edge_type].edge_weight
            else:
                num_edges = edge_index_dict[edge_type].shape[1]
                edge_weight_dict[edge_type] = torch.ones(num_edges, 
                                                         device=edge_index_dict[edge_type].device)
        
        # Prepare feature dict
        x_dict = {}
        for node_type in data.node_types:
            if getattr(data[node_type], "x", None) is not None:
                x_dict[node_type] = data[node_type].x
            else:
                # Create learnable embeddings for nodes without features
                num_nodes = data[node_type].num_nodes
                x_dict[node_type] = torch.randn(num_nodes, self.hidden_dim,
                                                 device=next(self.parameters()).device)
        
        # Input projection
        h_dict = {}
        for node_type in data.node_types:
            h_dict[node_type] = self.input_projs[node_type](x_dict[node_type])
        
        # Store hop-wise representations for paper nodes
        paper_hops = [h_dict[self.target_node_type].clone()]
        
        # Multi-hop propagation
        for layer_idx in range(self.num_layers):
            h_dict = self.prop_layers[layer_idx](h_dict, edge_index_dict, edge_weight_dict)
            
            if self.target_node_type in h_dict:
                paper_hops.append(h_dict[self.target_node_type].clone())
        
        # Combine hop-wise representations
        if self.use_attention and len(paper_hops) > 1:
            # Use attention to combine hops
            # paper_hops: list of [num_papers, hidden_dim]
            stacked = torch.stack(paper_hops, dim=1)  # [num_papers, num_hops, hidden_dim]
            
            # Compute attention weights per hop
            attn_scores = self.hop_attention(stacked)  # [num_papers, num_hops, 1]
            attn_weights = F.softmax(attn_scores, dim=1)  # [num_papers, num_hops, 1]
            
            # Weighted combination
            combined = (stacked * attn_weights).sum(dim=1)  # [num_papers, hidden_dim]
        else:
            # Simple average or just use last hop
            combined = torch.stack(paper_hops, dim=0).mean(dim=0)  # [num_papers, hidden_dim]
        
        if return_embeddings:
            return combined
        
        # Classification
        logits = self.classifier(combined)
        
        return logits


def create_model(
    data: HeteroData,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_classes: int = 349,
    dropout: float = 0.1,
    device: torch.device = torch.device("cpu")
) -> HGAMLPHOPE:
    """
    Create HGAMLPHOPE model for ogbn-mag.
    """
    node_type_names = list(data.node_types)
    
    # Get input dimensions
    input_dims = {}
    for node_type in node_type_names:
        if getattr(data[node_type], "x", None) is not None:
            input_dims[node_type] = data[node_type].x.shape[1]
        else:
            input_dims[node_type] = hidden_dim
    
    model = HGAMLPHOPE(
        num_node_types=len(node_type_names),
        node_type_names=node_type_names,
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        use_attention=True
    ).to(device)
    
    logger.info(f"Created HGAMLPHOPE model:")
    logger.info(f"  Hidden dim: {hidden_dim}")
    logger.info(f"  Num layers: {num_layers}")
    logger.info(f"  Num classes: {num_classes}")
    logger.info(f"  Dropout: {dropout}")
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model
