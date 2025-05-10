"""
Memory-efficient version of the balanced IPR-MPNN model for larger datasets like ENZYMES.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from models.balanced_cayley_utils import balanced_cayley_initialize_edge_weight, balanced_topk_pruning

class EfficientBalancedIPRMPNNModel(nn.Module):
    """
    Memory-efficient version of the Balanced IPR-MPNN model with batched processing
    to handle larger datasets like ENZYMES.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, max_virtual_nodes=None, 
                 edge_init_type='uniform', top_k=None, max_nodes_per_batch=100):
        super(EfficientBalancedIPRMPNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_virtual_nodes = max_virtual_nodes
        self.edge_init_type = edge_init_type
        self.top_k = top_k  # If specified, will keep only top-k connections per base node
        self.max_nodes_per_batch = max_nodes_per_batch  # Limit nodes processed at once
        
        # Node embedding layers
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Learnable affinity parameter for edge weights
        self.affinity_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Virtual node MLP
        self.virtual_node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Prediction head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Dictionary to store the learnable edge weights for each graph
        self.graph_edge_weights = {}
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Embed nodes
        x = self.node_embedding(x)
        
        # Initial graph convolution
        x = F.relu(self.conv1(x, edge_index))
        
        # Get graph information
        num_graphs = batch.max().item() + 1
        device = x.device
        
        # Process each graph separately
        graph_features = []
        
        for graph_idx in range(num_graphs):
            # Get nodes for this graph
            graph_mask = (batch == graph_idx)
            num_nodes = graph_mask.sum().item()
            graph_x = x[graph_mask]
            
            # Generate a unique identifier for this graph
            graph_id = f"{num_nodes}_{graph_idx}"
            
            # First time seeing this graph, initialize edge weights
            if graph_id not in self.graph_edge_weights:
                from models.cayley_utils import calculate_optimal_virtual_nodes
                
                # Calculate optimal number of virtual nodes and Cayley parameter
                num_virtual_nodes, cayley_n = calculate_optimal_virtual_nodes(
                    num_base_nodes=num_nodes, 
                    verbose=False
                )
                
                if self.edge_init_type == 'cayley':
                    # Use balanced Cayley graph initialization with moderate contrast
                    init_weights = balanced_cayley_initialize_edge_weight(
                        num_base_nodes=num_nodes, 
                        num_virtual_nodes=num_virtual_nodes,
                        cayley_n=cayley_n,
                        high_value=2.0,
                        low_value=0.2,
                        verbose=False
                    ).to(device)
                else:
                    # Use uniform initialization
                    init_weights = torch.ones(num_nodes, num_virtual_nodes, device=device) / num_virtual_nodes
                
                # Create learnable parameter from the initialization
                self.graph_edge_weights[graph_id] = nn.Parameter(init_weights.clone()).to(device)
                
                # Register the parameter
                self.register_parameter(f"edge_weights_{graph_id}", self.graph_edge_weights[graph_id])
            
            # Get the edge weights for this graph
            edge_weights = self.graph_edge_weights[graph_id]
            num_virtual_nodes = edge_weights.size(1)
            
            # Process in batches if the graph is large
            if num_nodes > self.max_nodes_per_batch:
                # Process in batches to reduce memory usage
                batch_size = min(self.max_nodes_per_batch, num_nodes)
                num_batches = (num_nodes + batch_size - 1) // batch_size
                
                virtual_nodes = torch.zeros(num_virtual_nodes, self.hidden_dim, device=device)
                
                for b in range(num_batches):
                    start_idx = b * batch_size
                    end_idx = min((b + 1) * batch_size, num_nodes)
                    
                    # Get batch of node features
                    batch_features = graph_x[start_idx:end_idx]
                    batch_weights = edge_weights[start_idx:end_idx]
                    
                    # Apply affinity transformation
                    transformed_features = self.affinity_mlp(batch_features)
                    
                    # Compute batch attention
                    batch_avg = transformed_features.mean(dim=0, keepdim=True)
                    batch_similarity = F.cosine_similarity(
                        transformed_features.unsqueeze(1),
                        batch_avg.unsqueeze(0),
                        dim=2
                    ).squeeze(1)
                    
                    # Create a moderate attention mechanism
                    batch_attention = (1.0 + batch_similarity.unsqueeze(1)) / 2.0
                    
                    # Apply attention to weights
                    modified_weights = batch_weights.clone() * batch_attention
                    
                    # Apply top-k if specified
                    if self.top_k is not None and self.top_k < num_virtual_nodes:
                        modified_weights = balanced_topk_pruning(modified_weights, k=self.top_k)
                    else:
                        # Normalize
                        row_sums = modified_weights.sum(dim=1, keepdim=True)
                        row_sums[row_sums == 0] = 1.0
                        modified_weights = modified_weights / row_sums
                    
                    # Update virtual nodes from this batch
                    for v_idx in range(num_virtual_nodes):
                        weights = modified_weights[:, v_idx].reshape(-1, 1)
                        virtual_nodes[v_idx] += (batch_features * weights).sum(dim=0)
                
                # Normalize virtual nodes if we had multiple batches
                if num_batches > 1:
                    virtual_nodes = virtual_nodes / num_batches
            else:
                # For smaller graphs, process all at once
                transformed_features = self.affinity_mlp(graph_x)
                
                # Create virtual prototype
                virtual_prototype = transformed_features.mean(dim=0, keepdim=True)
                
                # Compute similarity scores
                similarity = F.cosine_similarity(
                    transformed_features.unsqueeze(1),
                    virtual_prototype.unsqueeze(0),
                    dim=2
                ).squeeze(1)
                
                # Create a moderate attention mechanism
                attention = (1.0 + similarity.unsqueeze(1)) / 2.0
                
                # Apply attention to weights
                modified_weights = edge_weights.clone() * attention
                
                # Apply top-k if specified
                if self.top_k is not None and self.top_k < num_virtual_nodes:
                    modified_weights = balanced_topk_pruning(modified_weights, k=self.top_k)
                else:
                    # Normalize
                    row_sums = modified_weights.sum(dim=1, keepdim=True)
                    row_sums[row_sums == 0] = 1.0
                    modified_weights = modified_weights / row_sums
                
                # Create virtual nodes
                virtual_nodes = torch.zeros(num_virtual_nodes, self.hidden_dim, device=device)
                
                # Update virtual nodes
                for v_idx in range(num_virtual_nodes):
                    weights = modified_weights[:, v_idx].reshape(-1, 1)
                    virtual_nodes[v_idx] = (graph_x * weights).sum(dim=0)
            
            # Apply MLP to virtual nodes
            virtual_nodes = self.virtual_node_mlp(virtual_nodes)
            
            # Pool virtual nodes for this graph
            graph_feature = virtual_nodes.mean(dim=0)
            graph_features.append(graph_feature)
        
        # Stack graph features
        graph_features = torch.stack(graph_features, dim=0)
        
        # Final prediction
        out = self.mlp(graph_features)
        
        return out
