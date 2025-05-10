"""
Improved version of IPR-MPNN model with structure-aware Cayley initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from models.improved_cayley_utils import improved_cayley_initialize_edge_weight, structure_aware_topk

class ImprovedIPRMPNNModel(nn.Module):
    """
    Improved IPR-MPNN model with structure-aware learnable edge weights.
    This version better preserves the structural advantages of Cayley graph
    while ensuring proper connectivity patterns.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, max_virtual_nodes=None, edge_init_type='uniform', 
                 top_k=None, high_contrast=True, structure_factor=0.8):
        super(ImprovedIPRMPNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_virtual_nodes = max_virtual_nodes
        self.edge_init_type = edge_init_type
        self.top_k = top_k  # If specified, will keep only top-k connections per base node
        self.high_contrast = high_contrast  # Whether to apply high contrast to edge weights
        self.structure_factor = structure_factor  # How much to emphasize existing structure
        
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
        
        # Process each graph separately as each will have its own virtual node structure
        graph_features = []
        
        for graph_idx in range(num_graphs):
            # Get nodes for this graph
            graph_mask = (batch == graph_idx)
            num_nodes = graph_mask.sum().item()
            graph_x = x[graph_mask]
            
            # Generate a unique identifier for this graph
            graph_id = f"{num_nodes}_{graph_idx}"
            
            # For each graph, calculate the optimal number of virtual nodes based on Cayley structure
            if graph_id not in self.graph_edge_weights:
                # First time seeing this graph, initialize edge weights
                from models.cayley_utils import calculate_optimal_virtual_nodes
                
                # Calculate optimal number of virtual nodes and Cayley parameter
                num_virtual_nodes, cayley_n = calculate_optimal_virtual_nodes(
                    num_base_nodes=num_nodes, 
                    verbose=False
                )
                
                if self.edge_init_type == 'cayley':
                    # Use improved Cayley graph initialization
                    init_weights = improved_cayley_initialize_edge_weight(
                        num_base_nodes=num_nodes, 
                        num_virtual_nodes=num_virtual_nodes,
                        cayley_n=cayley_n,
                        high_value=10.0,  # Higher value for stronger contrast
                        low_value=0.01,   # Smaller value for clearer distinction
                        contrast_factor=0.95,  # 95% of weight mass on Cayley edges
                        verbose=False
                    ).to(device)
                else:
                    # Use uniform initialization but with same number of virtual nodes for fair comparison
                    init_weights = torch.ones(num_nodes, num_virtual_nodes, device=device) / num_virtual_nodes
                
                # Create learnable parameter starting from the initialization
                self.graph_edge_weights[graph_id] = nn.Parameter(init_weights.clone()).to(device)
                
                # Register the parameter so it's included in optimizer
                self.register_parameter(f"edge_weights_{graph_id}", self.graph_edge_weights[graph_id])
            
            # Get the learnable edge weights for this graph
            edge_weights = self.graph_edge_weights[graph_id]
            num_virtual_nodes = edge_weights.size(1)
            
            # Compute affinity scores between base node features and virtual nodes
            affinity_features = self.affinity_mlp(graph_x)  # [num_nodes, hidden_dim]
            
            # Create virtual node embeddings (initially random, but learned during training)
            virtual_node_embeddings = torch.randn(num_virtual_nodes, self.hidden_dim, device=device)
            virtual_node_embeddings = virtual_node_embeddings / torch.norm(virtual_node_embeddings, dim=1, keepdim=True)
            
            # Compute attention scores between base nodes and virtual nodes
            base_features = affinity_features.unsqueeze(1)  # [num_nodes, 1, hidden_dim]
            virtual_features = virtual_node_embeddings.unsqueeze(0)  # [1, num_virtual_nodes, hidden_dim]
            
            # Compute dot product attention scores
            attention_scores = torch.sum(base_features * virtual_features, dim=2)  # [num_nodes, num_virtual_nodes]
            
            # Scale attention scores to be in a reasonable range
            attention_scores = attention_scores / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float, device=device))
            
            # Combine initial edge weights with learned attention scores
            # The multiplication preserves the structure of the initialization while allowing learning
            # Keep it in bounds with a sigmoid transformation
            edge_weights = edge_weights * (1.0 + torch.sigmoid(attention_scores))
            
            # If using top-k connectivity, apply our improved structure-aware pruning
            if self.top_k is not None and self.top_k < num_virtual_nodes:
                edge_weights = structure_aware_topk(
                    edge_weights, 
                    k=self.top_k,
                    high_contrast=self.high_contrast,
                    structure_factor=self.structure_factor
                )
            else:
                # Normalize weights to sum to 1 for each base node
                row_sums = edge_weights.sum(dim=1, keepdim=True)
                row_sums[row_sums == 0] = 1.0  # Avoid division by zero
                edge_weights = edge_weights / row_sums
            
            # Create virtual nodes for this graph
            virtual_nodes = torch.zeros(num_virtual_nodes, self.hidden_dim, device=device)
            
            # Update virtual nodes based on the learned weights
            for v_idx in range(num_virtual_nodes):
                # Get weights for this virtual node
                weights = edge_weights[:, v_idx].reshape(-1, 1)
                
                # Weighted aggregation from base nodes to virtual node
                virtual_nodes[v_idx] = (graph_x * weights).sum(dim=0)
            
            # Apply MLP to virtual nodes
            virtual_nodes = self.virtual_node_mlp(virtual_nodes)
            
            # Pool virtual nodes for this graph (mean pooling)
            graph_feature = virtual_nodes.mean(dim=0)
            graph_features.append(graph_feature)
        
        # Stack graph features
        graph_features = torch.stack(graph_features, dim=0)
        
        # Final prediction
        out = self.mlp(graph_features)
        
        return out
