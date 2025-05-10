"""
Example of how IPR-MPNN could be implemented with learnable connectivity.
This is a demonstration only and not meant to be integrated into the existing code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableIPRMPNN(nn.Module):
    """
    IPR-MPNN with learnable connectivity between base and virtual nodes.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_virtual_nodes=10, top_k_connections=5, initialization='cayley'):
        super(LearnableIPRMPNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_virtual_nodes = num_virtual_nodes
        self.top_k_connections = top_k_connections  # Number of connections to keep per virtual node
        self.initialization = initialization
        
        # Node embedding layer
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Affinity scores - learnable parameters determining connectivity
        # This will learn which base nodes should connect to which virtual nodes
        self.affinity_scores = nn.Parameter(torch.randn(1, hidden_dim, num_virtual_nodes))
        
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
        
    def initialize_connectivity(self, base_node_features, initialization='cayley', cayley_n=None):
        """
        Initialize connectivity pattern between base and virtual nodes.
        
        Args:
            base_node_features: Features of base nodes [num_nodes, hidden_dim]
            initialization: Type of initialization ('cayley' or 'uniform')
            cayley_n: Parameter for Cayley graph if using Cayley initialization
            
        Returns:
            torch.Tensor: Initial weights [num_nodes, num_virtual_nodes]
        """
        num_nodes = base_node_features.shape[0]
        device = base_node_features.device
        
        if initialization == 'cayley':
            from models.cayley_utils import calculate_optimal_virtual_nodes, cayley_initialize_edge_weight
            
            # Use our existing Cayley utilities
            if self.num_virtual_nodes != 'optimal':
                # If virtual nodes count is fixed, use it
                cayley_n = cayley_n or get_cayley_n(num_nodes + self.num_virtual_nodes)
                weights = cayley_initialize_edge_weight(
                    num_base_nodes=num_nodes,
                    num_virtual_nodes=self.num_virtual_nodes,
                    cayley_n=cayley_n,
                    verbose=False
                ).to(device)
            else:
                # Calculate optimal count
                num_virtual_nodes, cayley_n = calculate_optimal_virtual_nodes(
                    num_base_nodes=num_nodes,
                    verbose=False
                )
                weights = cayley_initialize_edge_weight(
                    num_base_nodes=num_nodes,
                    num_virtual_nodes=num_virtual_nodes,
                    cayley_n=cayley_n,
                    verbose=False
                ).to(device)
        else:
            # Uniform initialization
            weights = torch.ones(num_nodes, self.num_virtual_nodes, device=device) / self.num_virtual_nodes
            
        return weights
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Embed nodes
        x = self.node_embedding(x)
        
        # Get graph information
        num_graphs = batch.max().item() + 1
        device = x.device
        
        # Process each graph separately
        graph_features = []
        
        for graph_idx in range(num_graphs):
            # Get nodes for this graph
            graph_mask = (batch == graph_idx)
            graph_x = x[graph_mask]
            num_nodes = graph_mask.sum().item()
            
            # Initialize weights based on chosen strategy
            initial_weights = self.initialize_connectivity(
                graph_x, 
                initialization=self.initialization
            )
            
            # Now compute learnable affinity scores to determine connectivity
            # This is where the model learns which connections matter most
            
            # Compute affinity between base nodes and virtual nodes using a dot product
            # [num_nodes, hidden_dim] @ [hidden_dim, num_virtual_nodes] -> [num_nodes, num_virtual_nodes]
            affinity = torch.matmul(graph_x, self.affinity_scores.expand(num_nodes, -1, -1))
            affinity = affinity.squeeze(1)
            
            # IMPORTANT PART: Only keep top-k connections for each virtual node
            # This implements the pruning step of IPR-MPNN
            topk_values, topk_indices = torch.topk(affinity, k=min(self.top_k_connections, num_nodes), dim=0)
            
            # Create a mask for the top-k connections per virtual node
            mask = torch.zeros_like(affinity)
            for v_idx in range(self.num_virtual_nodes):
                mask[topk_indices[:, v_idx], v_idx] = 1
                
            # Apply mask and use softmax to normalize the weights for each virtual node
            # This ensures that the weights sum to 1 for each virtual node
            masked_affinity = affinity * mask
            normalized_weights = F.softmax(masked_affinity, dim=0)
            
            # Create virtual nodes
            virtual_nodes = torch.zeros(self.num_virtual_nodes, self.hidden_dim, device=device)
            
            # Update virtual nodes using the learned weights
            for v_idx in range(self.num_virtual_nodes):
                # Get weights for this virtual node
                weights = normalized_weights[:, v_idx].reshape(-1, 1)
                
                # Weighted aggregation from base nodes to virtual node
                aggregated = (graph_x * weights).sum(dim=0)
                
                # Update the virtual node
                virtual_nodes[v_idx] = aggregated
            
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

# Example usage
if __name__ == "__main__":
    # Create a dummy input
    from torch_geometric.data import Data, Batch
    import numpy as np
    
    # Create a small graph
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn(3, 5)  # 3 nodes, 5 features
    data = Data(x=x, edge_index=edge_index)
    
    # Create a batch with multiple graphs
    batch = Batch.from_data_list([data, data, data])
    
    # Create model
    model = LearnableIPRMPNN(
        input_dim=5,
        hidden_dim=16,
        output_dim=2,
        num_virtual_nodes=4,
        top_k_connections=2,  # Each virtual node connects to 2 base nodes
        initialization='cayley'
    )
    
    # Forward pass
    out = model(batch)
    print(f"Output shape: {out.shape}")  # Should be [3, 2] for 3 graphs, 2 classes
