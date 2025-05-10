"""
Ultra-lightweight version of the IPR-MPNN model specifically for memory-constrained environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from models.balanced_cayley_utils import balanced_cayley_initialize_edge_weight, balanced_topk_pruning

class MemorySaverIPRMPNNModel(nn.Module):
    """
    Ultra memory-efficient IPR-MPNN model that:
    1. Uses smaller hidden dimensions
    2. Processes graphs in small batches
    3. Uses shared parameters across similar graphs
    4. Employs aggressive garbage collection
    """
    def __init__(self, input_dim, hidden_dim, output_dim, edge_init_type='uniform', top_k=3):
        super(MemorySaverIPRMPNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # Use smaller hidden dimension (e.g., 16 or 32)
        self.output_dim = output_dim
        self.edge_init_type = edge_init_type
        self.top_k = top_k  # Use smaller k value to reduce memory
        
        # Node embedding layer
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Single graph convolution layer (reduced from 2)
        self.conv = GCNConv(hidden_dim, hidden_dim)
        
        # Simplified affinity function (fewer parameters)
        self.affinity = nn.Linear(hidden_dim, hidden_dim)
        
        # Simple prediction head (fewer layers)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Create a small set of shared edge weight templates to reuse
        # This drastically reduces memory by not creating unique parameters for each graph
        self.max_template_nodes = 30
        self.weight_templates = {}
        
        # Store last computed edge weights for analysis
        self.last_edge_weights = {}
        self.collect_oversquashing_metrics = False
        
    def _get_or_create_template(self, num_nodes, num_virtual_nodes, device):
        """Get existing weight template or create a new one based on closest match"""
        
        # If exact template exists, use it
        template_key = f"{num_nodes}_{num_virtual_nodes}"
        if template_key in self.weight_templates:
            return self.weight_templates[template_key]
        
        # Find closest template by node count
        closest_key = None
        min_diff = float('inf')
        
        for key in self.weight_templates:
            n, v = map(int, key.split('_'))
            # Only consider templates with sufficient virtual nodes
            if v >= num_virtual_nodes:
                diff = abs(n - num_nodes)
                if diff < min_diff:
                    min_diff = diff
                    closest_key = key
        
        # If close enough template exists, adapt it
        if closest_key is not None and min_diff <= 5:
            base_template = self.weight_templates[closest_key]
            n, v = map(int, closest_key.split('_'))
            
            # Adapt template to current size
            if n > num_nodes:
                # Trim the template
                adapted_template = base_template[:num_nodes, :num_virtual_nodes].clone()
                # Renormalize
                row_sums = adapted_template.sum(dim=1, keepdim=True)
                row_sums[row_sums == 0] = 1.0
                adapted_template = adapted_template / row_sums
            else:  # n < num_nodes
                # Expand by repeating patterns
                adapted_template = torch.zeros(num_nodes, num_virtual_nodes, device=device)
                for i in range(num_nodes):
                    adapted_template[i] = base_template[i % n, :num_virtual_nodes]
                # Renormalize
                row_sums = adapted_template.sum(dim=1, keepdim=True)
                row_sums[row_sums == 0] = 1.0
                adapted_template = adapted_template / row_sums
                
            return adapted_template
        
        # Otherwise create new template if we haven't exceeded the limit
        if len(self.weight_templates) < 10:  # Limit number of templates to save memory
            from models.cayley_utils import calculate_optimal_virtual_nodes
            
            cayley_n = None
            if num_virtual_nodes == 0:
                # Calculate optimal virtual nodes if not specified
                num_virtual_nodes, cayley_n = calculate_optimal_virtual_nodes(
                    num_base_nodes=num_nodes, 
                    verbose=False
                )
            
            # Create template based on initialization type
            if self.edge_init_type == 'cayley':
                template = balanced_cayley_initialize_edge_weight(
                    num_base_nodes=num_nodes, 
                    num_virtual_nodes=num_virtual_nodes,
                    cayley_n=cayley_n,
                    high_value=2.0,
                    low_value=0.2,
                    verbose=False
                ).to(device)
            else:
                template = torch.ones(num_nodes, num_virtual_nodes, device=device) / num_virtual_nodes
                
            # Apply top-k pruning to reduce connections
            if self.top_k is not None and self.top_k < num_virtual_nodes:
                template = balanced_topk_pruning(template, k=self.top_k)
                
            # Store template
            self.weight_templates[template_key] = template
            return template
            
        # Last resort: use smallest existing template
        smallest_key = min(self.weight_templates.keys(), key=lambda k: int(k.split('_')[0]))
        return self._get_or_create_template(int(smallest_key.split('_')[0]), num_virtual_nodes, device)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Get device
        device = x.device
        
        # Embed nodes (smaller embedding size reduces memory)
        x = self.node_embedding(x)
        
        # Single graph convolution
        x = F.relu(self.conv(x, edge_index))
        
        # Process each graph separately but efficiently
        num_graphs = batch.max().item() + 1
        graph_features = []
        
        # Clear previous edge weights if collecting metrics
        if self.collect_oversquashing_metrics:
            self.last_edge_weights = {}
        
        for graph_idx in range(num_graphs):
            # Process one graph at a time
            graph_mask = (batch == graph_idx)
            num_nodes = graph_mask.sum().item()
            graph_x = x[graph_mask]
            
            # Skip extremely large graphs to avoid OOM errors
            if num_nodes > self.max_template_nodes:
                # For very large graphs, use global mean pooling as a fallback
                graph_feature = global_mean_pool(graph_x, torch.zeros(num_nodes, device=device, dtype=torch.long))
                graph_features.append(graph_feature.squeeze())
                continue
            
            # Calculate virtual nodes (fewer than standard model)
            num_virtual_nodes = min(num_nodes // 2, 10)  # Limit number of virtual nodes
            
            # Get edge weights from template (shared parameters)
            edge_weights = self._get_or_create_template(num_nodes, num_virtual_nodes, device)
            
            # Apply simple learned transformation
            node_features = F.relu(self.affinity(graph_x))
            
            # Very simple attention-like mechanism (much more memory efficient)
            attention = torch.sigmoid(node_features.mean(dim=1, keepdim=True))
            
            # Apply to edge weights and renormalize
            modified_weights = edge_weights.clone() * attention
            row_sums = modified_weights.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1.0
            modified_weights = modified_weights / row_sums
            
            # Apply top-k connectivity if specified
            if self.top_k is not None and self.top_k < num_virtual_nodes:
                # Make sure to avoid in-place operations on tensors that require gradients
                modified_weights = balanced_topk_pruning(modified_weights, k=self.top_k)
            
            # Store modified weights for analysis if collecting metrics
            if self.collect_oversquashing_metrics:
                # Use detach to avoid tracking computation graph for analysis
                print(f"Storing edge weights for graph {graph_idx}, shape: {modified_weights.shape}")
                self.last_edge_weights[graph_idx] = {
                    'edge_weights': modified_weights.detach().clone(),
                    'num_nodes': num_nodes,
                    'num_virtual_nodes': num_virtual_nodes,
                    'attention': attention.detach().clone() if attention is not None else None
                }
            
            # Create virtual nodes
            virtual_nodes = torch.zeros(num_virtual_nodes, self.hidden_dim, device=device)
            
            # More efficient virtual node update (use matrix multiplication)
            virtual_nodes = torch.mm(modified_weights.t(), graph_x)
            
            # Mean pooling of virtual nodes (skip MLP to save memory)
            graph_feature = virtual_nodes.mean(dim=0)
            graph_features.append(graph_feature.squeeze())
            
            # Explicitly clear tensors to help with memory management
            del graph_x, edge_weights, modified_weights, virtual_nodes
        
        # Stack graph features
        graph_features = torch.stack(graph_features, dim=0)
        
        # Final prediction (simplified)
        out = self.mlp(graph_features)
        
        return out
        
    def enable_oversquashing_tracking(self):
        """Enable collection of edge weights for oversquashing analysis"""
        self.collect_oversquashing_metrics = True
        print("Oversquashing tracking enabled")
        
    def disable_oversquashing_tracking(self):
        """Disable collection of edge weights to save memory"""
        self.collect_oversquashing_metrics = False
        self.last_edge_weights = {}
        print("Oversquashing tracking disabled")
        
    def get_final_edge_weights(self, data):
        """
        Get the final edge weights for a specific graph.
        Must be called after forward pass with oversquashing tracking enabled.
        
        Args:
            data: PyG Data object or graph index
            
        Returns:
            Dict with edge weights information or None if not available
        """
        if not self.last_edge_weights:
            print("No edge weights available. Make sure to call forward with tracking enabled.")
            return None
            
        if isinstance(data, int):
            # If data is an integer, assume it's the graph index
            graph_idx = data
        else:
            # Extract graph index from batch
            if hasattr(data, 'batch'):
                # Single graph case
                if data.batch.max().item() == 0:
                    graph_idx = 0
                else:
                    # Multiple graphs case - not supported in this simplified implementation
                    print("Cannot extract weights for multiple graphs at once")
                    return None
            else:
                # Assume it's a single graph without batch information
                graph_idx = 0
        
        # Return stored weights for the graph
        if graph_idx in self.last_edge_weights:
            print(f"Successfully retrieved weights for graph {graph_idx}")
            return self.last_edge_weights[graph_idx]
        else:
            print(f"No weights stored for graph index {graph_idx}")
            return None
