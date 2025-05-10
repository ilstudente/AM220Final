"""
Test if using top-k connectivity forces both initialization approaches to have 
the same number of edges after learning.
"""

import torch
from torch_geometric.data import Data
from custom_aligned_comparison import IPRMPNNModel
import matplotlib.pyplot as plt

def test_topk_connectivity(num_base_nodes=30, hidden_dim=64, num_classes=2, num_steps=20, top_k=5):
    """
    Test if using top-k connectivity forces both initialization approaches to have 
    the same number of edges after learning.
    
    Args:
        num_base_nodes: Number of base nodes to test with
        hidden_dim: Hidden dimension size
        num_classes: Number of output classes
        num_steps: Number of training steps
        top_k: Number of top connections per base node to keep
    """
    print(f"\nTesting top-{top_k} connectivity with {num_base_nodes} base nodes:")
    
    # Create dummy features and labels
    x = torch.randn(num_base_nodes, 10)  # 10-dim features
    y = torch.randint(0, num_classes, (1,))
    
    # Create dummy batch info
    edge_index = torch.zeros(2, 0).long()  # Empty edge index
    batch = torch.zeros(num_base_nodes).long()  # All nodes are in batch 0
    
    # Create a Data object
    data = Data(x=x, edge_index=edge_index, batch=batch, y=y)
    
    # Create both model types WITH top-k connectivity
    uniform_model = IPRMPNNModel(
        input_dim=10,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        edge_init_type='uniform',
        top_k=top_k
    )
    
    cayley_model = IPRMPNNModel(
        input_dim=10,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        edge_init_type='cayley',
        top_k=top_k
    )
    
    # Initial forward pass to create edge weights
    _ = uniform_model(data)
    _ = cayley_model(data)
    
    # Get the graph ID
    graph_id = f"{num_base_nodes}_0"
    
    # Get initial weights
    uniform_init = uniform_model.graph_edge_weights[graph_id].clone().detach()
    cayley_init = cayley_model.graph_edge_weights[graph_id].clone().detach()
    
    # Get virtual node count
    num_virtual_nodes = uniform_init.shape[1]
    print(f"Number of virtual nodes: {num_virtual_nodes}")
    
    # Create optimizers
    uniform_optimizer = torch.optim.Adam(uniform_model.parameters(), lr=0.01)
    cayley_optimizer = torch.optim.Adam(cayley_model.parameters(), lr=0.01)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    print("Training models...")
    for step in range(num_steps):
        # Uniform model step
        uniform_optimizer.zero_grad()
        uniform_out = uniform_model(data)
        uniform_loss = criterion(uniform_out, y)
        uniform_loss.backward()
        uniform_optimizer.step()
        
        # Cayley model step
        cayley_optimizer.zero_grad()
        cayley_out = cayley_model(data)
        cayley_loss = criterion(cayley_out, y)
        cayley_loss.backward()
        cayley_optimizer.step()
        
        if (step + 1) % 5 == 0:
            print(f"Step {step+1}/{num_steps}")
    
    # Get final weights
    uniform_final = uniform_model.graph_edge_weights[graph_id].clone().detach()
    cayley_final = cayley_model.graph_edge_weights[graph_id].clone().detach()
    
    # Apply top-k pruning to get final effective weights
    def apply_topk(weights, k):
        _, top_indices = torch.topk(weights, k=k, dim=1)
        mask = torch.zeros_like(weights)
        for i in range(weights.shape[0]):
            mask[i, top_indices[i]] = 1.0
        return weights * mask
    
    uniform_final_topk = apply_topk(uniform_final, top_k)
    cayley_final_topk = apply_topk(cayley_final, top_k)
    
    # Count non-zero connections after applying top-k
    uniform_nonzero_topk = (uniform_final_topk > 0).sum().item()
    cayley_nonzero_topk = (cayley_final_topk > 0).sum().item()
    
    # Count total possible connections
    total_possible = num_base_nodes * num_virtual_nodes
    
    # Expected number of connections with top-k
    expected_connections = num_base_nodes * top_k
    
    print("\nConnectivity after top-k pruning:")
    print(f"Total possible connections: {total_possible}")
    print(f"Expected connections with top-{top_k}: {expected_connections}")
    print(f"Uniform: {uniform_nonzero_topk} connections ({uniform_nonzero_topk/total_possible*100:.2f}%)")
    print(f"Cayley: {cayley_nonzero_topk} connections ({cayley_nonzero_topk/total_possible*100:.2f}%)")
    
    # Visualize the weight changes
    plt.figure(figsize=(15, 8))
    
    # Plot uniform weights before and after
    plt.subplot(2, 2, 1)
    plt.imshow(uniform_init, cmap='viridis')
    plt.title(f"Uniform Initialization")
    plt.colorbar()
    
    plt.subplot(2, 2, 2)
    plt.imshow(uniform_final_topk, cmap='viridis')
    plt.title(f"Uniform After Learning\nwith top-{top_k} pruning")
    plt.colorbar()
    
    # Plot cayley weights before and after
    plt.subplot(2, 2, 3)
    plt.imshow(cayley_init, cmap='viridis')
    plt.title(f"Cayley Initialization")
    plt.colorbar()
    
    plt.subplot(2, 2, 4)
    plt.imshow(cayley_final_topk, cmap='viridis')
    plt.title(f"Cayley After Learning\nwith top-{top_k} pruning")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f"topk_{top_k}_connectivity_n{num_base_nodes}.png")
    print(f"Visualization saved to topk_{top_k}_connectivity_n{num_base_nodes}.png")

if __name__ == "__main__":
    # Test with different top-k values
    test_topk_connectivity(num_base_nodes=30, top_k=3)
    test_topk_connectivity(num_base_nodes=30, top_k=5)
    test_topk_connectivity(num_base_nodes=50, top_k=8)
