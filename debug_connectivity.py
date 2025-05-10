"""
Debug script to compare connectivity patterns between Cayley and uniform initialization.
"""

import torch
from models.cayley_utils import calculate_optimal_virtual_nodes, cayley_initialize_edge_weight
import numpy as np
import matplotlib.pyplot as plt
from custom_aligned_comparison import IPRMPNNModel

def print_connectivity_stats(num_base_nodes=20):
    """
    Print statistics about the connectivity patterns for a given number of base nodes.
    
    Args:
        num_base_nodes: Number of base nodes to test with
    """
    # Calculate optimal number of virtual nodes
    num_virtual_nodes, cayley_n = calculate_optimal_virtual_nodes(
        num_base_nodes=num_base_nodes, 
        verbose=True
    )
    
    print(f"\nConnectivity stats for {num_base_nodes} base nodes and {num_virtual_nodes} virtual nodes:")
    
    # Get Cayley edge weights
    cayley_weights = cayley_initialize_edge_weight(
        num_base_nodes=num_base_nodes, 
        num_virtual_nodes=num_virtual_nodes,
        cayley_n=cayley_n,
        verbose=True
    )
    
    # Create uniform edge weights
    uniform_weights = torch.ones(num_base_nodes, num_virtual_nodes) / num_virtual_nodes
    
    # Count non-zero connections for Cayley
    cayley_nonzero = (cayley_weights > 0).sum().item()
    cayley_connections_per_base = (cayley_weights > 0).sum(dim=1).tolist()
    cayley_connections_per_virtual = (cayley_weights > 0).sum(dim=0).tolist()
    
    # Count non-zero connections for uniform
    uniform_nonzero = (uniform_weights > 0).sum().item()
    uniform_connections_per_base = (uniform_weights > 0).sum(dim=1).tolist()
    uniform_connections_per_virtual = (uniform_weights > 0).sum(dim=0).tolist()
    
    # Print stats
    print(f"Cayley: Total connections: {cayley_nonzero} ({cayley_nonzero/(num_base_nodes*num_virtual_nodes)*100:.2f}% of possible connections)")
    print(f"Uniform: Total connections: {uniform_nonzero} ({uniform_nonzero/(num_base_nodes*num_virtual_nodes)*100:.2f}% of possible connections)")
    
    print(f"\nCayley: Avg connections per base node: {np.mean(cayley_connections_per_base):.2f}")
    print(f"Uniform: Avg connections per base node: {np.mean(uniform_connections_per_base):.2f}")
    
    print(f"\nCayley: Avg connections per virtual node: {np.mean(cayley_connections_per_virtual):.2f}")
    print(f"Uniform: Avg connections per virtual node: {np.mean(uniform_connections_per_virtual):.2f}")

def simulate_learning(num_base_nodes=20, hidden_dim=64, num_classes=2, num_steps=10):
    """
    Simulate the learning of edge weights with a toy example.
    
    Args:
        num_base_nodes: Number of base nodes to test with
        hidden_dim: Hidden dimension size
        num_classes: Number of output classes
        num_steps: Number of training steps
    """
    print(f"\nSimulating edge weight learning for {num_base_nodes} base nodes:")
    
    # Create dummy features and labels
    x = torch.randn(num_base_nodes, 10)  # 10-dim features
    y = torch.randint(0, num_classes, (1,))
    
    # Create dummy batch info
    edge_index = torch.zeros(2, 0).long()  # Empty edge index
    batch = torch.zeros(num_base_nodes).long()  # All nodes are in batch 0
    
    # Create a Data object
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, batch=batch, y=y)
    
    # Create both model types
    uniform_model = IPRMPNNModel(
        input_dim=10,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        edge_init_type='uniform'
    )
    
    cayley_model = IPRMPNNModel(
        input_dim=10,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        edge_init_type='cayley'
    )
    
    # Initial forward pass to create edge weights
    _ = uniform_model(data)
    _ = cayley_model(data)
    
    # Get the graph ID
    graph_id = f"{num_base_nodes}_0"
    
    # Get initial weights
    uniform_init = uniform_model.graph_edge_weights[graph_id].clone().detach()
    cayley_init = cayley_model.graph_edge_weights[graph_id].clone().detach()
    
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
    
    # Calculate nonzero ratios
    uniform_init_nonzero = (uniform_init > 0.01).float().mean().item() * 100
    uniform_final_nonzero = (uniform_final > 0.01).float().mean().item() * 100
    
    cayley_init_nonzero = (cayley_init > 0.01).float().mean().item() * 100
    cayley_final_nonzero = (cayley_final > 0.01).float().mean().item() * 100
    
    print("\nConnectivity changes after learning:")
    print(f"Uniform: Initial nonzero: {uniform_init_nonzero:.2f}%, Final nonzero: {uniform_final_nonzero:.2f}%")
    print(f"Cayley: Initial nonzero: {cayley_init_nonzero:.2f}%, Final nonzero: {cayley_final_nonzero:.2f}%")
    
    # Visualize the weight changes
    plt.figure(figsize=(15, 8))
    
    # Plot uniform weights before and after
    plt.subplot(2, 2, 1)
    plt.imshow(uniform_init, cmap='viridis')
    plt.title(f"Uniform Initialization\n({uniform_init_nonzero:.1f}% nonzero)")
    plt.colorbar()
    
    plt.subplot(2, 2, 2)
    plt.imshow(uniform_final, cmap='viridis')
    plt.title(f"Uniform After Learning\n({uniform_final_nonzero:.1f}% nonzero)")
    plt.colorbar()
    
    # Plot cayley weights before and after
    plt.subplot(2, 2, 3)
    plt.imshow(cayley_init, cmap='viridis')
    plt.title(f"Cayley Initialization\n({cayley_init_nonzero:.1f}% nonzero)")
    plt.colorbar()
    
    plt.subplot(2, 2, 4)
    plt.imshow(cayley_final, cmap='viridis')
    plt.title(f"Cayley After Learning\n({cayley_final_nonzero:.1f}% nonzero)")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f"connectivity_learning_n{num_base_nodes}.png")
    print(f"Visualization saved to connectivity_learning_n{num_base_nodes}.png")

if __name__ == "__main__":
    # Test with different numbers of base nodes
    print_connectivity_stats(20)  # Small graph
    print_connectivity_stats(40)  # Medium graph
    print_connectivity_stats(100)  # Large graph
    
    # Simulate learning
    simulate_learning(num_base_nodes=20, num_steps=20)
    simulate_learning(num_base_nodes=40, num_steps=20)
    simulate_learning(num_base_nodes=100, num_steps=20)  # Large graph
