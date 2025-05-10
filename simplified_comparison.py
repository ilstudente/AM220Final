"""
A simplified comparison script for testing the Cayley graph initialization.
This script doesn't rely on the full IPR-MPNN training pipeline but provides
a basic test of the main initialization methods.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import deque
from datetime import datetime

def simple_cayley_graph(n):
    """
    Simple implementation of Cayley graph generation.
    """
    generators = np.array([
        [[1, 1], [0, 1]],
        [[1, n-1], [0, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [n-1, 1]]])
    ind = 1

    queue = deque([np.array([[1, 0], [0, 1]])])
    nodes = {(1, 0, 0, 1): 0}

    senders = []
    receivers = []

    while queue:
        x = queue.popleft()
        x_flat = (x[0][0], x[0][1], x[1][0], x[1][1])
        ind_x = nodes[x_flat]
        for i in range(4):
            tx = np.matmul(x, generators[i])
            tx = np.mod(tx, n)
            tx_flat = (tx[0][0], tx[0][1], tx[1][0], tx[1][1])
            if tx_flat not in nodes:
                nodes[tx_flat] = ind
                ind += 1
                queue.append(tx)
            ind_tx = nodes[tx_flat]

            senders.append(ind_x)
            receivers.append(ind_tx)
            
    return torch.tensor([senders, receivers]), len(nodes)

def cayley_init_weights(num_base_nodes, num_virtual_nodes, n=2):
    """Initialize edge weights using Cayley graph"""
    # Generate Cayley graph
    edge_index, num_nodes = simple_cayley_graph(n)
    
    # Create edge weight matrix
    edge_weights = torch.zeros(num_base_nodes, num_virtual_nodes)
    
    # Assign weights based on graph structure
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        
        # Check if the edge connects a base node to a virtual node
        if src < num_base_nodes and dst >= num_base_nodes and dst < num_base_nodes + num_virtual_nodes:
            virtual_idx = dst - num_base_nodes
            edge_weights[src, virtual_idx] = 1.0
        elif dst < num_base_nodes and src >= num_base_nodes and src < num_base_nodes + num_virtual_nodes:
            virtual_idx = src - num_base_nodes
            edge_weights[dst, virtual_idx] = 1.0
    
    # Ensure each base node has at least one connection
    row_sums = edge_weights.sum(dim=1)
    zero_rows = torch.where(row_sums == 0)[0]
    
    if len(zero_rows) > 0:
        print(f"{len(zero_rows)} base nodes have no connections. Adding random connections.")
        # Add random connections for nodes with zero connections
        for row in zero_rows:
            virtual_idx = torch.randint(0, num_virtual_nodes, (1,))
            edge_weights[row, virtual_idx] = 1.0
    
    return edge_weights

def uniform_init_weights(num_base_nodes, num_virtual_nodes):
    """Initialize edge weights uniformly"""
    # Create a uniform distribution of weights
    edge_weights = torch.ones(num_base_nodes, num_virtual_nodes) / num_virtual_nodes
    return edge_weights

def visualize_edge_weights(uniform_weights, cayley_weights, save_path):
    """Visualize the edge weight matrices"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot uniform initialization
    im1 = ax1.imshow(uniform_weights.numpy(), cmap='Blues')
    ax1.set_title('Uniform Initialization')
    ax1.set_xlabel('Virtual Nodes')
    ax1.set_ylabel('Base Nodes')
    fig.colorbar(im1, ax=ax1)
    
    # Plot Cayley initialization
    im2 = ax2.imshow(cayley_weights.numpy(), cmap='Blues')
    ax2.set_title('Cayley Initialization')
    ax2.set_xlabel('Virtual Nodes')
    ax2.set_ylabel('Base Nodes')
    fig.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Visualization saved to {save_path}")

def compute_statistics(uniform_weights, cayley_weights):
    """Compute some statistics for comparison"""
    stats = {
        "uniform": {
            "mean": uniform_weights.mean().item(),
            "std": uniform_weights.std().item(),
            "min": uniform_weights.min().item(),
            "max": uniform_weights.max().item(),
            "nonzero": (uniform_weights > 0).sum().item(),
            "sparsity": 1.0 - (uniform_weights > 0).sum().item() / uniform_weights.numel()
        },
        "cayley": {
            "mean": cayley_weights.mean().item(),
            "std": cayley_weights.std().item(),
            "min": cayley_weights.min().item(),
            "max": cayley_weights.max().item(),
            "nonzero": (cayley_weights > 0).sum().item(),
            "sparsity": 1.0 - (cayley_weights > 0).sum().item() / cayley_weights.numel()
        }
    }
    
    return stats

def main():
    print("Simplified Cayley vs. Uniform Initialization Comparison")
    
    # Set up output directory
    output_dir = "simplified_comparison"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Define test parameters
    num_base_nodes = 20
    num_virtual_nodes = 10
    
    print(f"Testing with {num_base_nodes} base nodes and {num_virtual_nodes} virtual nodes")
    
    # Generate weights using both methods
    uniform_weights = uniform_init_weights(num_base_nodes, num_virtual_nodes)
    cayley_weights = cayley_init_weights(num_base_nodes, num_virtual_nodes)
    
    # Visualize the weights
    visualize_edge_weights(
        uniform_weights, 
        cayley_weights, 
        save_path=f"{output_dir}/weight_visualization_{timestamp}.png"
    )
    
    # Compute statistics
    stats = compute_statistics(uniform_weights, cayley_weights)
    
    # Print statistics
    print("\nStatistics:")
    print("Uniform Initialization:")
    for key, value in stats["uniform"].items():
        print(f"  {key}: {value}")
    
    print("\nCayley Initialization:")
    for key, value in stats["cayley"].items():
        print(f"  {key}: {value}")
    
    # Save statistics
    with open(f"{output_dir}/statistics_{timestamp}.txt", "w") as f:
        f.write("Uniform Initialization:\n")
        for key, value in stats["uniform"].items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nCayley Initialization:\n")
        for key, value in stats["cayley"].items():
            f.write(f"  {key}: {value}\n")
    
    print(f"\nResults saved to {output_dir}/")

if __name__ == "__main__":
    main()
