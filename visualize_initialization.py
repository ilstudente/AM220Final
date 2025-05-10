"""
Simple script to visualize uniform vs. Cayley initialization in IPR-MPNN.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.cayley_utils import cayley_initialize_edge_weight
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def uniform_init_weights(num_base_nodes, num_virtual_nodes):
    """Initialize edge weights uniformly"""
    # Create a uniform distribution of weights
    edge_weights = torch.ones(num_base_nodes, num_virtual_nodes) / num_virtual_nodes
    return edge_weights

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create output directory
    output_dir = 'comparison_results/visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define test parameters
    num_base_nodes = 30
    num_virtual_nodes = 15
    
    logger.info(f"Creating visualizations for {num_base_nodes} base nodes and {num_virtual_nodes} virtual nodes")
    
    # Initialize with both methods
    uniform_weights = uniform_init_weights(num_base_nodes, num_virtual_nodes)
    cayley_weights = cayley_initialize_edge_weight(num_base_nodes, num_virtual_nodes)
    
    # Get timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Plot weight matrices
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(uniform_weights.numpy(), cmap='Blues')
    plt.title('Uniform Initialization')
    plt.xlabel('Virtual Nodes')
    plt.ylabel('Base Nodes')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(cayley_weights.numpy(), cmap='Blues')
    plt.title('Cayley Initialization')
    plt.xlabel('Virtual Nodes')
    plt.ylabel('Base Nodes')
    plt.colorbar()
    
    plt.tight_layout()
    weights_file = os.path.join(output_dir, f"weight_visualization_{timestamp}.png")
    plt.savefig(weights_file)
    plt.close()
    logger.info(f"Weight visualization saved to {weights_file}")
    
    # Compute statistics
    uniform_stats = {
        "mean": uniform_weights.mean().item(),
        "std": uniform_weights.std().item(),
        "min": uniform_weights.min().item(),
        "max": uniform_weights.max().item(),
        "nonzero": (uniform_weights > 0).sum().item(),
        "sparsity": 1.0 - (uniform_weights > 0).sum().item() / uniform_weights.numel(),
        "avg_connections_per_base_node": (uniform_weights > 0).sum(dim=1).float().mean().item()
    }
    
    cayley_stats = {
        "mean": cayley_weights.mean().item(),
        "std": cayley_weights.std().item(),
        "min": cayley_weights.min().item(),
        "max": cayley_weights.max().item(),
        "nonzero": (cayley_weights > 0).sum().item(),
        "sparsity": 1.0 - (cayley_weights > 0).sum().item() / cayley_weights.numel(),
        "avg_connections_per_base_node": (cayley_weights > 0).sum(dim=1).float().mean().item()
    }
    
    # Plot statistics as bar charts
    plt.figure(figsize=(15, 10))
    
    # Prepare data for plotting
    stats_to_plot = ["mean", "std", "sparsity", "avg_connections_per_base_node"]
    x = np.arange(len(stats_to_plot))
    width = 0.35
    
    plt.bar(x - width/2, [uniform_stats[k] for k in stats_to_plot], width, label='Uniform')
    plt.bar(x + width/2, [cayley_stats[k] for k in stats_to_plot], width, label='Cayley')
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Comparison of Initialization Statistics')
    plt.xticks(x, stats_to_plot)
    plt.legend()
    
    plt.tight_layout()
    stats_file = os.path.join(output_dir, f"stats_visualization_{timestamp}.png")
    plt.savefig(stats_file)
    plt.close()
    logger.info(f"Statistics visualization saved to {stats_file}")
    
    # Save statistics to file
    with open(os.path.join(output_dir, f"statistics_{timestamp}.txt"), "w") as f:
        f.write("Comparison of Initialization Methods\n")
        f.write("==================================\n\n")
        f.write(f"Base Nodes: {num_base_nodes}\n")
        f.write(f"Virtual Nodes: {num_virtual_nodes}\n\n")
        
        f.write("Uniform Initialization:\n")
        for key, value in uniform_stats.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nCayley Initialization:\n")
        for key, value in cayley_stats.items():
            f.write(f"  {key}: {value}\n")
            
        f.write("\nComparison:\n")
        for key in uniform_stats:
            if key in ["sparsity", "nonzero"]:
                # For these metrics, higher sparsity (lower nonzero) might be better
                diff = cayley_stats[key] - uniform_stats[key]
                f.write(f"  {key} difference (Cayley - Uniform): {diff:.6f}\n")
            else:
                # For other metrics, just report the difference
                diff = cayley_stats[key] - uniform_stats[key]
                f.write(f"  {key} difference (Cayley - Uniform): {diff:.6f}\n")
    
    logger.info("Visualization complete!")

if __name__ == "__main__":
    main()
