"""
Test script to compare the standard Cayley initialization with the enhanced version.
"""

import torch
import matplotlib.pyplot as plt
from models.cayley_utils import cayley_initialize_edge_weight, calculate_optimal_virtual_nodes
from models.enhanced_cayley_utils import enhanced_cayley_initialize_edge_weight, force_topk_connections
import numpy as np

def compare_initializations(num_base_nodes=20, k=5):
    """
    Compare standard and enhanced Cayley initializations and their behavior with top-k pruning.
    
    Args:
        num_base_nodes (int): Number of base nodes to test with
        k (int): Number of top connections to keep per base node
    """
    print(f"\nComparing initializations for {num_base_nodes} base nodes with top-{k} pruning:")
    
    # Calculate optimal number of virtual nodes
    num_virtual_nodes, cayley_n = calculate_optimal_virtual_nodes(
        num_base_nodes=num_base_nodes, 
        verbose=True
    )
    
    # Standard Cayley initialization
    standard_weights = cayley_initialize_edge_weight(
        num_base_nodes=num_base_nodes, 
        num_virtual_nodes=num_virtual_nodes,
        cayley_n=cayley_n,
        verbose=True
    )
    
    # Enhanced Cayley initialization
    enhanced_weights = enhanced_cayley_initialize_edge_weight(
        num_base_nodes=num_base_nodes, 
        num_virtual_nodes=num_virtual_nodes,
        cayley_n=cayley_n,
        high_value=5.0,
        low_value=0.1,
        verbose=True
    )
    
    # Uniform initialization for comparison
    uniform_weights = torch.ones(num_base_nodes, num_virtual_nodes) / num_virtual_nodes
    
    # Apply top-k pruning
    standard_topk = force_topk_connections(standard_weights, k)
    enhanced_topk = force_topk_connections(enhanced_weights, k)
    uniform_topk = force_topk_connections(uniform_weights, k)
    
    # Count non-zero connections
    standard_nonzero = (standard_weights > 0).sum().item()
    enhanced_nonzero = (enhanced_weights > 0).sum().item()
    uniform_nonzero = (uniform_weights > 0).sum().item()
    
    standard_topk_nonzero = (standard_topk > 0).sum().item()
    enhanced_topk_nonzero = (enhanced_topk > 0).sum().item()
    uniform_topk_nonzero = (uniform_topk > 0).sum().item()
    
    # Calculate percentages
    total_connections = num_base_nodes * num_virtual_nodes
    standard_pct = standard_nonzero / total_connections * 100
    enhanced_pct = enhanced_nonzero / total_connections * 100
    uniform_pct = uniform_nonzero / total_connections * 100
    
    standard_topk_pct = standard_topk_nonzero / total_connections * 100
    enhanced_topk_pct = enhanced_topk_nonzero / total_connections * 100
    uniform_topk_pct = uniform_topk_nonzero / total_connections * 100
    
    # Expected number of connections with top-k pruning
    expected_topk = num_base_nodes * k
    expected_topk_pct = expected_topk / total_connections * 100
    
    print("\nInitial connectivity statistics:")
    print(f"Total possible connections: {total_connections}")
    print(f"Standard Cayley: {standard_nonzero} connections ({standard_pct:.2f}%)")
    print(f"Enhanced Cayley: {enhanced_nonzero} connections ({enhanced_pct:.2f}%)")
    print(f"Uniform: {uniform_nonzero} connections ({uniform_pct:.2f}%)")
    
    print("\nAfter top-k pruning:")
    print(f"Expected with top-{k}: {expected_topk} connections ({expected_topk_pct:.2f}%)")
    print(f"Standard Cayley: {standard_topk_nonzero} connections ({standard_topk_pct:.2f}%)")
    print(f"Enhanced Cayley: {enhanced_topk_nonzero} connections ({enhanced_topk_pct:.2f}%)")
    print(f"Uniform: {uniform_topk_nonzero} connections ({uniform_topk_pct:.2f}%)")
    
    # Count how many base nodes have exactly k connections
    standard_exact_k = sum([(standard_topk[i] > 0).sum().item() == k for i in range(num_base_nodes)])
    enhanced_exact_k = sum([(enhanced_topk[i] > 0).sum().item() == k for i in range(num_base_nodes)])
    uniform_exact_k = sum([(uniform_topk[i] > 0).sum().item() == k for i in range(num_base_nodes)])
    
    print(f"\nBase nodes with exactly {k} connections:")
    print(f"Standard Cayley: {standard_exact_k}/{num_base_nodes} ({standard_exact_k/num_base_nodes*100:.2f}%)")
    print(f"Enhanced Cayley: {enhanced_exact_k}/{num_base_nodes} ({enhanced_exact_k/num_base_nodes*100:.2f}%)")
    print(f"Uniform: {uniform_exact_k}/{num_base_nodes} ({uniform_exact_k/num_base_nodes*100:.2f}%)")
    
    # Visualize initializations and pruned weights
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 2, 1)
    plt.imshow(standard_weights, cmap='viridis')
    plt.title(f"Standard Cayley Initialization\n({standard_pct:.1f}% non-zero)")
    plt.colorbar()
    
    plt.subplot(3, 2, 2)
    plt.imshow(standard_topk, cmap='viridis')
    plt.title(f"Standard Cayley with Top-{k}\n({standard_topk_pct:.1f}% non-zero)")
    plt.colorbar()
    
    plt.subplot(3, 2, 3)
    plt.imshow(enhanced_weights, cmap='viridis')
    plt.title(f"Enhanced Cayley Initialization\n({enhanced_pct:.1f}% non-zero)")
    plt.colorbar()
    
    plt.subplot(3, 2, 4)
    plt.imshow(enhanced_topk, cmap='viridis')
    plt.title(f"Enhanced Cayley with Top-{k}\n({enhanced_topk_pct:.1f}% non-zero)")
    plt.colorbar()
    
    plt.subplot(3, 2, 5)
    plt.imshow(uniform_weights, cmap='viridis')
    plt.title(f"Uniform Initialization\n({uniform_pct:.1f}% non-zero)")
    plt.colorbar()
    
    plt.subplot(3, 2, 6)
    plt.imshow(uniform_topk, cmap='viridis')
    plt.title(f"Uniform with Top-{k}\n({uniform_topk_pct:.1f}% non-zero)")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f"cayley_initialization_comparison_n{num_base_nodes}_k{k}.png")
    print(f"Visualization saved to cayley_initialization_comparison_n{num_base_nodes}_k{k}.png")


if __name__ == "__main__":
    # Test with different sizes and k values
    compare_initializations(num_base_nodes=20, k=3)
    compare_initializations(num_base_nodes=30, k=5)
    compare_initializations(num_base_nodes=50, k=8)
