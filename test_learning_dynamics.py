"""
Test script to verify that the improved Cayley initialization performs at least as well as uniform.
This script isolates the key components to better understand the learning dynamics.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from models.cayley_utils import cayley_initialize_edge_weight, calculate_optimal_virtual_nodes
from models.enhanced_cayley_utils import enhanced_cayley_initialize_edge_weight, force_topk_connections
from models.improved_cayley_utils import improved_cayley_initialize_edge_weight, structure_aware_topk

def simulate_learning(edge_weights, num_steps=100, learning_rate=0.1):
    """
    Simulate the learning process on edge weights.
    
    This simulates how the edge weights would change during training by applying
    random gradients and updating the weights, similar to what happens in the real model.
    """
    # Make a copy of the weights that we can modify
    weights = edge_weights.clone().requires_grad_(True)
    
    # Track weight evolution
    weight_history = [weights.detach().clone()]
    
    # Simulate learning process
    for step in range(num_steps):
        # Random "task" - simulate a learning signal
        # In reality, this would come from the loss function
        target_distribution = torch.rand_like(weights)
        target_distribution = target_distribution / target_distribution.sum(dim=1, keepdim=True)
        
        # Simple loss: MSE between current weights and target
        loss = ((weights - target_distribution) ** 2).mean()
        
        # Compute gradients
        loss.backward()
        
        # Update weights (gradient descent)
        with torch.no_grad():
            weights -= learning_rate * weights.grad
            
            # Ensure weights are non-negative
            weights.clamp_(min=0.0)
            
            # Normalize weights to sum to 1 for each base node
            row_sums = weights.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1.0  # Avoid division by zero
            weights.data = weights.data / row_sums
        
        # Reset gradients
        weights.grad.zero_()
        
        # Store current weights
        weight_history.append(weights.detach().clone())
    
    return weight_history

def basic_learning_test(num_base_nodes=30, k=5):
    """
    Test basic learning behavior with different initializations.
    """
    print(f"Testing learning dynamics with {num_base_nodes} base nodes and k={k}")
    
    # Calculate optimal number of virtual nodes
    num_virtual_nodes, cayley_n = calculate_optimal_virtual_nodes(
        num_base_nodes=num_base_nodes, 
        verbose=False
    )
    
    # Standard Cayley initialization
    cayley_weights = cayley_initialize_edge_weight(
        num_base_nodes=num_base_nodes, 
        num_virtual_nodes=num_virtual_nodes,
        cayley_n=cayley_n,
        verbose=False
    )
    
    # Enhanced Cayley initialization
    enhanced_weights = enhanced_cayley_initialize_edge_weight(
        num_base_nodes=num_base_nodes, 
        num_virtual_nodes=num_virtual_nodes,
        cayley_n=cayley_n,
        high_value=5.0,
        low_value=0.1,
        verbose=False
    )
    
    # Improved Cayley initialization
    improved_weights = improved_cayley_initialize_edge_weight(
        num_base_nodes=num_base_nodes, 
        num_virtual_nodes=num_virtual_nodes,
        cayley_n=cayley_n,
        high_value=10.0,
        low_value=0.01,
        contrast_factor=0.95,
        verbose=False
    )
    
    # Improved Cayley with lower contrast
    balanced_weights = improved_cayley_initialize_edge_weight(
        num_base_nodes=num_base_nodes, 
        num_virtual_nodes=num_virtual_nodes,
        cayley_n=cayley_n,
        high_value=2.0,
        low_value=0.1,
        contrast_factor=0.7,  # Less extreme contrast
        verbose=False
    )
    
    # Uniform initialization
    uniform_weights = torch.ones(num_base_nodes, num_virtual_nodes) / num_virtual_nodes
    
    # Apply top-k pruning to all
    if k is not None:
        cayley_topk = force_topk_connections(cayley_weights.clone(), k)
        enhanced_topk = force_topk_connections(enhanced_weights.clone(), k)
        improved_topk = structure_aware_topk(improved_weights.clone(), k)
        balanced_topk = structure_aware_topk(balanced_weights.clone(), k, high_contrast=False)
        uniform_topk = force_topk_connections(uniform_weights.clone(), k)
    else:
        cayley_topk = cayley_weights
        enhanced_topk = enhanced_weights
        improved_topk = improved_weights
        balanced_topk = balanced_weights
        uniform_topk = uniform_weights
    
    # Simulate the learning process for each initialization
    print("Simulating learning process...")
    cayley_history = simulate_learning(cayley_topk)
    enhanced_history = simulate_learning(enhanced_topk)
    improved_history = simulate_learning(improved_topk)
    balanced_history = simulate_learning(balanced_topk)
    uniform_history = simulate_learning(uniform_topk)
    
    # Analyze learning dynamics
    # Calculate change in weights over time
    cayley_changes = [torch.norm(cayley_history[i] - cayley_history[i-1]).item() for i in range(1, len(cayley_history))]
    enhanced_changes = [torch.norm(enhanced_history[i] - enhanced_history[i-1]).item() for i in range(1, len(enhanced_history))]
    improved_changes = [torch.norm(improved_history[i] - improved_history[i-1]).item() for i in range(1, len(improved_history))]
    balanced_changes = [torch.norm(balanced_history[i] - balanced_history[i-1]).item() for i in range(1, len(balanced_history))]
    uniform_changes = [torch.norm(uniform_history[i] - uniform_history[i-1]).item() for i in range(1, len(uniform_history))]
    
    # Plot the learning dynamics
    plt.figure(figsize=(10, 6))
    plt.plot(cayley_changes, label='Standard Cayley')
    plt.plot(enhanced_changes, label='Enhanced Cayley')
    plt.plot(improved_changes, label='Improved Cayley (High Contrast)')
    plt.plot(balanced_changes, label='Improved Cayley (Balanced)')
    plt.plot(uniform_changes, label='Uniform')
    plt.xlabel('Training Step')
    plt.ylabel('Weight Change Magnitude')
    plt.title(f'Learning Dynamics (n={num_base_nodes}, k={k})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate the final average change for the last 25% of steps
    final_quarter = len(cayley_changes) // 4
    avg_cayley = np.mean(cayley_changes[-final_quarter:])
    avg_enhanced = np.mean(enhanced_changes[-final_quarter:])
    avg_improved = np.mean(improved_changes[-final_quarter:])
    avg_balanced = np.mean(balanced_changes[-final_quarter:])
    avg_uniform = np.mean(uniform_changes[-final_quarter:])
    
    print("\nFinal learning rates (average weight change in last 25% of steps):")
    print(f"Standard Cayley: {avg_cayley:.6f}")
    print(f"Enhanced Cayley: {avg_enhanced:.6f}")
    print(f"Improved Cayley (High Contrast): {avg_improved:.6f}")
    print(f"Improved Cayley (Balanced): {avg_balanced:.6f}")
    print(f"Uniform: {avg_uniform:.6f}")
    
    # Save the figure
    plt.savefig(f'learning_dynamics_n{num_base_nodes}_k{k}.png')
    plt.close()
    
    return {
        'standard_cayley': avg_cayley,
        'enhanced_cayley': avg_enhanced,
        'improved_cayley': avg_improved,
        'balanced_cayley': avg_balanced,
        'uniform': avg_uniform
    }

if __name__ == "__main__":
    print("Testing learning dynamics for different initializations...")
    
    # Test with different node counts and k values
    results_30_5 = basic_learning_test(num_base_nodes=30, k=5)
    results_50_8 = basic_learning_test(num_base_nodes=50, k=8)
    results_100_10 = basic_learning_test(num_base_nodes=100, k=10)
    
    print("\nSummary of results:")
    print(f"n=30, k=5: Uniform/Balanced ratio: {results_30_5['uniform'] / results_30_5['balanced_cayley']:.2f}")
    print(f"n=50, k=8: Uniform/Balanced ratio: {results_50_8['uniform'] / results_50_8['balanced_cayley']:.2f}")
    print(f"n=100, k=10: Uniform/Balanced ratio: {results_100_10['uniform'] / results_100_10['balanced_cayley']:.2f}")
