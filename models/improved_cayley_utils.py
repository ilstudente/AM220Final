"""
An improved version of enhanced_cayley_utils.py with better preservation of
Cayley graph structure while ensuring proper k-connectivity.
"""

import torch
import numpy as np
from collections import deque
import torch.nn.functional as F
from models.cayley_utils import get_cayley_graph, cayley_graph_size, get_cayley_n, calculate_optimal_virtual_nodes

def improved_cayley_initialize_edge_weight(num_base_nodes, num_virtual_nodes, cayley_n=None, 
                                          high_value=10.0, low_value=0.01, contrast_factor=0.95, verbose=True):
    """
    Improved initialization of edge weights between base and virtual nodes using the Cayley graph expansion.
    This version preserves the Cayley structure more strongly while ensuring every node has proper connectivity.
    
    Args:
        num_base_nodes (int): Number of base (original) nodes.
        num_virtual_nodes (int): Number of virtual (centroid) nodes.
        cayley_n (int, optional): The Cayley graph parameter n to use. If None, will be calculated.
        high_value (float): The weight value for Cayley graph edges.
        low_value (float): The small positive weight value for non-Cayley edges.
        contrast_factor (float): Fraction of weight mass to allocate to Cayley edges (0.95 = 95% of weight mass).
        verbose (bool): Whether to print warnings.
        
    Returns:
        torch.Tensor: Edge weights tensor of shape [num_base_nodes, num_virtual_nodes].
    """
    # Use provided cayley_n or calculate one if not provided
    if cayley_n is None:
        # Find the Cayley graph order needed to cover the number of nodes
        cayley_n = get_cayley_n(num_base_nodes + num_virtual_nodes)
    
    # Generate the Cayley graph
    edge_index, cayley_num_nodes = get_cayley_graph(cayley_n)
    
    # Check for mismatch between Cayley graph size and required nodes
    if cayley_num_nodes != num_base_nodes + num_virtual_nodes and verbose:
        print(f"Warning: Cayley graph has {cayley_num_nodes} nodes, more than the required {num_base_nodes + num_virtual_nodes}")
    
    # Initialize all edges with a very small positive value
    # This ensures all edges start with some small gradient
    edge_weights = torch.ones(num_base_nodes, num_virtual_nodes) * low_value
    
    # Track Cayley connections for each base node
    cayley_connections = [[] for _ in range(num_base_nodes)]
    
    # For each edge in the Cayley graph that connects a base node to a virtual node
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        
        # Check if the edge connects a base node to a virtual node
        if src < num_base_nodes and dst >= num_base_nodes and dst < num_base_nodes + num_virtual_nodes:
            # Map the virtual node index to the range [0, num_virtual_nodes-1]
            virtual_idx = dst - num_base_nodes
            cayley_connections[src].append(virtual_idx)
        elif dst < num_base_nodes and src >= num_base_nodes and src < num_base_nodes + num_virtual_nodes:
            # Map the virtual node index to the range [0, num_virtual_nodes-1]
            virtual_idx = src - num_base_nodes
            cayley_connections[dst].append(virtual_idx)
    
    # Process each base node to ensure it's properly connected while preserving Cayley structure
    for i in range(num_base_nodes):
        connections = cayley_connections[i]
        
        if connections:
            # Node has Cayley connections - give them most of the weight mass
            for v_idx in connections:
                edge_weights[i, v_idx] = high_value
                
            # Normalize to make Cayley connections have 'contrast_factor' of the total weight
            cayley_sum = len(connections) * high_value
            non_cayley_sum = (num_virtual_nodes - len(connections)) * low_value
            total_sum = cayley_sum + non_cayley_sum
            
            # Calculate scaling factors to achieve desired contrast
            target_cayley_sum = total_sum * contrast_factor
            target_non_cayley_sum = total_sum * (1 - contrast_factor)
            
            cayley_scale = target_cayley_sum / cayley_sum if cayley_sum > 0 else 1.0
            non_cayley_scale = target_non_cayley_sum / non_cayley_sum if non_cayley_sum > 0 else 1.0
            
            # Apply scaling
            for v_idx in range(num_virtual_nodes):
                if v_idx in connections:
                    edge_weights[i, v_idx] *= cayley_scale
                else:
                    edge_weights[i, v_idx] *= non_cayley_scale
        else:
            # Node has no Cayley connections - add stronger connections to random virtual nodes
            # while still maintaining a clear distinction between strong and weak connections
            num_random = min(3, num_virtual_nodes)
            random_indices = torch.randperm(num_virtual_nodes)[:num_random]
            
            for v_idx in random_indices:
                edge_weights[i, v_idx] = high_value
                
            # Apply the same contrast factor logic as above
            random_sum = num_random * high_value
            other_sum = (num_virtual_nodes - num_random) * low_value
            total_sum = random_sum + other_sum
            
            target_random_sum = total_sum * contrast_factor
            target_other_sum = total_sum * (1 - contrast_factor)
            
            random_scale = target_random_sum / random_sum if random_sum > 0 else 1.0
            other_scale = target_other_sum / other_sum if other_sum > 0 else 1.0
            
            for v_idx in range(num_virtual_nodes):
                if v_idx in random_indices:
                    edge_weights[i, v_idx] *= random_scale
                else:
                    edge_weights[i, v_idx] *= other_scale
    
    # Normalize the weights for each base node to sum to 1
    row_sums = edge_weights.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    edge_weights = edge_weights / row_sums
    
    return edge_weights


def structure_aware_topk(edge_weights, k, high_contrast=True, structure_factor=0.8):
    """
    Structure-aware top-k pruning that better preserves the important structural patterns.
    This pruning approach tries to keep high-weight edges while ensuring exactly k connections per node.
    
    Args:
        edge_weights (torch.Tensor): Edge weights tensor of shape [num_base_nodes, num_virtual_nodes].
        k (int): The number of connections to keep per base node.
        high_contrast (bool): Whether to apply high contrast to the final weights (makes top weights stand out more).
        structure_factor (float): How much to emphasize existing weight structure (0-1).
        
    Returns:
        torch.Tensor: Processed edge weights with exactly k connections per base node.
    """
    num_nodes, num_virtual = edge_weights.shape
    result = torch.zeros_like(edge_weights)
    
    for i in range(num_nodes):
        # Get node's weights
        node_weights = edge_weights[i]
        
        # Always take the top-k values
        values, indices = torch.topk(node_weights, k=min(k, num_virtual))
        
        # Set the selected connections in the result tensor
        result[i, indices] = values
        
        if high_contrast and k > 1:
            # Apply contrast enhancement to make the weight distribution more distinct
            # This helps the model focus on the most important connections
            result_row = result[i, indices]
            min_val = result_row.min()
            max_val = result_row.max()
            
            if max_val > min_val:  # Avoid division by zero
                # Apply softmax with temperature to increase contrast
                temp = 0.1
                logits = (result_row - min_val) / (max_val - min_val) / temp
                enhanced = F.softmax(logits, dim=0)
                
                # Mix original normalized weights with enhanced version
                normalized = result_row / result_row.sum()
                result[i, indices] = normalized * (1 - structure_factor) + enhanced * structure_factor
            else:
                # If all values are equal, just normalize
                result[i, indices] = 1.0 / len(indices)
        else:
            # Just normalize the weights to sum to 1
            row_sum = result[i].sum()
            if row_sum > 0:
                result[i] = result[i] / row_sum
    
    return result
