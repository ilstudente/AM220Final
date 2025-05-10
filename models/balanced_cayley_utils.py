"""
A balanced version of Cayley initialization that preserves some structure
while ensuring stable learning dynamics.
"""

import torch
import numpy as np
from collections import deque
import torch.nn.functional as F
from models.cayley_utils import get_cayley_graph, cayley_graph_size, get_cayley_n, calculate_optimal_virtual_nodes

def balanced_cayley_initialize_edge_weight(num_base_nodes, num_virtual_nodes, cayley_n=None, 
                                          high_value=2.0, low_value=0.2, verbose=True):
    """
    Balanced Cayley initialization with moderate contrast between Cayley and non-Cayley edges.
    This creates a gentler initialization that still preserves some Cayley structure
    but allows for more flexible learning.
    
    Args:
        num_base_nodes (int): Number of base (original) nodes.
        num_virtual_nodes (int): Number of virtual (centroid) nodes.
        cayley_n (int, optional): The Cayley graph parameter n to use. If None, will be calculated.
        high_value (float): The weight value for Cayley graph edges.
        low_value (float): The weight value for non-Cayley edges.
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
    
    # Initialize all edges with the low value
    edge_weights = torch.ones(num_base_nodes, num_virtual_nodes) * low_value
    
    # Keep track of Cayley edges for each base node
    cayley_connections = [[] for _ in range(num_base_nodes)]
    
    # For each edge in the Cayley graph that connects a base node to a virtual node
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        
        # Check if the edge connects a base node to a virtual node
        if src < num_base_nodes and dst >= num_base_nodes and dst < num_base_nodes + num_virtual_nodes:
            # Map the virtual node index to the range [0, num_virtual_nodes-1]
            virtual_idx = dst - num_base_nodes
            edge_weights[src, virtual_idx] = high_value
            cayley_connections[src].append(virtual_idx)
        elif dst < num_base_nodes and src >= num_base_nodes and src < num_base_nodes + num_virtual_nodes:
            # Map the virtual node index to the range [0, num_virtual_nodes-1]
            virtual_idx = src - num_base_nodes
            edge_weights[dst, virtual_idx] = high_value
            cayley_connections[dst].append(virtual_idx)
    
    # For nodes without Cayley connections, add random connections
    for i, connections in enumerate(cayley_connections):
        if not connections:
            # Add 2-3 stronger random connections
            num_random = min(3, num_virtual_nodes)
            random_indices = torch.randperm(num_virtual_nodes)[:num_random]
            for v_idx in random_indices:
                edge_weights[i, v_idx] = high_value
    
    # Normalize the weights for each base node to sum to 1
    row_sums = edge_weights.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    edge_weights = edge_weights / row_sums
    
    return edge_weights


def balanced_topk_pruning(edge_weights, k):
    """
    A simpler top-k pruning that preserves the relative weights while ensuring exactly k connections.
    This implementation avoids in-place operations to be compatible with autograd.
    
    Args:
        edge_weights (torch.Tensor): Edge weights tensor of shape [num_base_nodes, num_virtual_nodes].
        k (int): The number of connections to keep per base node.
        
    Returns:
        torch.Tensor: Pruned edge weights with exactly k connections per base node.
    """
    num_nodes, num_virtual = edge_weights.shape
    # Always create a new tensor to avoid in-place operations
    result = torch.zeros_like(edge_weights)
    
    for i in range(num_nodes):
        # Get the top-k indices for this node
        _, indices = torch.topk(edge_weights[i], k=min(k, num_virtual))
        
        # Keep the original weight values for selected connections
        # This is not an in-place operation as we're writing to a new tensor
        for idx in indices:
            result[i, idx] = edge_weights[i, idx]
    
    # Renormalize the weights for each base node to sum to 1
    row_sums = result.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    result = result / row_sums
    
    return result
