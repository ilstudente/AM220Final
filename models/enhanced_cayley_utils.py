"""
An enhanced version of cayley_utils.py with modified initialization to ensure
all edges have non-zero weights, with Cayley edges having higher weights.
"""

import torch
import numpy as np
from collections import deque
import torch.nn.functional as F
from models.cayley_utils import get_cayley_graph, cayley_graph_size, get_cayley_n, calculate_optimal_virtual_nodes

def enhanced_cayley_initialize_edge_weight(num_base_nodes, num_virtual_nodes, cayley_n=None, 
                                           high_value=5.0, low_value=0.1, verbose=True):
    """
    Enhanced initialization of edge weights between base and virtual nodes using the Cayley graph expansion.
    Edges that exist in the Cayley graph are given higher weights, but all edges have at least a small positive value.
    
    Args:
        num_base_nodes (int): Number of base (original) nodes.
        num_virtual_nodes (int): Number of virtual (centroid) nodes.
        cayley_n (int, optional): The Cayley graph parameter n to use. If None, will be calculated.
        high_value (float): The weight value for Cayley graph edges.
        low_value (float): The small positive weight value for non-Cayley edges.
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
    
    # Initialize all edges with the low value (small positive)
    edge_weights = torch.ones(num_base_nodes, num_virtual_nodes) * low_value
    
    # Keep track of edges for each base node to ensure connectivity
    base_node_connections = [[] for _ in range(num_base_nodes)]
    
    # For each edge in the Cayley graph that connects a base node to a virtual node
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        
        # Check if the edge connects a base node to a virtual node
        if src < num_base_nodes and dst >= num_base_nodes and dst < num_base_nodes + num_virtual_nodes:
            # Map the virtual node index to the range [0, num_virtual_nodes-1]
            virtual_idx = dst - num_base_nodes
            edge_weights[src, virtual_idx] = high_value  # Use higher weight for Cayley edges
            base_node_connections[src].append(virtual_idx)
        elif dst < num_base_nodes and src >= num_base_nodes and src < num_base_nodes + num_virtual_nodes:
            # Map the virtual node index to the range [0, num_virtual_nodes-1]
            virtual_idx = src - num_base_nodes
            edge_weights[dst, virtual_idx] = high_value  # Use higher weight for Cayley edges
            base_node_connections[dst].append(virtual_idx)
    
    # Ensure each base node has at least one high-value connection
    zero_rows = [i for i, connections in enumerate(base_node_connections) if not connections]
    
    if zero_rows and verbose:
        print(f"Warning: {len(zero_rows)} base nodes have no Cayley connections. Adding stronger random connections.")
    
    # For each node without Cayley connections, add some stronger random connections
    for row in zero_rows:
        # Add a few stronger random connections to virtual nodes
        num_random = min(3, num_virtual_nodes)
        random_indices = torch.randperm(num_virtual_nodes)[:num_random]
        for v_idx in random_indices:
            edge_weights[row, v_idx] = high_value
            base_node_connections[row].append(v_idx.item())
    
    # Normalize the weights for each base node to sum to 1
    row_sums = edge_weights.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero (shouldn't happen with our initialization)
    edge_weights = edge_weights / row_sums
    
    return edge_weights


def force_topk_connections(edge_weights, k, min_value=0.001):
    """
    Force exactly k connections for each base node, selecting the highest weight edges.
    If a node has fewer than k non-zero weights, random edges are selected to reach k.
    
    Args:
        edge_weights (torch.Tensor): Edge weights tensor of shape [num_base_nodes, num_virtual_nodes].
        k (int): The number of connections to keep per base node.
        min_value (float): The minimum weight value for random connections.
        
    Returns:
        torch.Tensor: Processed edge weights with exactly k connections per base node.
    """
    num_nodes, num_virtual = edge_weights.shape
    result = torch.zeros_like(edge_weights)
    
    for i in range(num_nodes):
        # Get node's weights
        node_weights = edge_weights[i]
        
        # Count non-zero weights
        non_zero = (node_weights > 0).sum().item()
        
        if non_zero >= k:
            # If we have enough non-zero weights, take top-k
            _, top_indices = torch.topk(node_weights, k=k)
        else:
            # Take all non-zero weights
            non_zero_indices = torch.nonzero(node_weights).view(-1)
            
            # For remaining connections, sample randomly from zeros
            zero_indices = torch.nonzero(node_weights == 0).view(-1)
            
            # Randomly select needed number of zero indices
            if len(zero_indices) > 0 and k > non_zero:
                perm = torch.randperm(len(zero_indices))
                random_indices = zero_indices[perm[:k-non_zero]]
                
                # Combine non-zero and random indices
                top_indices = torch.cat([non_zero_indices, random_indices])
            else:
                # If there aren't enough zero indices or k <= non_zero
                if non_zero < k:
                    # If we don't have enough indices total, repeat some
                    repeats_needed = k - non_zero
                    repeat_indices = non_zero_indices[torch.randperm(len(non_zero_indices))[:min(len(non_zero_indices), repeats_needed)]]
                    top_indices = torch.cat([non_zero_indices, repeat_indices])
                else:
                    # Take the top k from non-zero indices
                    _, top_idx = torch.topk(node_weights[non_zero_indices], k=min(k, len(non_zero_indices)))
                    top_indices = non_zero_indices[top_idx]
        
        # Set the selected connections
        result[i, top_indices] = node_weights[top_indices]
        
        # If weights were zero, assign small positive value
        zero_weights = (result[i, top_indices] == 0)
        result[i, top_indices[zero_weights]] = min_value
    
    # Normalize
    row_sums = result.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    result = result / row_sums
    
    return result
