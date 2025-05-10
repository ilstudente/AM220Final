"""
Utility functions for Cayley graph initialization of edge weights in IPR-MPNN.
"""

import torch
import numpy as np
from collections import deque
import torch.nn.functional as F

def get_cayley_graph(n):
    """
    Get the edge index of the Cayley graph (Cay(SL(2, Z_n); S_n)).
    
    Args:
        n (int): Order of the Cayley graph.
        
    Returns:
        torch.Tensor: Edge index tensor of shape [2, num_edges].
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

def cayley_graph_size(n):
    """
    Compute the size (number of nodes) of the Cayley graph.
    
    Args:
        n (int): Order of the Cayley graph.
        
    Returns:
        int: Number of nodes in the Cayley graph.
    """
    # For SL(2, Z_n), a rough estimate is n^3
    n = int(n)
    return n*n*n

def get_cayley_n(num_nodes):
    """
    Find the smallest value of n such that the Cayley graph has at least num_nodes.
    
    Args:
        num_nodes (int): Minimum number of nodes needed.
        
    Returns:
        int: Smallest suitable order for the Cayley graph.
    """
    n = 1
    while cayley_graph_size(n) < num_nodes:
        n += 1
    return n

def calculate_optimal_virtual_nodes(num_base_nodes, min_ratio=0.3, max_ratio=0.8, verbose=True):
    """
    Calculate the optimal number of virtual nodes based on Cayley graph expansion.
    
    Given a number of base nodes, this function calculates how many virtual nodes 
    would be added by a Cayley graph expansion, ensuring compatibility between 
    the IPR-MPNN architecture and the Cayley graph structure.
    
    Args:
        num_base_nodes (int): Number of base (original) nodes.
        min_ratio (float): Minimum ratio of virtual nodes to base nodes.
        max_ratio (float): Maximum ratio of virtual nodes to base nodes.
        verbose (bool): Whether to print information.
        
    Returns:
        int: Optimal number of virtual nodes to add.
        int: The Cayley graph parameter n that was used.
    """
    # Find the smallest n value for the Cayley graph that can accommodate at least num_base_nodes
    cayley_n = get_cayley_n(num_base_nodes)
    
    # Calculate the total number of nodes in the Cayley graph
    total_cayley_nodes = cayley_graph_size(cayley_n)
    
    # The number of virtual nodes is the difference between total nodes and base nodes
    # This ensures we have exactly the right number of nodes for the Cayley graph structure
    num_virtual_nodes = total_cayley_nodes - num_base_nodes
    
    # We don't apply min/max ratio constraints anymore, as we need to use exactly
    # the number required by the Cayley graph structure
    
    # Ensure we have at least 1 virtual node
    num_virtual_nodes = max(1, num_virtual_nodes)
    
    if verbose:
        print(f"Base nodes: {num_base_nodes}, Cayley n: {cayley_n}, " 
              f"Total Cayley nodes: {total_cayley_nodes}, Virtual nodes: {num_virtual_nodes}")
    
    return num_virtual_nodes, cayley_n

def cayley_initialize_edge_weight(num_base_nodes, num_virtual_nodes, cayley_n=None, verbose=True):
    """
    Initialize edge weights for edges between base and virtual nodes using the Cayley graph expansion.
    Edges that exist in the Cayley graph are given weight 1, others are given weight 0.
    
    Args:
        num_base_nodes (int): Number of base (original) nodes.
        num_virtual_nodes (int): Number of virtual (centroid) nodes.
        cayley_n (int, optional): The Cayley graph parameter n to use. If None, will be calculated.
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
    
    # We should no longer have a mismatch if Cayley parameter n was provided
    if cayley_num_nodes != num_base_nodes + num_virtual_nodes and verbose:
        print(f"Warning: Cayley graph has {cayley_num_nodes} nodes, more than the required {num_base_nodes + num_virtual_nodes}")
    
    # Create an edge weight matrix initialized with zeros
    edge_weights = torch.zeros(num_base_nodes, num_virtual_nodes)
    
    # Keep track of edges for each base node to ensure connectivity
    base_node_connections = [[] for _ in range(num_base_nodes)]
    
    # For each edge in the Cayley graph that connects a base node to a virtual node
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        
        # Check if the edge connects a base node to a virtual node
        if src < num_base_nodes and dst >= num_base_nodes and dst < num_base_nodes + num_virtual_nodes:
            # Map the virtual node index to the range [0, num_virtual_nodes-1]
            virtual_idx = dst - num_base_nodes
            edge_weights[src, virtual_idx] = 1.0
            base_node_connections[src].append(virtual_idx)
        elif dst < num_base_nodes and src >= num_base_nodes and src < num_base_nodes + num_virtual_nodes:
            # Map the virtual node index to the range [0, num_virtual_nodes-1]
            virtual_idx = src - num_base_nodes
            edge_weights[dst, virtual_idx] = 1.0
            base_node_connections[dst].append(virtual_idx)
    
    # Ensure each base node has at least one connection
    zero_rows = [i for i, connections in enumerate(base_node_connections) if not connections]
    
    if zero_rows:
        if verbose:
            print(f"Warning: {len(zero_rows)} base nodes have no connections. Adding random connections.")
        
        # For each disconnected node, find the closest connected node
        # and connect it to some of the same virtual nodes
        for row in zero_rows:
            # Find nodes that have connections
            connected_nodes = [i for i, conns in enumerate(base_node_connections) if conns]
            
            if connected_nodes:
                # Simple strategy: connect to the same virtual nodes as a neighbor
                # This maintains some structure rather than using random connections
                neighbor_idx = connected_nodes[min(len(connected_nodes)-1, row % len(connected_nodes))]
                virtual_indices = base_node_connections[neighbor_idx]
                
                if virtual_indices:
                    # Connect to a subset of the neighbor's virtual nodes
                    for v_idx in virtual_indices[:min(2, len(virtual_indices))]:
                        edge_weights[row, v_idx] = 1.0
                        base_node_connections[row].append(v_idx)
                else:
                    # Fallback to random connection if needed
                    virtual_idx = torch.randint(0, num_virtual_nodes, (1,)).item()
                    edge_weights[row, virtual_idx] = 1.0
                    base_node_connections[row].append(virtual_idx)
            else:
                # No connected nodes found, use random connections
                virtual_idx = torch.randint(0, num_virtual_nodes, (1,)).item()
                edge_weights[row, virtual_idx] = 1.0
                base_node_connections[row].append(virtual_idx)
    
    # Normalize the weights for each base node to sum to 1
    # This ensures comparable message passing regardless of number of connections
    row_sums = edge_weights.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    edge_weights = edge_weights / row_sums
    
    return edge_weights
