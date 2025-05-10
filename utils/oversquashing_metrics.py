"""
Utility to measure oversquashing in graph neural networks.
Based on the oversquashing metric from Alon & Yahav, "On the Bottleneck of Graph Neural Networks
and its Practical Implications" (ICLR 2021)
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx

def compute_oversquashing_metric(adjacency_matrix, edge_weights=None):
    """
    Compute the oversquashing metric based on the effective resistance between nodes.
    
    Args:
        adjacency_matrix: Adjacency matrix or edge_index of the graph
        edge_weights: Optional edge weights to use (if None, use binary adjacency)
    
    Returns:
        Dict containing:
        - mean_effective_resistance: Average effective resistance across all pairs
        - max_effective_resistance: Maximum effective resistance between any pair
        - std_effective_resistance: Standard deviation of effective resistances
        - bottleneck_edges: Edges with highest removal impact on resistance
    """
    # Handle edge_index format
    if isinstance(adjacency_matrix, torch.Tensor) and adjacency_matrix.dim() == 2 and adjacency_matrix.size(0) == 2:
        # This is an edge_index tensor
        edge_index = adjacency_matrix
        
        # Determine number of nodes
        num_nodes = max(edge_index[0].max().item(), edge_index[1].max().item()) + 1
        
        # Create adjacency matrix
        device = edge_index.device
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        adj[edge_index[0], edge_index[1]] = 1
        
        # Make undirected if it's not already
        adj = (adj + adj.t()) > 0
        adj = adj.float()
    else:
        # Assume it's already an adjacency matrix
        adj = adjacency_matrix.float()
    
    # If edge weights provided, apply them to the adjacency matrix
    if edge_weights is not None and len(edge_weights) > 0:
        # Create weighted adjacency matrix
        rows, cols = adj.nonzero()
        if len(rows) > 0:  # Ensure there are edges
            # Check if we have enough weights
            if len(edge_weights) >= len(rows):
                for i, (r, c) in enumerate(zip(rows, cols)):
                    adj[r, c] = edge_weights[i]
            else:
                print(f"Warning: Not enough edge weights ({len(edge_weights)}) for edges ({len(rows)})")
    
    # Convert to NetworkX graph for resistance calculation
    G = nx.from_numpy_array(adj.cpu().numpy())
    
    # Compute effective resistance using NetworkX's Laplacian
    n = G.number_of_nodes()
    
    try:
        laplacian = nx.laplacian_matrix(G).toarray().astype(np.float32)
        
        # Use pseudoinverse to handle disconnected graphs
        lap_pinv = np.linalg.pinv(laplacian)
        
        # Compute effective resistances for all pairs
        resistances = []
        for i in range(n):
            for j in range(i+1, n):
                r_eff = lap_pinv[i, i] + lap_pinv[j, j] - 2 * lap_pinv[i, j]
                resistances.append((i, j, r_eff))
        
        # Extract statistics
        resistance_values = [r for _, _, r in resistances]
        mean_resistance = np.mean(resistance_values) if resistance_values else float('inf')
        max_resistance = np.max(resistance_values) if resistance_values else float('inf')
        std_resistance = np.std(resistance_values) if resistance_values else 0
        
        # Find bottleneck edges (those with highest effective resistance)
        bottleneck_edges = sorted(resistances, key=lambda x: x[2], reverse=True)[:5]
        
        return {
            "mean_effective_resistance": float(mean_resistance),
            "max_effective_resistance": float(max_resistance),
            "std_effective_resistance": float(std_resistance),
            "bottleneck_edges": [(int(i), int(j), float(r)) for i, j, r in bottleneck_edges]
        }
    except Exception as e:
        print(f"Error computing effective resistance: {e}")
        return {
            "mean_effective_resistance": float('inf'),
            "max_effective_resistance": float('inf'),
            "std_effective_resistance": 0,
            "bottleneck_edges": []
        }

def compute_graph_connectivity_metrics(adjacency_matrix, edge_weights=None):
    """
    Compute graph connectivity metrics to assess oversquashing.
    
    Args:
        adjacency_matrix: Adjacency matrix or edge_index of the graph
        edge_weights: Optional edge weights
        
    Returns:
        Dict with connectivity metrics
    """
    # Handle edge_index format
    if isinstance(adjacency_matrix, torch.Tensor) and adjacency_matrix.dim() == 2 and adjacency_matrix.size(0) == 2:
        # This is an edge_index tensor
        edge_index = adjacency_matrix
        
        # Determine number of nodes
        num_nodes = max(edge_index[0].max().item(), edge_index[1].max().item()) + 1
        
        # Create adjacency matrix
        device = edge_index.device
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        adj[edge_index[0], edge_index[1]] = 1
        
        # Make undirected if it's not already
        adj = (adj + adj.t()) > 0
        adj = adj.float()
    else:
        # Assume it's already an adjacency matrix
        adj = adjacency_matrix.float()
    
    # If edge weights provided, apply them to the adjacency matrix
    if edge_weights is not None and len(edge_weights) > 0:
        # Create weighted adjacency matrix
        rows, cols = adj.nonzero()
        if len(rows) > 0:  # Ensure there are edges
            # Check if we have enough weights
            if len(edge_weights) >= len(rows):
                weighted_adj = adj.clone()
                for i, (r, c) in enumerate(zip(rows, cols)):
                    weighted_adj[r, c] = edge_weights[i]
                adj = weighted_adj
    
    # Convert to NetworkX for analysis
    G = nx.from_numpy_array(adj.cpu().numpy())
    
    # Calculate basic graph metrics
    n = G.number_of_nodes()
    
    try:
        # Average shortest path length (only for connected graphs)
        avg_path_length = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        # Handle disconnected graphs
        avg_path_length = float('inf')
        # Use largest connected component instead
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        if len(largest_cc) > 1:
            try:
                avg_path_length = nx.average_shortest_path_length(subgraph)
            except:
                avg_path_length = float('inf')
    
    # Calculate clustering coefficient
    try:
        clustering = nx.average_clustering(G)
    except:
        clustering = 0.0
    
    # Diameter (maximum shortest path)
    try:
        diameter = nx.diameter(G)
    except nx.NetworkXError:
        diameter = float('inf')
        # Use largest connected component
        if 'largest_cc' in locals() and len(largest_cc) > 1:
            try:
                diameter = nx.diameter(subgraph)
            except:
                diameter = float('inf')
    
    # Edge distribution statistics
    degrees = [d for _, d in G.degree()]
    mean_degree = np.mean(degrees) if degrees else 0
    std_degree = np.std(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0
    
    # Calculate spectral gap (related to mixing time)
    try:
        laplacian = nx.normalized_laplacian_matrix(G).toarray()
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = sorted(eigenvalues)
        # Spectral gap is difference between smallest non-zero eigenvalue and zero
        spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0
    except:
        spectral_gap = 0
    
    return {
        "avg_path_length": float(avg_path_length),
        "diameter": float(diameter),
        "clustering_coefficient": float(clustering),
        "mean_degree": float(mean_degree),
        "std_degree": float(std_degree),
        "max_degree": float(max_degree),
        "spectral_gap": float(spectral_gap)
    }

def analyze_model_oversquashing(model, data, device='cpu'):
    """
    Analyze the oversquashing in a model using the final learned edge weights.
    
    Args:
        model: The GNN model (must have a method to access edge weights)
        data: The graph data
        device: Device to run computations on
        
    Returns:
        Dict with oversquashing metrics
    """
    try:
        # Move data to device
        data = data.to(device)
        
        # Put model in eval mode
        model.eval()
        
        # Get the learned edge weights from model
        # This will need to be adapted based on your model's implementation
        with torch.no_grad():
            # For IPR-MPNN models, we need to extract the edge weights
            if hasattr(model, 'get_final_edge_weights'):
                edge_weights_info = model.get_final_edge_weights(0)  # Use graph index 0 for single graphs
                if edge_weights_info is None:
                    print("No edge weights info available from model")
                    return None
                
                # Extract weights from the model's output format
                edge_weights = edge_weights_info.get('edge_weights', None)
            else:
                # Try to locate edge weights (depends on model structure)
                try:
                    # Try accessing edge_weights through standard accessor
                    edge_weights = model.edge_weights
                except:
                    edge_weights = None
                    print("Could not extract edge weights from model")
        
        # Get the adjacency matrix
        if hasattr(data, 'edge_index'):
            edge_index = data.edge_index
            
            # Compute metrics directly using the edge_index
            oversquashing_metrics = compute_oversquashing_metric(edge_index, edge_weights)
            connectivity_metrics = compute_graph_connectivity_metrics(edge_index, edge_weights)
            
            # Combine metrics
            results = {
                "oversquashing_metrics": oversquashing_metrics,
                "connectivity_metrics": connectivity_metrics
            }
            
            return results
        else:
            print("Data doesn't have an edge_index attribute")
            return None
    except Exception as e:
        print(f"Error in analyze_model_oversquashing: {e}")
        return None
