"""
Simplified utility functions for measuring oversquashing in graphs.
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import matplotlib.pyplot as plt

# Import calculate_optimal_virtual_nodes and cayley_initialize_edge_weight functions
# These are used in the evaluate_with_virtual_nodes function
from .cayley_utils import calculate_optimal_virtual_nodes, cayley_initialize_edge_weight

def compute_cheeger_constant(data):
    """
    Compute an approximation of the Cheeger constant (isoperimetric number) of a graph.
    
    Args:
        data: PyG data object
        
    Returns:
        float: Approximate Cheeger constant
    """
    # Convert PyG graph to NetworkX for easier analysis
    g = to_networkx(data, to_undirected=True)
    
    # If graph is not connected, return 0
    if not nx.is_connected(g):
        return 0.0
    
    # For small graphs, use spectral approximation
    try:
        # Compute normalized Laplacian
        laplacian = nx.normalized_laplacian_matrix(g)
        # Get second smallest eigenvalue (Fiedler value)
        eigenvalues = np.sort(np.linalg.eigvalsh(laplacian.toarray()))
        # Cheeger is approximated by the second smallest eigenvalue
        return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    except:
        # Fall back to a simple approximation if computation fails
        return 0.0

def compute_dirichlet_energy(data):
    """
    Compute the Dirichlet energy of a graph which measures 
    how much the node features vary across edges.
    
    Args:
        data: PyG data object
        
    Returns:
        float: Dirichlet energy
    """
    edge_index = data.edge_index
    if not hasattr(data, 'x') or data.x is None:
        # If no features, use node degrees
        g = to_networkx(data, to_undirected=True)
        degrees = np.array([d for _, d in g.degree()])
        x = torch.tensor(degrees, dtype=torch.float).view(-1, 1)
    else:
        x = data.x
    
    # Sum of squared differences across edges
    energy = 0.0
    num_edges = edge_index.shape[1]
    
    if num_edges == 0:
        return 0.0
    
    for i in range(num_edges):
        source, target = edge_index[0, i], edge_index[1, i]
        diff = x[source] - x[target]
        energy += torch.sum(diff * diff).item()
    
    # Normalize by number of edges
    return energy / num_edges

def compute_effective_resistance(data):
    """
    Compute the average effective resistance of a graph.
    
    Args:
        data: PyG data object
        
    Returns:
        float: Average effective resistance
    """
    # Convert to NetworkX
    g = to_networkx(data, to_undirected=True)
    
    # If graph is not connected, return a high value
    if not nx.is_connected(g):
        return 10.0
    
    # Use average shortest path length as a proxy for effective resistance
    try:
        return nx.average_shortest_path_length(g)
    except:
        # For disconnected graphs, fall back to a default value
        return 5.0

def evaluate_oversquashing_metrics(data_loader, device='cpu'):
    """
    Evaluate various oversquashing metrics on a dataset.
    
    Args:
        data_loader: PyG data loader
        device: Computation device
        
    Returns:
        dict: Metrics
    """
    metrics = {
        'cheeger_constants': [],
        'dirichlet_energies': [],
        'effective_resistances': []
    }
    
    for batch in data_loader:
        # For each graph in the batch
        num_graphs = batch.batch[-1].item() + 1 if hasattr(batch, 'batch') else 1
        
        for i in range(num_graphs):
            # Extract individual graph
            if num_graphs > 1:
                # Get mask for this graph
                mask = batch.batch == i
                # Create a new data object for this graph
                graph_data = type(batch)()
                graph_data.edge_index = batch.edge_index[:, mask[batch.edge_index[0]]]
                if hasattr(batch, 'x') and batch.x is not None:
                    graph_data.x = batch.x[mask]
                
                # Adjust node indices
                if graph_data.edge_index.numel() > 0:
                    node_idx_old = mask.nonzero(as_tuple=True)[0]
                    node_idx_new = torch.arange(mask.sum())
                    for j in range(graph_data.edge_index.shape[1]):
                        # Map old indices to new ones
                        src, dst = graph_data.edge_index[0, j], graph_data.edge_index[1, j]
                        graph_data.edge_index[0, j] = node_idx_new[node_idx_old == src][0]
                        graph_data.edge_index[1, j] = node_idx_new[node_idx_old == dst][0]
            else:
                graph_data = batch
            
            # Compute metrics
            if graph_data.edge_index.numel() > 0:
                metrics['cheeger_constants'].append(compute_cheeger_constant(graph_data))
                metrics['dirichlet_energies'].append(compute_dirichlet_energy(graph_data))
                metrics['effective_resistances'].append(compute_effective_resistance(graph_data))
    
    # Compute averages
    avg_metrics = {}
    for key in list(metrics.keys()):
        if metrics[key]:
            avg_metrics[f'avg_{key}'] = sum(metrics[key]) / len(metrics[key])
        else:
            avg_metrics[f'avg_{key}'] = 0.0
    
    return avg_metrics

def evaluate_oversquashing_with_virtual_nodes(data_loader, model, device='cpu'):
    """
    Evaluate oversquashing metrics on graphs after adding virtual nodes according to model's initialization.
    
    Args:
        data_loader: PyG data loader
        model: The GNN model with information about virtual node initialization
        device: Computation device
        
    Returns:
        dict: Metrics for graphs with virtual nodes
    """
    metrics = {
        'cheeger_constants': [],
        'dirichlet_energies': [],
        'effective_resistances': []
    }
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Process each graph in the loader
    for batch in data_loader:
        batch = batch.to(device)
        # For each graph in the batch
        num_graphs = batch.batch[-1].item() + 1 if hasattr(batch, 'batch') else 1
        
        for i in range(num_graphs):
            # Extract original graph
            mask = batch.batch == i
            num_nodes = mask.sum().item()
            
            # Calculate the number of virtual nodes and connectivity according to the model's initialization
            if model.edge_init_type == 'cayley':
                num_virtual_nodes, cayley_n = calculate_optimal_virtual_nodes(
                    num_base_nodes=num_nodes, 
                    verbose=False
                )
                
                # Get edge weights from Cayley initialization
                edge_weights = cayley_initialize_edge_weight(
                    num_base_nodes=num_nodes, 
                    num_virtual_nodes=num_virtual_nodes,
                    cayley_n=cayley_n,
                    verbose=False
                ).to(device)
            else:
                # Use uniform initialization
                num_virtual_nodes, cayley_n = calculate_optimal_virtual_nodes(
                    num_base_nodes=num_nodes, 
                    verbose=False
                )
                edge_weights = torch.ones(num_nodes, num_virtual_nodes, device=device) / num_virtual_nodes
            
            # Extract the original graph's nodes and edges
            graph_nodes = mask.sum().item()
            edge_mask = mask[batch.edge_index[0]]
            graph_edges = batch.edge_index[:, edge_mask].clone()
            
            # Adjust node indices to start from 0
            if graph_edges.numel() > 0:
                node_idx_old = mask.nonzero(as_tuple=True)[0]
                node_idx_new = torch.arange(mask.sum(), device=device)
                for j in range(graph_edges.shape[1]):
                    src, dst = graph_edges[0, j], graph_edges[1, j]
                    graph_edges[0, j] = node_idx_new[node_idx_old == src][0]
                    graph_edges[1, j] = node_idx_new[node_idx_old == dst][0]
            
            # Calculate total number of nodes (original + virtual)
            total_nodes = graph_nodes + num_virtual_nodes
            
            # Create edges connecting original nodes to virtual nodes based on edge_weights
            virtual_edges = []
            for n in range(graph_nodes):
                for v in range(num_virtual_nodes):
                    if edge_weights[n, v] > 0:
                        # Connect base node to virtual node (bidirectional)
                        virtual_edges.append([n, graph_nodes + v])
                        virtual_edges.append([graph_nodes + v, n])
            
            if virtual_edges:
                virtual_edges = torch.tensor(virtual_edges, device=device).t()
                
                # Combine original edges with virtual node connections
                if graph_edges.numel() > 0:
                    combined_edges = torch.cat([graph_edges, virtual_edges], dim=1)
                else:
                    combined_edges = virtual_edges
                
                # Create a new data object with the combined graph
                from torch_geometric.data import Data
                combined_graph = Data(edge_index=combined_edges, num_nodes=total_nodes)
                
                # Compute oversquashing metrics on the combined graph
                if combined_edges.numel() > 0:
                    metrics['cheeger_constants'].append(compute_cheeger_constant(combined_graph))
                    metrics['dirichlet_energies'].append(compute_dirichlet_energy(combined_graph))
                    metrics['effective_resistances'].append(compute_effective_resistance(combined_graph))
    
    # Compute averages
    avg_metrics = {}
    for key in list(metrics.keys()):
        if metrics[key]:
            avg_metrics[f'avg_{key}'] = sum(metrics[key]) / len(metrics[key])
        else:
            avg_metrics[f'avg_{key}'] = 0.0
    
    return avg_metrics

def evaluate_with_virtual_nodes(data_loader, model, device='cpu'):
    """
    Evaluate oversquashing metrics on graphs with virtual nodes added
    according to the model's initialization scheme.
    
    Args:
        data_loader: PyG data loader
        model: The IPR-MPNN model with virtual nodes
        device: Computation device
        
    Returns:
        dict: Metrics
    """
    metrics = {
        'cheeger_constants': [],
        'dirichlet_energies': [],
        'effective_resistances': []
    }
    
    # Track connectivity statistics
    connectivity_stats = {
        'nonzero_ratio': [],
        'avg_connections_per_base': [],
        'avg_connections_per_virtual': []
    }
    
    # Process each batch
    for batch in data_loader:
        batch = batch.to(device)
        
        # For each graph in the batch
        num_graphs = batch.batch[-1].item() + 1 if hasattr(batch, 'batch') else 1
        
        for graph_idx in range(num_graphs):
            # Extract nodes for this graph
            graph_mask = (batch.batch == graph_idx)
            num_nodes = graph_mask.sum().item()
            
            # Generate graph ID consistent with the model
            graph_id = f"{num_nodes}_{graph_idx}"
            
            # Get optimized edge weights if they exist, otherwise use initialization
            if hasattr(model, 'graph_edge_weights') and graph_id in model.graph_edge_weights:
                # Get the learned weights from the model
                edge_weights = model.graph_edge_weights[graph_id].detach()
                
                # If model is using top-k pruning, apply it
                if model.top_k is not None and model.top_k < edge_weights.size(1):
                    # Get the top-k indices per base node
                    _, top_indices = torch.topk(edge_weights, k=model.top_k, dim=1)
                    
                    # Create a mask for the top-k connections
                    mask = torch.zeros_like(edge_weights)
                    for i in range(num_nodes):
                        mask[i, top_indices[i]] = 1.0
                    
                    # Apply the mask
                    edge_weights = edge_weights * mask
                
                # Normalize the weights
                row_sums = edge_weights.sum(dim=1, keepdim=True)
                row_sums[row_sums == 0] = 1.0  # Avoid division by zero
                edge_weights = edge_weights / row_sums
                
                num_virtual_nodes = edge_weights.size(1)
            else:
                # Fall back to initialization
                if model.edge_init_type == 'cayley':
                    num_virtual_nodes, cayley_n = calculate_optimal_virtual_nodes(
                        num_base_nodes=num_nodes, 
                        verbose=False
                    )
                    
                    # Get Cayley graph weights
                    edge_weights = cayley_initialize_edge_weight(
                        num_base_nodes=num_nodes, 
                        num_virtual_nodes=num_virtual_nodes,
                        cayley_n=cayley_n,
                        verbose=False
                    ).to(device)
                else:  # uniform
                    num_virtual_nodes, _ = calculate_optimal_virtual_nodes(
                        num_base_nodes=num_nodes, 
                        verbose=False
                    )
                    
                    # Uniform weights
                    edge_weights = torch.ones(num_nodes, num_virtual_nodes, device=device) / num_virtual_nodes
            
            # Record connectivity statistics
            nonzero_ratio = (edge_weights > 1e-5).float().mean().item() * 100  # Percentage of non-zero weights
            connections_per_base = (edge_weights > 1e-5).float().sum(dim=1).mean().item()
            connections_per_virtual = (edge_weights > 1e-5).float().sum(dim=0).mean().item()
            
            connectivity_stats['nonzero_ratio'].append(nonzero_ratio)
            connectivity_stats['avg_connections_per_base'].append(connections_per_base)
            connectivity_stats['avg_connections_per_virtual'].append(connections_per_virtual)
            
            # Create a new graph with virtual nodes
            total_nodes = num_nodes + num_virtual_nodes
            
            # First, extract original graph edges
            orig_edge_index = batch.edge_index[:, graph_mask[batch.edge_index[0]]]
            
            # Adjust indices to start from 0
            if orig_edge_index.shape[1] > 0:
                min_idx = orig_edge_index.min()
                orig_edge_index = orig_edge_index - min_idx
            
            # Create edges between base and virtual nodes based on weights
            base_to_virtual_edges = []
            for base_idx in range(num_nodes):
                for v_idx in range(num_virtual_nodes):
                    if edge_weights[base_idx, v_idx] > 1e-5:  # Only add edges with non-negligible weight
                        # Add edge from base to virtual (using indices adjusted to the new graph)
                        base_to_virtual_edges.append([base_idx, num_nodes + v_idx])
                        # Add reverse edge
                        base_to_virtual_edges.append([num_nodes + v_idx, base_idx])
            
            # Convert to tensor if we have any edges
            if base_to_virtual_edges:
                base_to_virtual_edges = torch.tensor(base_to_virtual_edges, device=device).t()
                
                # Combine original and new edges
                if orig_edge_index.shape[1] > 0:
                    new_edge_index = torch.cat([orig_edge_index, base_to_virtual_edges], dim=1)
                else:
                    new_edge_index = base_to_virtual_edges
            else:
                new_edge_index = orig_edge_index
            
            # Create a PyG data object for the augmented graph
            augmented_graph = Data(edge_index=new_edge_index, num_nodes=total_nodes)
            
            # Compute metrics on the augmented graph
            if augmented_graph.edge_index.numel() > 0:
                metrics['cheeger_constants'].append(compute_cheeger_constant(augmented_graph))
                metrics['dirichlet_energies'].append(compute_dirichlet_energy(augmented_graph))
                metrics['effective_resistances'].append(compute_effective_resistance(augmented_graph))
    
    # Compute averages for metrics
    avg_metrics = {}
    for key in list(metrics.keys()):
        if metrics[key]:
            avg_metrics[f'avg_{key}'] = sum(metrics[key]) / len(metrics[key])
        else:
            avg_metrics[f'avg_{key}'] = 0.0
    
    # Add connectivity statistics to metrics
    for key in connectivity_stats:
        if connectivity_stats[key]:
            avg_metrics[key] = sum(connectivity_stats[key]) / len(connectivity_stats[key])
        else:
            avg_metrics[key] = 0.0
    
    return avg_metrics

def plot_oversquashing_metrics(uniform_metrics, cayley_metrics, output_dir, timestamp):
    """
    Plot comparison of oversquashing metrics between uniform and Cayley initialization.
    
    Args:
        uniform_metrics: Metrics for uniform initialization
        cayley_metrics: Metrics for Cayley initialization
        output_dir: Directory to save plots
        timestamp: Timestamp for file naming
    """
    # Metrics to plot
    metrics = [
        ('avg_cheeger_constants', 'Average Cheeger Constant\n(higher is better)'),
        ('avg_dirichlet_energies', 'Average Dirichlet Energy\n(lower is better)'),
        ('avg_effective_resistances', 'Average Effective Resistance\n(lower is better)')
    ]
    
    # Create plot
    plt.figure(figsize=(15, 5))
    
    for i, (metric_key, metric_name) in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        
        # Get values, handling cases where a metric might be missing
        uniform_val = uniform_metrics.get(metric_key, 0)
        cayley_val = cayley_metrics.get(metric_key, 0)
        
        # Plot bars
        bars = plt.bar([0, 1], [uniform_val, cayley_val], width=0.6)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.4f}', ha='center', va='bottom')
        
        # Set labels and title
        plt.xticks([0, 1], ['Uniform', 'Cayley'])
        plt.ylabel('Value')
        plt.title(metric_name)
        
        # Highlight better value
        if metric_key == 'avg_cheeger_constants':
            # For Cheeger constant, higher is better (less oversquashing)
            better_idx = 0 if uniform_val > cayley_val else 1
            bars[better_idx].set_color('green')
            bars[1-better_idx].set_color('lightblue')
        else:
            # For other metrics, lower is better (less oversquashing)
            better_idx = 0 if uniform_val < cayley_val else 1
            bars[better_idx].set_color('green')
            bars[1-better_idx].set_color('lightblue')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/oversquashing_metrics_{timestamp}.png')
    plt.close()
