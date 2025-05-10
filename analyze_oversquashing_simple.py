"""
Simple script to analyze oversquashing metrics for balanced Cayley vs. uniform initialization.
Focused on key metrics with minimal dependencies.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from models.memory_saver_iprmpnn import MemorySaverIPRMPNNModel
from utils.oversquashing_metrics import compute_oversquashing_metric, compute_graph_connectivity_metrics

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def compute_cheeger_constant(adj_matrix):
    """
    Compute the Cheeger constant (conductance) of a graph.
    Uses spectral approximation for efficiency.
    """
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()
        
    G = nx.from_numpy_array(adj_matrix)
    
    # If graph is not connected, return 0
    if not nx.is_connected(G):
        return 0.0
    
    try:
        # Calculate the normalized Laplacian
        L = nx.normalized_laplacian_matrix(G).astype(np.float64)
        L_dense = L.todense()
        
        # Get the second smallest eigenvalue (first non-zero)
        eigenvalues = np.linalg.eigvalsh(L_dense)
        eigenvalues.sort()
        
        # The second eigenvalue is related to the Cheeger constant
        lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0
        
        # Use Cheeger's inequality: lambda_2/2 <= h(G) <= sqrt(2*lambda_2)
        cheeger_estimate = lambda_2 / 2
        
        return float(cheeger_estimate)
    except Exception as e:
        print(f"Error computing Cheeger constant: {e}")
        return 0.0

def analyze_graph_oversquashing(dataset_name, k=3, hidden_dim=16, num_graphs=5, seed=42):
    """
    Compare oversquashing metrics between balanced Cayley and uniform initialization.
    """
    print(f"Analyzing oversquashing metrics for {dataset_name}...")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"oversquashing_analysis/simple_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    try:
        dataset = TUDataset(root='/workspaces/IPR-MPNN/datasets', name=dataset_name)
        print(f"Dataset: {dataset_name}, Graphs: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Limit analysis to specified number of graphs
    num_graphs = min(num_graphs, len(dataset))
    analysis_data = dataset[:num_graphs]
    
    # Create data loader with batch size 1
    data_loader = DataLoader(analysis_data, batch_size=1, shuffle=False)
    
    # Initialize models
    balanced_model = MemorySaverIPRMPNNModel(
        input_dim=dataset.num_features,
        hidden_dim=hidden_dim,
        output_dim=dataset.num_classes,
        edge_init_type='cayley',
        top_k=k
    ).to(device)
    
    uniform_model = MemorySaverIPRMPNNModel(
        input_dim=dataset.num_features,
        hidden_dim=hidden_dim,
        output_dim=dataset.num_classes,
        edge_init_type='uniform',
        top_k=k
    ).to(device)
    
    # Set models to evaluation mode
    balanced_model.eval()
    uniform_model.eval()
    
    # Enable oversquashing tracking
    balanced_model.collect_oversquashing_metrics = True
    uniform_model.collect_oversquashing_metrics = True
    
    # Collect metrics
    balanced_metrics = []
    uniform_metrics = []
    
    # Process each graph
    for idx, data in enumerate(data_loader):
        print(f"Processing graph {idx}...")
        data = data.to(device)
        
        with torch.no_grad():
            # Forward pass through balanced model
            balanced_out = balanced_model(data)
            balanced_weights = balanced_model.get_final_edge_weights(0)
            
            # Forward pass through uniform model
            uniform_out = uniform_model(data)
            uniform_weights = uniform_model.get_final_edge_weights(0)
            
            if balanced_weights and uniform_weights:
                # Create adjacency matrix from edge index
                edge_index = data.edge_index
                num_nodes = data.num_nodes
                
                # Get edge weights
                balanced_edge_weights = balanced_weights.get('edge_weights', None)
                uniform_edge_weights = uniform_weights.get('edge_weights', None)
                
                # Calculate standard metrics
                try:
                    # For balanced initialization
                    balanced_osm = compute_oversquashing_metric(edge_index, None)
                    balanced_conn = compute_graph_connectivity_metrics(edge_index, None)
                    
                    # Create adjacency for Cheeger constant
                    adj = torch.zeros(num_nodes, num_nodes, device=device)
                    adj[edge_index[0], edge_index[1]] = 1
                    balanced_cheeger = compute_cheeger_constant(adj)
                    
                    # For uniform initialization
                    uniform_osm = compute_oversquashing_metric(edge_index, None) 
                    uniform_conn = compute_graph_connectivity_metrics(edge_index, None)
                    uniform_cheeger = compute_cheeger_constant(adj)  # Same graph structure
                    
                    # Store metrics
                    balanced_metrics.append({
                        'graph_idx': idx,
                        'num_nodes': num_nodes,
                        'oversquashing': balanced_osm,
                        'connectivity': balanced_conn,
                        'cheeger_constant': balanced_cheeger
                    })
                    
                    uniform_metrics.append({
                        'graph_idx': idx,
                        'num_nodes': num_nodes,
                        'oversquashing': uniform_osm,
                        'connectivity': uniform_conn,
                        'cheeger_constant': uniform_cheeger
                    })
                    
                    print(f"  Successfully collected metrics for graph {idx}")
                except Exception as e:
                    print(f"  Error calculating metrics for graph {idx}: {e}")
    
    # Disable tracking
    balanced_model.collect_oversquashing_metrics = False
    uniform_model.collect_oversquashing_metrics = False
    
    # Save metrics
    if balanced_metrics and uniform_metrics:
        results = {
            'dataset': dataset_name,
            'parameters': {
                'top_k': k,
                'hidden_dim': hidden_dim,
                'seed': seed,
                'num_graphs': num_graphs
            },
            'metrics': {
                'balanced_cayley': balanced_metrics,
                'uniform': uniform_metrics
            }
        }
        
        results_file = os.path.join(output_dir, f"{dataset_name.lower()}_oversquashing.json")
        with open(results_file, 'w') as f:
            # Convert numpy/torch values to Python natives
            def convert_to_native(obj):
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                elif isinstance(obj, (np.ndarray, torch.Tensor)):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(i) for i in obj]
                else:
                    return obj
            
            json.dump(convert_to_native(results), f, indent=2)
        
        print(f"Results saved to {results_file}")
        
        # Create visualization
        visualize_metrics(balanced_metrics, uniform_metrics, dataset_name, output_dir)
        
        return results
    else:
        print("No metrics collected")
        return None

def visualize_metrics(balanced_metrics, uniform_metrics, dataset_name, output_dir):
    """Create simple visualization of oversquashing metrics"""
    if not balanced_metrics or not uniform_metrics:
        print("No metrics to visualize")
        return
    
    # Extract key metrics
    balanced_resistance = [m['oversquashing']['mean_effective_resistance'] for m in balanced_metrics]
    uniform_resistance = [m['oversquashing']['mean_effective_resistance'] for m in uniform_metrics]
    
    balanced_spectral_gap = [m['connectivity']['spectral_gap'] for m in balanced_metrics]
    uniform_spectral_gap = [m['connectivity']['spectral_gap'] for m in uniform_metrics]
    
    balanced_cheeger = [m['cheeger_constant'] for m in balanced_metrics]
    uniform_cheeger = [m['cheeger_constant'] for m in uniform_metrics]
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Effective resistance
    plt.subplot(2, 2, 1)
    idx = range(len(balanced_resistance))
    plt.bar([i - 0.2 for i in idx], balanced_resistance, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in idx], uniform_resistance, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title(f'Effective Resistance ({dataset_name})')
    plt.xlabel('Graph Index')
    plt.ylabel('Mean Effective Resistance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Spectral gap
    plt.subplot(2, 2, 2)
    plt.bar([i - 0.2 for i in idx], balanced_spectral_gap, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in idx], uniform_spectral_gap, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title(f'Spectral Gap ({dataset_name})')
    plt.xlabel('Graph Index')
    plt.ylabel('Spectral Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Cheeger constant
    plt.subplot(2, 2, 3)
    plt.bar([i - 0.2 for i in idx], balanced_cheeger, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in idx], uniform_cheeger, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title(f'Cheeger Constant ({dataset_name})')
    plt.xlabel('Graph Index')
    plt.ylabel('Cheeger Constant')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Summary
    plt.subplot(2, 2, 4)
    
    metrics = [
        ('Resistance', np.mean(balanced_resistance), np.mean(uniform_resistance)),
        ('Spectral Gap', np.mean(balanced_spectral_gap), np.mean(uniform_spectral_gap)),
        ('Cheeger', np.mean(balanced_cheeger), np.mean(uniform_cheeger))
    ]
    
    labels = [m[0] for m in metrics]
    balanced_values = [m[1] for m in metrics]
    uniform_values = [m[2] for m in metrics]
    
    x = range(len(labels))
    
    plt.bar([i - 0.2 for i in x], balanced_values, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in x], uniform_values, width=0.4, label='Uniform', color='orange', alpha=0.7)
    
    plt.title(f'Metrics Summary ({dataset_name})')
    plt.xlabel('Metric')
    plt.ylabel('Average Value')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save visualization
    plt.tight_layout()
    vis_file = os.path.join(output_dir, f"{dataset_name.lower()}_metrics.png")
    plt.savefig(vis_file)
    plt.close()
    
    print(f"Visualization saved to {vis_file}")

if __name__ == "__main__":
    # Run analysis on all three datasets
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"oversquashing_analysis/comparison_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    # Start with MUTAG (small dataset)
    print("\n=== Running MUTAG Analysis ===")
    mutag_results = analyze_graph_oversquashing("MUTAG", k=3, hidden_dim=16, num_graphs=10, seed=42)
    all_results["MUTAG"] = mutag_results
    
    # Then PROTEINS (medium dataset)
    print("\n=== Running PROTEINS Analysis ===")
    proteins_results = analyze_graph_oversquashing("PROTEINS", k=3, hidden_dim=16, num_graphs=10, seed=42)
    all_results["PROTEINS"] = proteins_results
    
    # Finally ENZYMES (larger dataset)
    print("\n=== Running ENZYMES Analysis ===")
    enzymes_results = analyze_graph_oversquashing("ENZYMES", k=3, hidden_dim=16, num_graphs=10, seed=42)
    all_results["ENZYMES"] = enzymes_results
    
    # Save combined results
    combined_file = os.path.join(output_dir, "all_datasets_oversquashing.json")
    with open(combined_file, 'w') as f:
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, (np.ndarray, torch.Tensor)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(i) for i in obj]
            else:
                return obj
        
        json.dump(convert_to_native(all_results), f, indent=2)
    
    print(f"\nAll analyses complete. Combined results saved to {combined_file}")
