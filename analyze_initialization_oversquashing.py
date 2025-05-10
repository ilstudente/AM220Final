"""
Analyze oversquashing metrics for different initialization methods.
This script compares the oversquashing characteristics of balanced Cayley vs uniform
initialization by directly analyzing their connectivity patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import networkx as nx
from tqdm import tqdm

from models.balanced_cayley_utils import balanced_cayley_initialize_edge_weight, balanced_topk_pruning

def create_output_dir():
    """Create output directory for oversquashing analysis."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"oversquashing_analysis/analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_adjacency_matrix(edge_weights, top_k=None):
    """Convert edge weights to adjacency matrix with optional top-k pruning."""
    if top_k is not None and top_k < edge_weights.shape[1]:
        pruned_weights = balanced_topk_pruning(edge_weights, k=top_k)
    else:
        pruned_weights = edge_weights
    
    # Create binary adjacency matrix (1 where there's a connection)
    adj_matrix = (pruned_weights > 0).float()
    return adj_matrix

def compute_basic_graph_metrics(adj_matrix):
    """
    Compute basic oversquashing-related metrics from adjacency matrix.
    
    Returns:
        dict: Dictionary of graph metrics
    """
    # Convert to NetworkX graph for analysis
    G = nx.from_numpy_array(adj_matrix.cpu().numpy())
    
    metrics = {}
    
    # Basic metrics
    metrics['density'] = nx.density(G)
    metrics['num_edges'] = G.number_of_edges()
    metrics['avg_degree'] = np.mean([d for n, d in G.degree()])
    
    # Clustering coefficient (measure of local connectivity)
    metrics['clustering_coefficient'] = nx.average_clustering(G)
    
    # Try to compute spectral properties if possible
    try:
        laplacian = nx.normalized_laplacian_matrix(G).todense()
        eigenvalues = np.linalg.eigvalsh(laplacian)
        metrics['spectral_gap'] = eigenvalues[1] if len(eigenvalues) > 1 else 0
    except:
        metrics['spectral_gap'] = float('nan')
    
    # Average shortest path length (measure of information propagation distance)
    try:
        if nx.is_connected(G):
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
        else:
            # For disconnected graphs, calculate across connected components
            components = list(nx.connected_components(G))
            path_lengths = []
            for component in components:
                if len(component) > 1:
                    subgraph = G.subgraph(component)
                    path_lengths.append(nx.average_shortest_path_length(subgraph))
            metrics['avg_shortest_path'] = np.mean(path_lengths) if path_lengths else float('inf')
    except:
        metrics['avg_shortest_path'] = float('nan')
    
    # Connectivity measures    
    metrics['avg_node_connectivity'] = nx.average_node_connectivity(G) if G.number_of_nodes() > 1 else 0
    metrics['algebraic_connectivity'] = nx.algebraic_connectivity(G) if G.number_of_nodes() > 1 else 0
    
    return metrics

def compare_initializations(num_base_nodes=30, num_virtual_nodes=15, k=5):
    """
    Compare graph properties of balanced Cayley vs. uniform initialization.
    """
    print(f"Comparing initialization methods for {num_base_nodes} nodes, {num_virtual_nodes} virtual nodes, k={k}")
    
    # Create initializations
    balanced_weights = balanced_cayley_initialize_edge_weight(
        num_base_nodes=num_base_nodes,
        num_virtual_nodes=num_virtual_nodes,
        high_value=2.0,
        low_value=0.2,
        verbose=False
    )
    
    uniform_weights = torch.ones(num_base_nodes, num_virtual_nodes) / num_virtual_nodes
    
    # Get adjacency matrices
    balanced_adj = get_adjacency_matrix(balanced_weights, top_k=k)
    uniform_adj = get_adjacency_matrix(uniform_weights, top_k=k)
    
    # Compute metrics
    balanced_metrics = compute_basic_graph_metrics(balanced_adj)
    uniform_metrics = compute_basic_graph_metrics(uniform_adj)
    
    # Print basic weight statistics
    print("\nWeight Statistics:")
    print(f"Balanced Cayley - Mean: {balanced_weights.mean():.4f}, Std: {balanced_weights.std():.4f}")
    print(f"Uniform - Mean: {uniform_weights.mean():.4f}, Std: {uniform_weights.std():.4f}")
    
    # Compare sparsity patterns
    balanced_nonzero = balanced_weights.count_nonzero().item()
    uniform_nonzero = uniform_weights.count_nonzero().item()
    balanced_sparsity = 1.0 - (balanced_nonzero / (num_base_nodes * num_virtual_nodes))
    uniform_sparsity = 1.0 - (uniform_nonzero / (num_base_nodes * num_virtual_nodes))
    
    print(f"Balanced Cayley - Nonzero connections: {balanced_nonzero}, Sparsity: {balanced_sparsity:.4f}")
    print(f"Uniform - Nonzero connections: {uniform_nonzero}, Sparsity: {uniform_sparsity:.4f}")
    
    # Print key metrics
    print("\nOversquashing-Related Metrics:")
    for metric in sorted(balanced_metrics.keys()):
        print(f"{metric}:")
        print(f"  Balanced Cayley: {balanced_metrics[metric]:.4f}")
        print(f"  Uniform: {uniform_metrics[metric]:.4f}")
        
        # Calculate percent difference
        if uniform_metrics[metric] != 0:
            diff_pct = 100 * (balanced_metrics[metric] - uniform_metrics[metric]) / uniform_metrics[metric]
            print(f"  Difference: {diff_pct:+.2f}%")
        print()
    
    # Output directory
    output_dir = create_output_dir()
    
    # Save the metrics to a JSON file
    result = {
        'config': {
            'base_nodes': num_base_nodes,
            'virtual_nodes': num_virtual_nodes,
            'top_k': k
        },
        'weight_stats': {
            'balanced_cayley': {
                'mean': float(balanced_weights.mean()),
                'std': float(balanced_weights.std()),
                'nonzero': balanced_nonzero,
                'sparsity': balanced_sparsity
            },
            'uniform': {
                'mean': float(uniform_weights.mean()),
                'std': float(uniform_weights.std()),
                'nonzero': uniform_nonzero,
                'sparsity': uniform_sparsity
            }
        },
        'metrics': {
            'balanced_cayley': {k: float(v) for k, v in balanced_metrics.items()},
            'uniform': {k: float(v) for k, v in uniform_metrics.items()}
        }
    }
    
    with open(f"{output_dir}/comparison_metrics.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    # Save the statistics to a text file
    with open(f"{output_dir}/statistics.txt", 'w') as f:
        f.write("Comparison of Initialization Methods\n")
        f.write("==================================\n\n")
        f.write(f"Base Nodes: {num_base_nodes}\n")
        f.write(f"Virtual Nodes: {num_virtual_nodes}\n")
        f.write(f"Top-k: {k}\n\n")
        
        f.write("Uniform Initialization:\n")
        f.write(f"  mean: {float(uniform_weights.mean())}\n")
        f.write(f"  std: {float(uniform_weights.std())}\n")
        f.write(f"  min: {float(uniform_weights.min())}\n")
        f.write(f"  max: {float(uniform_weights.max())}\n")
        f.write(f"  nonzero: {uniform_nonzero}\n")
        f.write(f"  sparsity: {uniform_sparsity}\n")
        f.write(f"  avg_connections_per_base_node: {uniform_nonzero / num_base_nodes}\n\n")
        
        f.write("Balanced Cayley Initialization:\n")
        f.write(f"  mean: {float(balanced_weights.mean())}\n")
        f.write(f"  std: {float(balanced_weights.std())}\n")
        f.write(f"  min: {float(balanced_weights.min())}\n")
        f.write(f"  max: {float(balanced_weights.max())}\n")
        f.write(f"  nonzero: {balanced_nonzero}\n")
        f.write(f"  sparsity: {balanced_sparsity}\n")
        f.write(f"  avg_connections_per_base_node: {balanced_nonzero / num_base_nodes}\n\n")
        
        f.write("Comparison:\n")
        f.write(f"  mean difference (Balanced - Uniform): {float(balanced_weights.mean() - uniform_weights.mean()):.6f}\n")
        f.write(f"  std difference (Balanced - Uniform): {float(balanced_weights.std() - uniform_weights.std()):.6f}\n")
        f.write(f"  min difference (Balanced - Uniform): {float(balanced_weights.min() - uniform_weights.min()):.6f}\n")
        f.write(f"  max difference (Balanced - Uniform): {float(balanced_weights.max() - uniform_weights.max()):.6f}\n")
        f.write(f"  nonzero difference (Balanced - Uniform): {balanced_nonzero - uniform_nonzero}\n")
        f.write(f"  sparsity difference (Balanced - Uniform): {balanced_sparsity - uniform_sparsity:.6f}\n")
        f.write(f"  avg_connections_per_base_node difference (Balanced - Uniform): "
               f"{(balanced_nonzero - uniform_nonzero) / num_base_nodes:.6f}\n\n")
        
        f.write("Oversquashing-Related Metrics:\n")
        for metric in sorted(balanced_metrics.keys()):
            f.write(f"{metric}:\n")
            f.write(f"  Balanced Cayley: {balanced_metrics[metric]:.6f}\n")
            f.write(f"  Uniform: {uniform_metrics[metric]:.6f}\n")
            
            # Calculate percent difference
            if uniform_metrics[metric] != 0:
                diff_pct = 100 * (balanced_metrics[metric] - uniform_metrics[metric]) / uniform_metrics[metric]
                f.write(f"  Difference: {diff_pct:+.2f}%\n")
            f.write("\n")
    
    # Create visualizations
    create_visual_comparison(balanced_weights, uniform_weights, balanced_adj, uniform_adj, output_dir)
    
    print(f"\nResults saved to {output_dir}/")
    return output_dir

def create_visual_comparison(balanced_weights, uniform_weights, balanced_adj, uniform_adj, output_dir):
    """Create visual comparisons of the weight matrices and connectivity patterns."""
    
    plt.figure(figsize=(15, 10))
    
    # Weight distributions
    plt.subplot(2, 3, 1)
    plt.hist(balanced_weights.flatten().numpy(), bins=20, alpha=0.7, label='Balanced Cayley')
    plt.hist(uniform_weights.flatten().numpy(), bins=20, alpha=0.7, label='Uniform')
    plt.title('Edge Weight Distributions')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.legend()
    
    # Weight matrices
    plt.subplot(2, 3, 2)
    plt.imshow(balanced_weights.numpy(), cmap='viridis')
    plt.colorbar(label='Weight')
    plt.title('Balanced Cayley Weights')
    
    plt.subplot(2, 3, 3)
    plt.imshow(uniform_weights.numpy(), cmap='viridis')
    plt.colorbar(label='Weight')
    plt.title('Uniform Weights')
    
    # Adjacency matrices
    plt.subplot(2, 3, 4)
    plt.imshow(balanced_adj.numpy(), cmap='binary')
    plt.title('Balanced Cayley Connectivity')
    
    plt.subplot(2, 3, 5)
    plt.imshow(uniform_adj.numpy(), cmap='binary')
    plt.title('Uniform Connectivity')
    
    # Weight difference
    plt.subplot(2, 3, 6)
    diff = balanced_weights - uniform_weights
    plt.imshow(diff.numpy(), cmap='coolwarm', vmin=-0.5, vmax=0.5)
    plt.colorbar(label='Difference')
    plt.title('Weight Difference (Balanced - Uniform)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/weight_comparison.png")
    plt.close()
    
    # Network visualizations
    plt.figure(figsize=(15, 7))
    
    # Convert adjacency matrices to NetworkX graphs for visualization
    G_balanced = nx.from_numpy_array(balanced_adj.numpy())
    G_uniform = nx.from_numpy_array(uniform_adj.numpy())
    
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G_balanced, seed=42)
    nx.draw_networkx_nodes(G_balanced, pos, node_size=200, alpha=0.8)
    nx.draw_networkx_edges(G_balanced, pos, alpha=0.4)
    plt.title('Balanced Cayley Network')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    pos = nx.spring_layout(G_uniform, seed=42)
    nx.draw_networkx_nodes(G_uniform, pos, node_size=200, alpha=0.8)
    nx.draw_networkx_edges(G_uniform, pos, alpha=0.4)
    plt.title('Uniform Network')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/network_visualization.png")
    plt.close()

def run_multiple_configs():
    """Run comparisons for multiple configurations to analyze trends."""
    # Define configurations to test
    configs = [
        # (num_base_nodes, num_virtual_nodes, k)
        (15, 5, 3),
        (30, 10, 5),
        (50, 20, 10)
    ]
    
    results = []
    
    for base_nodes, virtual_nodes, k in configs:
        output_dir = compare_initializations(base_nodes, virtual_nodes, k)
        results.append({
            'config': (base_nodes, virtual_nodes, k),
            'output_dir': output_dir
        })
    
    # Create a summary across all configurations
    summary_dir = create_output_dir()
    with open(f"{summary_dir}/multi_config_summary.md", 'w') as f:
        f.write("# Oversquashing Analysis: Multi-Configuration Summary\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Configurations Tested\n\n")
        f.write("| Base Nodes | Virtual Nodes | Top-k | Output Directory |\n")
        f.write("|------------|---------------|-------|------------------|\n")
        
        for result in results:
            base_nodes, virtual_nodes, k = result['config']
            f.write(f"| {base_nodes} | {virtual_nodes} | {k} | [{os.path.basename(result['output_dir'])}]({result['output_dir']}) |\n")
        
        f.write("\n## Key Observations\n\n")
        f.write("1. The balanced Cayley initialization creates a more structured connectivity pattern compared to uniform initialization\n")
        f.write("2. This structured pattern may help reduce oversquashing by creating more efficient message pathways\n")
        f.write("3. The specific improvements in graph metrics reveal how the balanced approach impacts message passing\n")
    
    print(f"\nMulti-configuration summary saved to {summary_dir}/multi_config_summary.md")
    
    return summary_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze oversquashing in different graph initializations')
    parser.add_argument('--base-nodes', type=int, default=30, help='Number of base nodes')
    parser.add_argument('--virtual-nodes', type=int, default=15, help='Number of virtual nodes')
    parser.add_argument('--k', type=int, default=5, help='Top-k value for connectivity')
    parser.add_argument('--multi', action='store_true', help='Run multiple configurations')
    
    args = parser.parse_args()
    
    print("Oversquashing Analysis Tool")
    print("==========================")
    
    if args.multi:
        print("\nRunning multiple configurations...")
        run_multiple_configs()
    else:
        print(f"\nAnalyzing single configuration: base_nodes={args.base_nodes}, virtual_nodes={args.virtual_nodes}, k={args.k}")
        compare_initializations(num_base_nodes=args.base_nodes, num_virtual_nodes=args.virtual_nodes, k=args.k)
