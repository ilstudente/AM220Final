"""
Direct comparison of balanced Cayley vs. uniform initialization for oversquashing metrics.
This script explicitly creates different graph structures for each initialization type
to show the difference in oversquashing properties.
"""

import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from datetime import datetime

from models.memory_saver_iprmpnn import MemorySaverIPRMPNNModel
from utils.oversquashing_metrics import compute_oversquashing_metric, compute_graph_connectivity_metrics

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def compute_cheeger_constant(G):
    """
    Compute the Cheeger constant (conductance) of a graph.
    Lower values indicate potential bottlenecks for message passing.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Cheeger constant value
    """
    # If graph is not connected, return 0
    if not nx.is_connected(G):
        return 0.0
    
    n = G.number_of_nodes()
    if n <= 1:
        return 1.0  # By definition
    
    # For small graphs, compute exactly
    if n <= 20:
        min_conductance = 1.0
        for i in range(1, 2**(n-1)):  # Check all possible cuts
            # Convert i to binary to represent a cut
            binary = format(i, f'0{n}b')
            set_s = {j for j in range(n) if binary[j] == '1'}
            if len(set_s) == 0 or len(set_s) == n:
                continue
                
            # Count edges crossing the cut
            cut_size = 0
            for u, v in G.edges():
                if (u in set_s and v not in set_s) or (u not in set_s and v in set_s):
                    cut_size += 1
            
            vol_s = sum(dict(G.degree()).get(v, 0) for v in set_s)
            vol_complement = 2 * G.number_of_edges() - vol_s
            denominator = min(vol_s, vol_complement)
            
            if denominator > 0:
                conductance = cut_size / denominator
                min_conductance = min(min_conductance, conductance)
        
        return float(min_conductance)
    else:
        # For larger graphs, approximate using the second smallest eigenvalue of normalized Laplacian
        try:
            L = nx.normalized_laplacian_matrix(G)
            eigenvalues = np.sort(np.linalg.eigvalsh(L.toarray()))
            return float(eigenvalues[1]) / 2.0  # λ₂/2 is a lower bound on the Cheeger constant
        except:
            return 0.0

def analyze_model_edges(model, data, graph_index=0):
    """
    Analyze edge weights from a model after forward pass
    
    Args:
        model: IPR-MPNN model
        data: PyG data object
        graph_index: Index of the graph to analyze
        
    Returns:
        Dict with graph analysis results
    """
    # Get edge weights from model
    weights_info = model.get_final_edge_weights(graph_index)
    
    if not weights_info or 'edge_weights' not in weights_info:
        return None
    
    # Extract relevant information
    edge_weights = weights_info['edge_weights']
    num_nodes = weights_info['num_nodes']
    num_virtual = weights_info['num_virtual_nodes']
    
    # Create a graph with learned connectivity
    # This is a direct bipartite graph from real to virtual nodes
    G_bipartite = nx.Graph()
    
    # Add nodes
    for i in range(num_nodes):
        G_bipartite.add_node(i, bipartite=0)  # Real nodes
    
    for j in range(num_virtual):
        G_bipartite.add_node(f'v{j}', bipartite=1)  # Virtual nodes
    
    # Add edges with weights
    for i in range(num_nodes):
        for j in range(num_virtual):
            weight = edge_weights[i, j].item()
            if weight > 0:
                G_bipartite.add_edge(i, f'v{j}', weight=weight)
    
    # Create an effective graph between real nodes (projection)
    G_effective = nx.Graph()
    for i in range(num_nodes):
        G_effective.add_node(i)
    
    # Connect real nodes that share virtual nodes
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            # Find shared virtual nodes
            i_neighbors = set(n for n in G_bipartite.neighbors(i) if isinstance(n, str))
            j_neighbors = set(n for n in G_bipartite.neighbors(j) if isinstance(n, str))
            shared_virtual = i_neighbors.intersection(j_neighbors)
            
            if shared_virtual:
                # Sum of product of weights to shared virtual nodes
                weight = sum(G_bipartite[i][v]['weight'] * G_bipartite[j][v]['weight'] 
                            for v in shared_virtual)
                G_effective.add_edge(i, j, weight=weight)
    
    # Calculate metrics for the effective graph
    metrics = {}
    
    # Basic graph properties
    metrics['num_nodes'] = num_nodes
    metrics['num_virtual_nodes'] = num_virtual
    metrics['num_edges'] = G_effective.number_of_edges()
    metrics['density'] = nx.density(G_effective)
    
    # Connectivity metrics
    try:
        if nx.is_connected(G_effective):
            metrics['avg_path_length'] = nx.average_shortest_path_length(G_effective, weight='weight')
            metrics['diameter'] = nx.diameter(G_effective, weight='weight')
        else:
            # For disconnected graphs, compute for largest component
            largest_cc = max(nx.connected_components(G_effective), key=len)
            subgraph = G_effective.subgraph(largest_cc)
            metrics['avg_path_length'] = nx.average_shortest_path_length(subgraph, weight='weight')
            metrics['diameter'] = nx.diameter(subgraph, weight='weight')
            metrics['largest_cc_size'] = len(largest_cc) / num_nodes
    except:
        metrics['avg_path_length'] = float('inf')
        metrics['diameter'] = float('inf')
        
    # Spectral properties
    try:
        L = nx.normalized_laplacian_matrix(G_effective)
        eigenvalues = np.sort(np.linalg.eigvalsh(L.toarray()))
        metrics['spectral_gap'] = eigenvalues[1] if len(eigenvalues) > 1 else 0
        metrics['algebraic_connectivity'] = eigenvalues[1] if len(eigenvalues) > 1 else 0
    except:
        metrics['spectral_gap'] = 0
        metrics['algebraic_connectivity'] = 0
    
    # Oversquashing-related metrics
    metrics['cheeger_constant'] = compute_cheeger_constant(G_effective)
    
    # Edge weight statistics
    edge_weights_list = [d['weight'] for _, _, d in G_effective.edges(data=True)]
    if edge_weights_list:
        metrics['mean_edge_weight'] = np.mean(edge_weights_list)
        metrics['max_edge_weight'] = np.max(edge_weights_list)
        metrics['min_edge_weight'] = np.min(edge_weights_list)
        metrics['std_edge_weight'] = np.std(edge_weights_list)
    else:
        metrics['mean_edge_weight'] = 0
        metrics['max_edge_weight'] = 0
        metrics['min_edge_weight'] = 0
        metrics['std_edge_weight'] = 0
    
    return {
        'metrics': metrics,
        'effective_graph': G_effective,
        'bipartite_graph': G_bipartite
    }

def compare_initializations(dataset_name, k=3, hidden_dim=16, num_graphs=5, seed=42):
    """
    Compare the oversquashing properties of balanced Cayley vs. uniform initialization
    by directly analyzing the edge weights and connectivity patterns.
    
    Args:
        dataset_name: Name of the dataset (MUTAG, PROTEINS, ENZYMES)
        k: Number of top-k connections
        hidden_dim: Hidden dimension size
        num_graphs: Number of graphs to analyze
        seed: Random seed
        
    Returns:
        Dict with comparison results
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"oversquashing_analysis/direct_comparison_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    dataset = TUDataset(root='/workspaces/IPR-MPNN/datasets', name=dataset_name)
    print(f"Dataset: {dataset_name}, Graphs: {len(dataset)}")
    
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
    
    # Collect comparison results
    results = []
    
    # Process each graph
    for idx, data in enumerate(tqdm(data_loader, desc=f"Analyzing {dataset_name} graphs")):
        data = data.to(device)
        
        with torch.no_grad():
            # Forward pass through balanced model
            print(f"\nProcessing graph {idx} with balanced model")
            balanced_model(data)
            
            # Forward pass through uniform model
            print(f"Processing graph {idx} with uniform model")
            uniform_model(data)
            
            # Get edge weights and analyze
            print(f"Analyzing edge weights for graph {idx}")
            balanced_analysis = analyze_model_edges(balanced_model, data, 0)
            uniform_analysis = analyze_model_edges(uniform_model, data, 0)
            
            if balanced_analysis and uniform_analysis:
                # Compare the analyses
                comparison = {
                    'graph_idx': idx,
                    'num_nodes': data.num_nodes,
                    'num_edges': data.edge_index.size(1) // 2,  # Assuming undirected
                    'balanced': balanced_analysis['metrics'],
                    'uniform': uniform_analysis['metrics']
                }
                
                # Calculate differences
                diff_metrics = {}
                for key in balanced_analysis['metrics'].keys():
                    if isinstance(balanced_analysis['metrics'][key], (int, float)) and \
                       isinstance(uniform_analysis['metrics'][key], (int, float)):
                        
                        balanced_val = balanced_analysis['metrics'][key]
                        uniform_val = uniform_analysis['metrics'][key]
                        
                        if balanced_val != float('inf') and uniform_val != float('inf'):
                            diff_metrics[key] = balanced_val - uniform_val
                
                comparison['differences'] = diff_metrics
                results.append(comparison)
                
                # Save graph visualizations
                if idx < 3:  # Only visualize first few graphs
                    # Visualize balanced model effective graph
                    plt.figure(figsize=(10, 8))
                    G = balanced_analysis['effective_graph']
                    pos = nx.spring_layout(G, seed=42)
                    
                    # Edge weights as width
                    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
                    
                    nx.draw(G, pos, with_labels=True, node_color='skyblue', 
                           node_size=500, font_size=10, width=edge_weights)
                    plt.title(f"Graph {idx} - Balanced Cayley Effective Connectivity")
                    plt.savefig(os.path.join(output_dir, f"graph_{idx}_balanced_effective.png"))
                    plt.close()
                    
                    # Visualize uniform model effective graph
                    plt.figure(figsize=(10, 8))
                    G = uniform_analysis['effective_graph']
                    pos = nx.spring_layout(G, seed=42)
                    
                    # Edge weights as width
                    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
                    
                    nx.draw(G, pos, with_labels=True, node_color='lightgreen', 
                           node_size=500, font_size=10, width=edge_weights)
                    plt.title(f"Graph {idx} - Uniform Effective Connectivity")
                    plt.savefig(os.path.join(output_dir, f"graph_{idx}_uniform_effective.png"))
                    plt.close()
                
                print(f"Graph {idx}: Analysis completed")
                print(f"  Balanced Cheeger: {balanced_analysis['metrics']['cheeger_constant']:.4f}")
                print(f"  Uniform Cheeger: {uniform_analysis['metrics']['cheeger_constant']:.4f}")
                print(f"  Balanced Spectral Gap: {balanced_analysis['metrics']['spectral_gap']:.4f}")
                print(f"  Uniform Spectral Gap: {uniform_analysis['metrics']['spectral_gap']:.4f}")
            else:
                print(f"Graph {idx}: Failed to analyze")
    
    # Generate summary statistics across all graphs
    if results:
        summary = {
            'dataset': dataset_name,
            'num_graphs_analyzed': len(results),
            'balanced_vs_uniform': {},
            'overall_better': {}
        }
        
        # Compute averages for all metrics
        balanced_avgs = {}
        uniform_avgs = {}
        
        for metric in results[0]['balanced'].keys():
            try:
                balanced_values = [r['balanced'][metric] for r in results 
                                  if isinstance(r['balanced'][metric], (int, float)) 
                                  and r['balanced'][metric] != float('inf')]
                
                uniform_values = [r['uniform'][metric] for r in results
                                 if isinstance(r['uniform'][metric], (int, float))
                                 and r['uniform'][metric] != float('inf')]
                
                if balanced_values and uniform_values:
                    balanced_avg = np.mean(balanced_values)
                    uniform_avg = np.mean(uniform_values)
                    
                    balanced_avgs[metric] = balanced_avg
                    uniform_avgs[metric] = uniform_avg
                    
                    # Determine which is better (depends on metric)
                    if metric in ['cheeger_constant', 'spectral_gap', 'density', 'algebraic_connectivity']:
                        # Higher is better for these metrics
                        better = 'balanced' if balanced_avg > uniform_avg else 'uniform'
                    elif metric in ['avg_path_length', 'diameter']:
                        # Lower is better for these metrics
                        better = 'balanced' if balanced_avg < uniform_avg else 'uniform'
                    else:
                        better = 'balanced' if balanced_avg > uniform_avg else 'uniform'
                    
                    summary['balanced_vs_uniform'][metric] = {
                        'balanced_avg': float(balanced_avg),
                        'uniform_avg': float(uniform_avg),
                        'difference': float(balanced_avg - uniform_avg),
                        'percent_diff': float((balanced_avg - uniform_avg) / uniform_avg * 100) if uniform_avg != 0 else 0,
                        'better': better
                    }
            except Exception as e:
                print(f"Error computing average for {metric}: {e}")
        
        # Determine overall better approach
        connectivity_metrics = ['cheeger_constant', 'spectral_gap', 'algebraic_connectivity']
        path_metrics = ['avg_path_length', 'diameter']
        
        balanced_connectivity_wins = sum(1 for metric in connectivity_metrics 
                                        if metric in summary['balanced_vs_uniform'] and 
                                        summary['balanced_vs_uniform'][metric]['better'] == 'balanced')
        
        balanced_path_wins = sum(1 for metric in path_metrics
                               if metric in summary['balanced_vs_uniform'] and
                               summary['balanced_vs_uniform'][metric]['better'] == 'balanced')
        
        summary['overall_better']['connectivity'] = 'balanced' if balanced_connectivity_wins > len(connectivity_metrics) / 2 else 'uniform'
        summary['overall_better']['path_efficiency'] = 'balanced' if balanced_path_wins > len(path_metrics) / 2 else 'uniform'
        summary['overall_better']['oversquashing_reduction'] = summary['overall_better']['connectivity']
        
        # Save results
        with open(os.path.join(output_dir, f"{dataset_name.lower()}_direct_comparison.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        with open(os.path.join(output_dir, f"{dataset_name.lower()}_detailed_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualization
        visualize_comparison_results(results, summary, dataset_name, output_dir)
        
        return summary
    else:
        print("No comparison results generated")
        return None

def visualize_comparison_results(results, summary, dataset_name, output_dir):
    """
    Visualize the comparison results between balanced Cayley and uniform initialization.
    
    Args:
        results: List of comparison results for each graph
        summary: Summary statistics
        dataset_name: Name of the dataset
        output_dir: Output directory for visualizations
    """
    # Create bar plots for key metrics
    key_metrics = ['cheeger_constant', 'spectral_gap', 'avg_path_length', 'diameter', 'density']
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(key_metrics):
        if metric in summary['balanced_vs_uniform']:
            plt.subplot(2, 3, i+1)
            
            # Extract values for each graph
            balanced_values = [r['balanced'][metric] for r in results 
                              if metric in r['balanced'] and 
                              isinstance(r['balanced'][metric], (int, float)) and 
                              r['balanced'][metric] != float('inf')]
            
            uniform_values = [r['uniform'][metric] for r in results
                             if metric in r['uniform'] and
                             isinstance(r['uniform'][metric], (int, float)) and
                             r['uniform'][metric] != float('inf')]
            
            # Get indices for plotting
            indices = range(len(balanced_values))
            
            # Plot bars
            plt.bar([i-0.2 for i in indices], balanced_values, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
            plt.bar([i+0.2 for i in indices], uniform_values, width=0.4, label='Uniform', color='orange', alpha=0.7)
            
            plt.title(f"{metric.replace('_', ' ').title()}")
            plt.xlabel('Graph Index')
            plt.xticks(indices)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            if i == 0:
                plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name.lower()}_metrics_comparison.png"))
    plt.close()
    
    # Create summary plot comparing averages
    plt.figure(figsize=(12, 8))
    
    metrics = list(summary['balanced_vs_uniform'].keys())
    balanced_avgs = [summary['balanced_vs_uniform'][m]['balanced_avg'] for m in metrics]
    uniform_avgs = [summary['balanced_vs_uniform'][m]['uniform_avg'] for m in metrics]
    
    # Normalize values to make them comparable
    max_vals = [max(b, u) for b, u in zip(balanced_avgs, uniform_avgs)]
    norm_balanced = [b/m if m != 0 else 0 for b, m in zip(balanced_avgs, max_vals)]
    norm_uniform = [u/m if m != 0 else 0 for u, m in zip(uniform_avgs, max_vals)]
    
    x = range(len(metrics))
    plt.bar([i-0.2 for i in x], norm_balanced, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i+0.2 for i in x], norm_uniform, width=0.4, label='Uniform', color='orange', alpha=0.7)
    
    plt.title(f"Normalized Metric Comparison - {dataset_name}")
    plt.xlabel('Metric')
    plt.xticks(x, [m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name.lower()}_summary_comparison.png"))
    plt.close()
    
    # Create markdown report
    with open(os.path.join(output_dir, f"{dataset_name.lower()}_report.md"), 'w') as f:
        f.write(f"# Oversquashing Analysis: {dataset_name}\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Metric | Balanced Cayley | Uniform | Difference | Better |\n")
        f.write("|--------|----------------|---------|------------|--------|\n")
        
        for metric in sorted(summary['balanced_vs_uniform'].keys()):
            info = summary['balanced_vs_uniform'][metric]
            f.write(f"| {metric.replace('_', ' ').title()} | {info['balanced_avg']:.4f} | {info['uniform_avg']:.4f} | {info['difference']:.4f} ({info['percent_diff']:.1f}%) | {info['better'].title()} |\n")
        
        f.write("\n## Overall Assessment\n\n")
        f.write(f"- **Connectivity**: {summary['overall_better']['connectivity'].title()} approach is better\n")
        f.write(f"- **Path Efficiency**: {summary['overall_better']['path_efficiency'].title()} approach is better\n")
        f.write(f"- **Oversquashing Reduction**: {summary['overall_better']['oversquashing_reduction'].title()} approach is better\n\n")
        
        f.write("## Interpretation\n\n")
        f.write("- **Cheeger constant**: Higher values indicate better connectivity with fewer bottlenecks\n")
        f.write("- **Spectral gap**: Higher values indicate faster information mixing\n")
        f.write("- **Average path length**: Lower values indicate more efficient message passing\n")
        f.write("- **Diameter**: Lower values indicate shorter maximum distances between nodes\n")
        f.write("- **Density**: Higher values indicate more connections between nodes\n\n")
        
        f.write("## Individual Graph Results\n\n")
        for idx, result in enumerate(results):
            f.write(f"### Graph {idx}\n\n")
            f.write("| Metric | Balanced Cayley | Uniform | Difference |\n")
            f.write("|--------|----------------|---------|------------|\n")
            
            for metric in sorted(result['balanced'].keys()):
                if isinstance(result['balanced'][metric], (int, float)) and \
                   isinstance(result['uniform'][metric], (int, float)):
                    
                    b_val = result['balanced'][metric]
                    u_val = result['uniform'][metric]
                    
                    if b_val != float('inf') and u_val != float('inf'):
                        diff = b_val - u_val
                        f.write(f"| {metric.replace('_', ' ').title()} | {b_val:.4f} | {u_val:.4f} | {diff:.4f} |\n")
            
            f.write("\n")

if __name__ == "__main__":
    # Analyze MUTAG dataset
    print("\n=== Running MUTAG Analysis ===")
    mutag_results = compare_initializations(
        dataset_name="MUTAG",
        k=3,
        hidden_dim=16,
        num_graphs=3,  # Analyze just 3 graphs to test
        seed=42
    )
    
    # If MUTAG works, uncomment these for the other datasets
    # print("\n=== Running PROTEINS Analysis ===")
    # proteins_results = compare_initializations(
    #     dataset_name="PROTEINS",
    #     k=3,
    #     hidden_dim=16,
    #     num_graphs=10,
    #     seed=42
    # )
    
    # print("\n=== Running ENZYMES Analysis ===")
    # enzymes_results = compare_initializations(
    #     dataset_name="ENZYMES",
    #     k=3,
    #     hidden_dim=16,
    #     num_graphs=10,
    #     seed=42
    # )
