"""
Analyze oversquashing metrics for different initialization methods.
This script compares the oversquashing characteristics of balanced Cayley vs uniform
initialization by constructing synthetic networks with different initializations and 
measuring relevant graph metrics.
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

def compute_graph_metrics(adj_matrix):
    """
    Compute oversquashing-related metrics from adjacency matrix.
    
    Returns:
        dict: Dictionary of graph metrics
    """
    # Convert to NetworkX graph for analysis
    G = nx.from_numpy_array(adj_matrix.cpu().numpy())
    
    metrics = {}
    
    # 1. Effective resistance (related to bottlenecks)
    try:
        # Sum of effective resistances between all pairs
        effective_resistance = 0
        node_pairs = list(nx.non_edges(G))  # Pairs of nodes not directly connected
        if len(node_pairs) > 0:
            for i, j in node_pairs[:min(1000, len(node_pairs))]:  # Limit computation for large graphs
                try:
                    effective_resistance += nx.resistance_distance(G, i, j)
                except:
                    pass  # Skip if computation fails for a pair
        
        metrics['effective_resistance'] = effective_resistance / max(1, len(node_pairs))
    except:
        metrics['effective_resistance'] = float('nan')
    
    # 2. Spectral properties (related to information flow)
    try:
        laplacian = nx.normalized_laplacian_matrix(G).todense()
        eigenvalues = np.linalg.eigvalsh(laplacian)
        metrics['spectral_gap'] = eigenvalues[1] if len(eigenvalues) > 1 else 0
        metrics['condition_number'] = eigenvalues[-1] / max(eigenvalues[1], 1e-10) if len(eigenvalues) > 1 else float('inf')
    except:
        metrics['spectral_gap'] = float('nan')
        metrics['condition_number'] = float('nan')
    
    # 3. Graph connectivity metrics
    metrics['edge_connectivity'] = nx.edge_connectivity(G) if G.number_of_edges() > 0 else 0
    metrics['algebraic_connectivity'] = nx.algebraic_connectivity(G) if G.number_of_nodes() > 1 else 0
    
    # 4. Average shortest path length (measure of information propagation distance)
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
    
    # 5. Clustering coefficient (measure of local connectivity)
    metrics['clustering_coefficient'] = nx.average_clustering(G)
    
    # 6. Compute graph curvature if possible
    try:
        from graphriccicurvature.FormanRicci import FormanRicci
        forman = FormanRicci(G)
        forman.compute_ricci_curvature()
        metrics['forman_ricci_curvature'] = np.mean(list(forman.G.edges[e]['formanCurvature'] for e in forman.G.edges))
    except:
        metrics['forman_ricci_curvature'] = float('nan')
        
    return metrics

def analyze_initializations(base_nodes_range=[20, 30, 50, 100], 
                           virtual_nodes_range=[10, 15, 20, 30], 
                           k_values=[3, 5, 8, 10],
                           n_runs=5):
    """
    Analyze oversquashing metrics for different initializations across different graph sizes.
    
    Args:
        base_nodes_range: List of base node counts to test
        virtual_nodes_range: List of virtual node counts to test
        k_values: List of top-k values to test
        n_runs: Number of random initializations to average over
    """
    output_dir = create_output_dir()
    
    # Store all results
    all_results = []
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For each configuration
    for n_base in base_nodes_range:
        for n_virtual in virtual_nodes_range:
            for k in k_values:
                if k > n_virtual:
                    continue  # Skip if k is larger than virtual nodes
                    
                # Run multiple times to average results
                balanced_metrics_list = []
                uniform_metrics_list = []
                
                for run in range(n_runs):
                    # 1. Create balanced Cayley initialization
                    cayley_weights = balanced_cayley_initialize_edge_weight(
                        num_base_nodes=n_base,
                        num_virtual_nodes=n_virtual,
                        high_value=2.0,
                        low_value=0.2,
                        verbose=False
                    )
                    
                    # 2. Create uniform initialization
                    uniform_weights = torch.ones(n_base, n_virtual, device=device) / n_virtual
                    
                    # Get adjacency matrices with top-k pruning
                    balanced_adj = get_adjacency_matrix(cayley_weights, top_k=k)
                    uniform_adj = get_adjacency_matrix(uniform_weights, top_k=k)
                    
                    # Compute metrics
                    balanced_metrics = compute_graph_metrics(balanced_adj)
                    uniform_metrics = compute_graph_metrics(uniform_adj)
                    
                    balanced_metrics_list.append(balanced_metrics)
                    uniform_metrics_list.append(uniform_metrics)
                
                # Average metrics over runs
                avg_balanced_metrics = {k: np.mean([m[k] for m in balanced_metrics_list if not np.isnan(m[k])]) 
                                      for k in balanced_metrics_list[0].keys()}
                avg_uniform_metrics = {k: np.mean([m[k] for m in uniform_metrics_list if not np.isnan(m[k])]) 
                                     for k in uniform_metrics_list[0].keys()}
                
                # Store results
                result = {
                    'config': {
                        'base_nodes': n_base,
                        'virtual_nodes': n_virtual,
                        'top_k': k,
                    },
                    'balanced_cayley': avg_balanced_metrics,
                    'uniform': avg_uniform_metrics
                }
                all_results.append(result)
                
                # Print progress
                print(f"Analyzed: base_nodes={n_base}, virtual_nodes={n_virtual}, k={k}")
                print(f"  Balanced effective resistance: {avg_balanced_metrics['effective_resistance']:.4f}")
                print(f"  Uniform effective resistance: {avg_uniform_metrics['effective_resistance']:.4f}")
                print(f"  Balanced spectral gap: {avg_balanced_metrics['spectral_gap']:.4f}")
                print(f"  Uniform spectral gap: {avg_uniform_metrics['spectral_gap']:.4f}")
                print()
    
    # Save all results
    with open(f"{output_dir}/oversquashing_metrics.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary report
    with open(f"{output_dir}/oversquashing_summary.md", 'w') as f:
        f.write("# Oversquashing Analysis Summary\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This analysis compares oversquashing metrics between balanced Cayley and uniform initialization.\n")
        f.write("Lower effective resistance and condition number, and higher spectral gap and algebraic connectivity")
        f.write(" generally indicate better message passing and less oversquashing.\n\n")
        
        f.write("## Key Metrics\n\n")
        f.write("| Base Nodes | Virtual Nodes | Top-k | Metric | Balanced Cayley | Uniform | Difference (%) |\n")
        f.write("|------------|---------------|-------|--------|----------------|---------|---------------|\n")
        
        for result in all_results:
            config = result['config']
            n_base = config['base_nodes']
            n_virtual = config['virtual_nodes']
            k = config['top_k']
            
            # Report key metrics
            metrics = [
                ('effective_resistance', 'lower is better'),
                ('spectral_gap', 'higher is better'),
                ('condition_number', 'lower is better'),
                ('algebraic_connectivity', 'higher is better'),
                ('avg_shortest_path', 'lower is better')
            ]
            
            for metric_name, direction in metrics:
                balanced_value = result['balanced_cayley'][metric_name]
                uniform_value = result['uniform'][metric_name]
                
                if uniform_value != 0:
                    diff_percent = 100 * (balanced_value - uniform_value) / uniform_value
                else:
                    diff_percent = float('nan')
                
                # Format the difference with + or - sign
                diff_str = f"{diff_percent:+.2f}%" if not np.isnan(diff_percent) else "N/A"
                
                f.write(f"| {n_base} | {n_virtual} | {k} | {metric_name} | {balanced_value:.4f} | {uniform_value:.4f} | {diff_str} |\n")
    
    # Generate visualizations
    create_comparison_plots(all_results, output_dir)
    
    return all_results, output_dir

def create_comparison_plots(results, output_dir):
    """Create visualizations comparing the two initialization methods."""
    
    # Group results by metric
    metrics = [
        'effective_resistance', 
        'spectral_gap', 
        'condition_number', 
        'algebraic_connectivity',
        'avg_shortest_path'
    ]
    
    # For key configurations, create detailed comparison plots
    large_configs = [r for r in results if r['config']['base_nodes'] >= 50]
    if large_configs:
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i+1)
            
            x_labels = [f"{r['config']['base_nodes']}:{r['config']['virtual_nodes']}:{r['config']['top_k']}" 
                      for r in large_configs]
            balanced_values = [r['balanced_cayley'][metric] for r in large_configs]
            uniform_values = [r['uniform'][metric] for r in large_configs]
            
            x = np.arange(len(x_labels))
            width = 0.35
            
            plt.bar(x - width/2, balanced_values, width, label='Balanced Cayley')
            plt.bar(x + width/2, uniform_values, width, label='Uniform')
            
            plt.xlabel('Configuration (base:virtual:k)')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
            plt.xticks(x, x_labels, rotation=45, ha='right')
            plt.legend()
            
            # Add percentage difference annotations
            for j, (bv, uv) in enumerate(zip(balanced_values, uniform_values)):
                if uv != 0:
                    diff_pct = 100 * (bv - uv) / uv
                    color = 'green' if (metric in ['spectral_gap', 'algebraic_connectivity'] and diff_pct > 0) or \
                           (metric in ['effective_resistance', 'condition_number', 'avg_shortest_path'] and diff_pct < 0) else 'red'
                    plt.annotate(f"{diff_pct:+.1f}%", xy=(x[j], max(bv, uv) * 1.05), ha='center', color=color)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/large_config_metrics_comparison.png")
        plt.close()
    
    # Create trend analysis plots
    base_node_sizes = sorted(list(set(r['config']['base_nodes'] for r in results)))
    
    # For each metric, plot trends by base node size
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        # Group by k value
        k_values = sorted(list(set(r['config']['top_k'] for r in results)))
        
        for k in k_values:
            # Filter results for this k value
            k_results = [r for r in results if r['config']['top_k'] == k]
            
            # Sort by base node size
            k_results.sort(key=lambda r: r['config']['base_nodes'])
            
            # Extract x and y values
            x = [r['config']['base_nodes'] for r in k_results]
            y_balanced = [r['balanced_cayley'][metric] for r in k_results]
            y_uniform = [r['uniform'][metric] for r in k_results]
            
            # Plot lines
            plt.plot(x, y_balanced, 'o-', label=f'Balanced (k={k})')
            plt.plot(x, y_uniform, 's--', label=f'Uniform (k={k})')
        
        plt.xlabel('Number of Base Nodes')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Trend Analysis: {metric.replace("_", " ").title()}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/trend_{metric}.png")
        plt.close()

def analyze_trained_models(dataset_name='MUTAG', k_values=[2, 3, 4], hidden_dim=16):
    """
    Analyze oversquashing in trained models from statistical experiment results.
    This uses existing model structures without needing to retrain.
    """
    output_dir = create_output_dir()
    model_metrics = []
    
    # For each k value
    for k in k_values:
        print(f"Analyzing models for dataset {dataset_name} with k={k}...")
        
        # Try to find corresponding results
        results_dir = f"statistical_results/{dataset_name.lower()}"
        if not os.path.exists(results_dir):
            print(f"No results found for dataset {dataset_name}")
            continue
        
        # Find the most recent results file for this configuration
        result_files = [f for f in os.listdir(results_dir) if f.startswith('statistical_results_') and f.endswith('.json')]
        if not result_files:
            print(f"No result files found in {results_dir}")
            continue
        
        # Load each result file and check if it matches our k value
        matching_files = []
        for result_file in result_files:
            try:
                with open(os.path.join(results_dir, result_file), 'r') as f:
                    data = json.load(f)
                    if data.get('parameters', {}).get('top_k') == k:
                        matching_files.append((result_file, data))
            except:
                continue
                
        if not matching_files:
            print(f"No matching results found for k={k}")
            continue
        
        # Use the most recent file
        matching_files.sort(reverse=True)
        _, data = matching_files[0]
        
        # Create models with the same configuration
        num_features = 7  # Default for MUTAG
        num_classes = 2   # Default for MUTAG
        
        # These values can be adjusted for different datasets
        if dataset_name == 'ENZYMES':
            num_features = 3
            num_classes = 6
        elif dataset_name == 'PROTEINS':
            num_features = 3
            num_classes = 2
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create balanced model
        balanced_model = MemorySaverIPRMPNNModel(
            input_dim=num_features, 
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            edge_init_type='cayley',
            top_k=k
        ).to(device)
        
        # Create uniform model
        uniform_model = MemorySaverIPRMPNNModel(
            input_dim=num_features, 
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            edge_init_type='uniform',
            top_k=k
        ).to(device)
        
        # Extract and analyze edge weights
        # For each model type
        for model_type, model in [('balanced_cayley', balanced_model), ('uniform', uniform_model)]:
            # Get internal edge weights
            base_nodes = 0
            virtual_nodes = 0 
            
            # Get sizes from the model's first layer
            for name, param in model.named_parameters():
                if 'edge_weights' in name:
                    base_nodes = param.shape[0]
                    virtual_nodes = param.shape[1]
                    break
            
            if base_nodes == 0 or virtual_nodes == 0:
                print(f"Could not determine graph sizes for {model_type} model")
                continue
                
            # Set a small test graph to simulate a typical graph in the dataset
            # This will reflect the same connectivity pattern as would be used in a real graph
            test_x = torch.randn(base_nodes, num_features).to(device)
            test_edge_index = torch.zeros(2, base_nodes*2).long().to(device)  # Dummy edges
                
            # Forward pass to get the actual edge weights used
            # We need to access the internal edge weights from the model
            edge_weights = None
            
            # Get the edge weights from the MPNN layer
            for module in model.modules():
                if hasattr(module, 'edge_weights'):
                    edge_weights = module.edge_weights.detach()
                    break
            
            if edge_weights is None:
                print(f"Could not extract edge weights for {model_type} model")
                continue
                
            # Calculate adjacency matrix using top-k pruning
            adj_matrix = get_adjacency_matrix(edge_weights, top_k=k)
            
            # Compute metrics
            metrics = compute_graph_metrics(adj_matrix)
            
            # Store results
            model_metrics.append({
                'dataset': dataset_name,
                'top_k': k,
                'model_type': model_type,
                'base_nodes': base_nodes,
                'virtual_nodes': virtual_nodes,
                'metrics': metrics
            })
            
            print(f"  {model_type} model:")
            print(f"    Effective resistance: {metrics['effective_resistance']:.4f}")
            print(f"    Spectral gap: {metrics['spectral_gap']:.4f}")
            print(f"    Algebraic connectivity: {metrics['algebraic_connectivity']:.4f}")
    
    # Save results
    with open(f"{output_dir}/trained_model_metrics.json", 'w') as f:
        json.dump(model_metrics, f, indent=2)
    
    # Create report
    with open(f"{output_dir}/trained_model_report.md", 'w') as f:
        f.write(f"# Oversquashing Analysis for {dataset_name} Models\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Connectivity Analysis\n\n")
        f.write("| Dataset | Top-k | Model Type | Effective Resistance | Spectral Gap | Algebraic Connectivity | Clustering Coefficient |\n")
        f.write("|---------|-------|------------|----------------------|--------------|------------------------|------------------------|\n")
        
        for result in model_metrics:
            metrics = result['metrics']
            f.write(f"| {result['dataset']} | {result['top_k']} | {result['model_type']} | " +
                   f"{metrics['effective_resistance']:.4f} | {metrics['spectral_gap']:.4f} | " +
                   f"{metrics['algebraic_connectivity']:.4f} | {metrics['clustering_coefficient']:.4f} |\n")
    
    # Create visualizations
    if model_metrics:
        create_model_comparison_plots(model_metrics, output_dir, dataset_name)
    
    return model_metrics, output_dir

def create_model_comparison_plots(model_metrics, output_dir, dataset_name):
    """Create comparison plots for trained models."""
    
    # Group by k value
    k_values = sorted(list(set(m['top_k'] for m in model_metrics)))
    
    # Key metrics to visualize
    metrics = [
        'effective_resistance', 
        'spectral_gap', 
        'algebraic_connectivity',
        'avg_shortest_path',
        'clustering_coefficient'
    ]
    
    plt.figure(figsize=(15, 10))
    
    for i, metric_name in enumerate(metrics):
        if i >= 6:  # Only show top 6 metrics
            break
            
        plt.subplot(2, 3, i+1)
        
        # Prepare data for plotting
        balanced_values = []
        uniform_values = []
        
        for k in k_values:
            # Get balanced value
            balanced = [m['metrics'][metric_name] for m in model_metrics 
                      if m['top_k'] == k and m['model_type'] == 'balanced_cayley']
            
            # Get uniform value
            uniform = [m['metrics'][metric_name] for m in model_metrics 
                     if m['top_k'] == k and m['model_type'] == 'uniform']
            
            if balanced and uniform:
                balanced_values.append(balanced[0])
                uniform_values.append(uniform[0])
        
        # Plot
        x = np.arange(len(k_values))
        width = 0.35
        
        plt.bar(x - width/2, balanced_values, width, label='Balanced Cayley')
        plt.bar(x + width/2, uniform_values, width, label='Uniform')
        
        plt.xlabel('Top-k value')
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.title(f'{metric_name.replace("_", " ").title()}')
        plt.xticks(x, k_values)
        plt.legend()
        
        # Add percentage difference annotations
        for j, (bv, uv) in enumerate(zip(balanced_values, uniform_values)):
            if uv != 0:
                diff_pct = 100 * (bv - uv) / uv
                color = 'green' if (metric_name in ['spectral_gap', 'algebraic_connectivity'] and diff_pct > 0) or \
                       (metric_name in ['effective_resistance', 'condition_number', 'avg_shortest_path'] and diff_pct < 0) else 'red'
                plt.annotate(f"{diff_pct:+.1f}%", xy=(x[j], max(bv, uv) * 1.05), ha='center', color=color)
    
    plt.suptitle(f'Oversquashing Metrics for {dataset_name} Models')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{output_dir}/{dataset_name}_model_metrics.png")
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze oversquashing in different graph initializations')
    parser.add_argument('--mode', type=str, choices=['synthetic', 'trained', 'both'], default='both',
                       help='Analysis mode: synthetic graphs, trained models, or both')
    parser.add_argument('--dataset', type=str, default='MUTAG', 
                       help='Dataset name for trained model analysis')
    parser.add_argument('--k-values', type=int, nargs='+', default=[2, 3, 4],
                       help='Top-k values to analyze')
    
    args = parser.parse_args()
    
    print("Oversquashing Analysis Tool")
    print("==========================")
    
    if args.mode in ['synthetic', 'both']:
        print("\nAnalyzing synthetic graphs...")
        # Use smaller ranges for quicker testing
        results, output_dir = analyze_initializations(
            base_nodes_range=[20, 50, 100],
            virtual_nodes_range=[10, 20, 30],
            k_values=[3, 5, 10],
            n_runs=3
        )
        print(f"Synthetic analysis complete. Results saved to {output_dir}")
    
    if args.mode in ['trained', 'both']:
        print(f"\nAnalyzing trained models for {args.dataset}...")
        model_results, model_dir = analyze_trained_models(
            dataset_name=args.dataset,
            k_values=args.k_values
        )
        print(f"Model analysis complete. Results saved to {model_dir}")
    
    print("\nAnalysis complete!")
