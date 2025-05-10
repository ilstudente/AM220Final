"""
Run experiments with oversquashing metrics collection for IPR-MPNN.

This script:
1. Trains models with both balanced Cayley and uniform initialization
2. Collects comprehensive oversquashing metrics including:
   - Cheeger constant (measure of bottlenecks)
   - Dirichlet energy (measure of smoothness)
   - Effective resistance (measure of information flow)
   - Spectral gap (mixing time)
   - Graph conductance
3. Analyzes how the learned connectivity patterns affect these metrics
4. Visualizes the results for comparison

Usage:
  python run_oversquashing_metrics_analysis.py --dataset MUTAG --epochs 15 --hidden 16 --k 3
"""

import os
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import scipy.sparse as sp
from datetime import datetime
from tqdm import tqdm
import argparse
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx, to_scipy_sparse_matrix
import gc

from models.memory_saver_iprmpnn import MemorySaverIPRMPNNModel
from utils.oversquashing_metrics import compute_oversquashing_metric, compute_graph_connectivity_metrics

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_cheeger_constant(adj_matrix):
    """
    Compute the Cheeger constant (isoperimetric number) of a graph.
    The Cheeger constant measures bottlenecks in the graph - lower values indicate worse bottlenecks.
    
    Args:
        adj_matrix: Adjacency matrix (scipy sparse or numpy array)
        
    Returns:
        The Cheeger constant value (float)
    """
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()
    
    # Convert to NetworkX graph
    if sp.issparse(adj_matrix):
        G = nx.from_scipy_sparse_matrix(adj_matrix)
    else:
        G = nx.from_numpy_array(adj_matrix)
    
    # Ensure graph is connected (or get largest component)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    # For very small graphs, return a default value
    if G.number_of_nodes() <= 2:
        return 1.0
    
    # Calculate the Cheeger constant through spectral approach
    try:
        laplacian = nx.normalized_laplacian_matrix(G).todense()
        eigenvalues = np.linalg.eigvalsh(laplacian)
        sorted_eigenvalues = np.sort(eigenvalues)
        
        # Second smallest eigenvalue (Fiedler value) bounds the Cheeger constant
        fiedler_value = sorted_eigenvalues[1] if len(sorted_eigenvalues) > 1 else 0
        
        # Cheeger's inequality: h²/2 ≤ λ₂ ≤ 2h, where h is Cheeger constant
        # We'll approximate h ≈ sqrt(2*λ₂) 
        cheeger_approx = np.sqrt(2 * fiedler_value)
        return float(cheeger_approx)
    except:
        # Fallback for numerical issues
        return 0.0

def compute_dirichlet_energy(adj_matrix, node_features=None):
    """
    Compute the Dirichlet energy of the graph, which measures 
    the smoothness of a signal defined on the vertices.
    
    Lower Dirichlet energy = smoother signal = better message passing.
    
    Args:
        adj_matrix: Adjacency matrix
        node_features: Node features to use as the signal (if None, use constant signal)
        
    Returns:
        Dirichlet energy value (float)
    """
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()
    
    # Convert to NetworkX graph
    if sp.issparse(adj_matrix):
        G = nx.from_scipy_sparse_matrix(adj_matrix)
    else:
        G = nx.from_numpy_array(adj_matrix)
    
    # Create Laplacian matrix
    laplacian = nx.laplacian_matrix(G).todense()
    
    # If no node features provided, use a constant signal
    if node_features is None:
        n = G.number_of_nodes()
        # We'll use a simple gradient signal - increasing values across nodes
        signal = np.linspace(0, 1, n)
    else:
        if isinstance(node_features, torch.Tensor):
            # Use mean across feature dimensions
            signal = node_features.mean(dim=1).cpu().numpy()
        else:
            signal = node_features
    
    # Compute the Dirichlet energy: f^T L f
    dirichlet_energy = np.dot(signal, np.dot(laplacian, signal))
    return float(dirichlet_energy)

def compute_graph_conductance(adj_matrix):
    """
    Compute the conductance of the graph, which measures how well-connected the graph is.
    Lower conductance indicates the presence of bottlenecks.
    
    Args:
        adj_matrix: Adjacency matrix
        
    Returns:
        Conductance value (float)
    """
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()
    
    # Convert to NetworkX graph
    if sp.issparse(adj_matrix):
        G = nx.from_scipy_sparse_matrix(adj_matrix)
    else:
        G = nx.from_numpy_array(adj_matrix)
    
    # Ensure graph is connected (or get largest component)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    # For very small graphs, return a default value
    if G.number_of_nodes() <= 2:
        return 1.0
    
    # Compute conductance through clustering (approximate)
    try:
        # Try spectral clustering to get a reasonable partition
        from sklearn.cluster import SpectralClustering
        n_clusters = min(2, G.number_of_nodes() - 1)
        clustering = SpectralClustering(n_clusters=n_clusters, 
                                       affinity='precomputed',
                                       assign_labels='discretize')
        adj_dense = nx.to_numpy_array(G)
        labels = clustering.fit_predict(adj_dense)
        
        # Compute conductance of the partition
        edge_cut = 0
        cluster_volumes = [0] * n_clusters
        
        # Count edges between and within clusters
        for u, v in G.edges():
            if labels[u] != labels[v]:
                edge_cut += 1
            
            cluster_volumes[labels[u]] += 1
            if u != v:  # Don't double count self-loops
                cluster_volumes[labels[v]] += 1
        
        # Total volume
        vol_G = sum(dict(G.degree()).values())
        
        # Find smaller volume of the two clusters
        smaller_vol = min([vol for vol in cluster_volumes if vol > 0])
        
        # Conductance = edge_cut / smaller_vol
        conductance = edge_cut / max(smaller_vol, 1)
        return float(conductance)
    except:
        # Fallback
        return 1.0

def compute_all_oversquashing_metrics(edge_index, edge_weights=None, node_features=None, num_nodes=None):
    """
    Compute a comprehensive set of oversquashing metrics for a graph.
    
    Args:
        edge_index: PyG edge_index or adjacency matrix
        edge_weights: Optional edge weights
        node_features: Optional node features
        num_nodes: Number of nodes (only needed for edge_index format)
        
    Returns:
        Dict with all oversquashing metrics
    """
    # Convert edge_index to adjacency matrix if needed
    if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 and edge_index.size(0) == 2:
        # Determine number of nodes
        if num_nodes is None:
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
        adj = edge_index.float()
    
    # Apply edge weights to the adjacency matrix if provided
    if edge_weights is not None and len(edge_weights) > 0:
        # Create weighted adjacency matrix
        rows, cols = adj.nonzero()
        if len(rows) > 0:  # Ensure there are edges
            # Check if we have enough weights
            if len(edge_weights) >= len(rows):
                adj_weighted = adj.clone()
                for i, (r, c) in enumerate(zip(rows, cols)):
                    adj_weighted[r, c] = edge_weights[i]
                adj = adj_weighted
    
    # Convert to NetworkX for analysis
    G = nx.from_numpy_array(adj.cpu().numpy())
    
    # Get the core metrics already implemented
    try:
        effective_resistance = compute_oversquashing_metric(adj, edge_weights)
        connectivity = compute_graph_connectivity_metrics(adj, edge_weights)
    except Exception as e:
        print(f"Error in core metrics: {e}")
        effective_resistance = {"mean_effective_resistance": float('inf')}
        connectivity = {"avg_path_length": float('inf'), "spectral_gap": 0}
    
    # Calculate additional metrics
    try:
        cheeger = compute_cheeger_constant(adj)
    except Exception as e:
        print(f"Error computing Cheeger constant: {e}")
        cheeger = 0.0
    
    try:
        dirichlet = compute_dirichlet_energy(adj, node_features)
    except Exception as e:
        print(f"Error computing Dirichlet energy: {e}")
        dirichlet = float('inf')
    
    try:
        conductance = compute_graph_conductance(adj)
    except Exception as e:
        print(f"Error computing conductance: {e}")
        conductance = 0.0
    
    # Calculate algebraic connectivity (Fiedler value) - closely related to cheeger constant
    try:
        laplacian = nx.laplacian_matrix(G).todense()
        eigenvalues = np.linalg.eigvalsh(laplacian)
        algebraic_connectivity = sorted(eigenvalues)[1] if len(eigenvalues) > 1 else 0
    except Exception as e:
        print(f"Error computing algebraic connectivity: {e}")
        algebraic_connectivity = 0.0
    
    # Return all metrics
    return {
        **effective_resistance,
        **connectivity,
        "cheeger_constant": float(cheeger),
        "dirichlet_energy": float(dirichlet),
        "conductance": float(conductance),
        "algebraic_connectivity": float(algebraic_connectivity)
    }

def create_weighted_adjacency(edge_index, edge_weights, num_nodes):
    """
    Create a weighted adjacency matrix from edge_index and edge_weights.
    
    Args:
        edge_index: Graph connectivity as edge_index
        edge_weights: Weights for each edge
        num_nodes: Number of nodes in the graph
        
    Returns:
        Weighted adjacency matrix
    """
    device = edge_index.device
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    
    if edge_weights is not None and len(edge_weights) > 0:
        # Add weights to adjacency matrix
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            adj[src, dst] = edge_weights[i]
    else:
        # Binary adjacency matrix
        adj[edge_index[0], edge_index[1]] = 1
    
    return adj

def analyze_model_oversquashing(model, data_loader, device, num_samples=5):
    """
    Analyze oversquashing metrics for a model on a set of test graphs.
    
    Args:
        model: Trained GNN model
        data_loader: DataLoader with test graphs
        device: Device to run on
        num_samples: Number of graphs to analyze
        
    Returns:
        List of metrics for each analyzed graph
    """
    # Enable oversquashing tracking
    model.enable_oversquashing_tracking()
    
    # Collect metrics
    metrics = []
    
    # Process a subset of graphs
    for idx, data in enumerate(tqdm(data_loader, desc="Analyzing graphs")):
        if idx >= num_samples:
            break
            
        try:
            data = data.to(device)
            
            # Forward pass to collect edge weights
            with torch.no_grad():
                model(data)
                weights_info = model.get_final_edge_weights(0)
                
                if weights_info:
                    # Get learned edge weights
                    learned_edge_weights = weights_info.get('edge_weights', None)
                    
                    # Get the original graph structure
                    edge_index = data.edge_index
                    
                    # Get the node features
                    node_features = data.x
                    
                    # Compute comprehensive metrics
                    metrics_dict = compute_all_oversquashing_metrics(
                        edge_index=edge_index,
                        edge_weights=learned_edge_weights,
                        node_features=node_features,
                        num_nodes=data.num_nodes
                    )
                    
                    # Store metrics with graph info
                    metrics.append({
                        'graph_idx': idx,
                        'num_nodes': data.num_nodes,
                        'num_edges': data.edge_index.shape[1],
                        'metrics': metrics_dict
                    })
        except Exception as e:
            print(f"Error analyzing graph {idx}: {e}")
    
    # Disable tracking to save memory
    model.disable_oversquashing_tracking()
    
    return metrics

def run_analysis(dataset_name, k=3, hidden_dim=16, batch_size=4, num_epochs=15, 
                num_samples=5, seed=42, output_dir=None):
    """
    Run full analysis with both balanced Cayley and uniform initialization.
    
    Args:
        dataset_name: Name of the dataset (MUTAG, PROTEINS, ENZYMES)
        k: Number of top-k connections
        hidden_dim: Hidden dimension size
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        num_samples: Number of test samples to analyze
        seed: Random seed
        output_dir: Directory to save results
    
    Returns:
        Dict with results and oversquashing metrics
    """
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = f"oversquashing_analysis/metrics_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    dataset = TUDataset(root='/workspaces/IPR-MPNN/datasets', name=dataset_name)
    
    # Get dataset stats
    print(f"Dataset: {dataset_name}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Split into train and test (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
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
    
    # Optimizers
    balanced_optimizer = torch.optim.Adam(balanced_model.parameters(), lr=0.005)
    uniform_optimizer = torch.optim.Adam(uniform_model.parameters(), lr=0.005)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training function
    def train(model, optimizer, data_loader):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data in data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            try:
                # Forward pass
                out = model(data)
                loss = criterion(out, data.y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Calculate accuracy
                pred = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                
                total_loss += loss.item() * data.num_graphs
            except Exception as e:
                print(f"Error during training: {e}")
            
        return total_loss / len(data_loader.dataset), correct / total
    
    # Evaluation function
    def evaluate(model, data_loader):
        model.eval()
        correct = 0
        total = 0
        
        for data in data_loader:
            data = data.to(device)
            with torch.no_grad():
                out = model(data)
                pred = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                
        return correct / total
    
    # Training loop with early stopping
    print(f"Training for up to {num_epochs} epochs...")
    
    # Initialize trackers
    balanced_best_acc = 0
    uniform_best_acc = 0
    balanced_patience = 0
    uniform_patience = 0
    patience_limit = 5
    
    balanced_train_losses = []
    balanced_train_accs = []
    balanced_test_accs = []
    uniform_train_losses = []
    uniform_train_accs = []
    uniform_test_accs = []
    
    for epoch in range(num_epochs):
        # Train balanced model
        balanced_loss, balanced_train_acc = train(balanced_model, balanced_optimizer, train_loader)
        balanced_train_losses.append(balanced_loss)
        balanced_train_accs.append(balanced_train_acc)
        
        # Train uniform model
        uniform_loss, uniform_train_acc = train(uniform_model, uniform_optimizer, train_loader)
        uniform_train_losses.append(uniform_loss)
        uniform_train_accs.append(uniform_train_acc)
        
        # Evaluate
        balanced_test_acc = evaluate(balanced_model, test_loader)
        balanced_test_accs.append(balanced_test_acc)
        
        uniform_test_acc = evaluate(uniform_model, test_loader)
        uniform_test_accs.append(uniform_test_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Balanced=[Loss: {balanced_loss:.4f}, Acc: {balanced_test_acc:.4f}], "
              f"Uniform=[Loss: {uniform_loss:.4f}, Acc: {uniform_test_acc:.4f}]")
        
        # Check for early stopping (balanced model)
        if balanced_test_acc > balanced_best_acc:
            balanced_best_acc = balanced_test_acc
            balanced_patience = 0
        else:
            balanced_patience += 1
            
        # Check for early stopping (uniform model)
        if uniform_test_acc > uniform_best_acc:
            uniform_best_acc = uniform_test_acc
            uniform_patience = 0
        else:
            uniform_patience += 1
            
        # Stop if both models have converged
        if balanced_patience >= patience_limit and uniform_patience >= patience_limit:
            print(f"Early stopping at epoch {epoch+1} - both models converged")
            break
    
    # Create analysis loaders with batch size 1
    analysis_loader = DataLoader(test_dataset[:num_samples], batch_size=1, shuffle=False)
    
    # Collect oversquashing metrics
    print("\nAnalyzing oversquashing metrics for balanced Cayley model...")
    balanced_metrics = analyze_model_oversquashing(balanced_model, analysis_loader, device, num_samples)
    
    print("\nAnalyzing oversquashing metrics for uniform model...")
    uniform_metrics = analyze_model_oversquashing(uniform_model, analysis_loader, device, num_samples)
    
    # Final accuracies
    balanced_final_acc = evaluate(balanced_model, test_loader)
    uniform_final_acc = evaluate(uniform_model, test_loader)
    
    print(f"Final Test Accuracy: Balanced={balanced_final_acc:.4f}, Uniform={uniform_final_acc:.4f}")
    
    # Compile results
    results = {
        'dataset': dataset_name,
        'parameters': {
            'top_k': k,
            'hidden_dim': hidden_dim,
            'num_epochs': num_epochs,
            'seed': seed
        },
        'accuracy': {
            'balanced_cayley': balanced_final_acc,
            'uniform': uniform_final_acc
        },
        'oversquashing_metrics': {
            'balanced_cayley': balanced_metrics,
            'uniform': uniform_metrics
        },
        'learning_curves': {
            'balanced_cayley': {
                'train_loss': balanced_train_losses,
                'train_acc': balanced_train_accs,
                'test_acc': balanced_test_accs
            },
            'uniform': {
                'train_loss': uniform_train_losses,
                'train_acc': uniform_train_accs,
                'test_acc': uniform_test_accs
            }
        }
    }
    
    # Save results to file
    results_file = os.path.join(output_dir, f"{dataset_name.lower()}_oversquashing_metrics.json")
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
    
    # Visualize results
    visualize_metrics_comparison(balanced_metrics, uniform_metrics, dataset_name, output_dir)
    
    # Clean up memory
    del balanced_model, uniform_model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

def visualize_metrics_comparison(balanced_metrics, uniform_metrics, dataset_name, output_dir):
    """
    Create visualizations comparing the oversquashing metrics between
    balanced Cayley and uniform initialization approaches.
    
    Args:
        balanced_metrics: Metrics from balanced Cayley model
        uniform_metrics: Metrics from uniform model
        dataset_name: Name of the dataset
        output_dir: Directory to save visualizations
    """
    if not balanced_metrics or not uniform_metrics:
        print("No metrics to visualize")
        return
    
    # Create DataFrames for easier handling
    balanced_df = pd.DataFrame([
        {
            'graph_idx': m['graph_idx'],
            'num_nodes': m['num_nodes'],
            'mean_effective_resistance': m['metrics']['mean_effective_resistance'],
            'cheeger_constant': m['metrics']['cheeger_constant'],
            'dirichlet_energy': m['metrics']['dirichlet_energy'],
            'conductance': m['metrics']['conductance'],
            'spectral_gap': m['metrics']['spectral_gap'],
            'algebraic_connectivity': m['metrics']['algebraic_connectivity'],
            'model': 'Balanced Cayley'
        }
        for m in balanced_metrics
    ])
    
    uniform_df = pd.DataFrame([
        {
            'graph_idx': m['graph_idx'],
            'num_nodes': m['num_nodes'],
            'mean_effective_resistance': m['metrics']['mean_effective_resistance'],
            'cheeger_constant': m['metrics']['cheeger_constant'],
            'dirichlet_energy': m['metrics']['dirichlet_energy'],
            'conductance': m['metrics']['conductance'],
            'spectral_gap': m['metrics']['spectral_gap'],
            'algebraic_connectivity': m['metrics']['algebraic_connectivity'],
            'model': 'Uniform'
        }
        for m in uniform_metrics
    ])
    
    # Combine dataframes
    combined_df = pd.concat([balanced_df, uniform_df])
    
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # Create grid of visualizations for different metrics
    metrics_to_plot = [
        ('mean_effective_resistance', 'Mean Effective Resistance', 'Lower is better'),
        ('cheeger_constant', 'Cheeger Constant', 'Higher is better'),
        ('dirichlet_energy', 'Dirichlet Energy', 'Lower is better'),
        ('conductance', 'Graph Conductance', 'Higher is better'),
        ('spectral_gap', 'Spectral Gap', 'Higher is better'),
        ('algebraic_connectivity', 'Algebraic Connectivity', 'Higher is better')
    ]
    
    # Plot each metric
    for i, (metric_name, metric_title, metric_note) in enumerate(metrics_to_plot):
        plt.figure(figsize=(12, 6))
        
        # Create barplot for each graph
        ax = sns.barplot(
            data=combined_df, 
            x='graph_idx', 
            y=metric_name, 
            hue='model',
            palette=['blue', 'orange']
        )
        
        # Add title and labels
        plt.title(f'{metric_title} Comparison ({dataset_name})\n{metric_note}', fontsize=14)
        plt.xlabel('Graph Index', fontsize=12)
        plt.ylabel(metric_title, fontsize=12)
        plt.legend(title='Model')
        
        # Add values on top of bars
        for p in ax.patches:
            ax.annotate(
                f'{p.get_height():.3f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', 
                xytext=(0, 5), 
                textcoords='offset points'
            )
        
        # Save the plot
        plt.tight_layout()
        metric_file = os.path.join(output_dir, f"{dataset_name.lower()}_{metric_name}_comparison.png")
        plt.savefig(metric_file)
        plt.close()
    
    # Create a summary plot with all metrics
    plt.figure(figsize=(15, 10))
    
    # Calculate mean metrics
    metric_means = {}
    for model_df, model_name in [(balanced_df, 'Balanced Cayley'), (uniform_df, 'Uniform')]:
        metric_means[model_name] = {}
        for metric_name, _, _ in metrics_to_plot:
            metric_means[model_name][metric_name] = model_df[metric_name].mean()
    
    # Create summary dataframe for plotting
    summary_data = []
    for metric_name, metric_title, _ in metrics_to_plot:
        summary_data.append({
            'metric': metric_title,
            'Balanced Cayley': metric_means['Balanced Cayley'][metric_name],
            'Uniform': metric_means['Uniform'][metric_name]
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Reshape for seaborn
    summary_df_melted = pd.melt(
        summary_df, 
        id_vars=['metric'], 
        value_vars=['Balanced Cayley', 'Uniform'],
        var_name='model', 
        value_name='value'
    )
    
    # Create summary barplot
    plt.subplot(2, 1, 1)
    ax = sns.barplot(
        data=summary_df_melted,
        x='metric',
        y='value',
        hue='model',
        palette=['blue', 'orange']
    )
    
    # Add title and labels
    plt.title(f'Oversquashing Metrics Summary ({dataset_name})', fontsize=16)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    
    # Calculate relative difference (normalized)
    plt.subplot(2, 1, 2)
    relative_diff = []
    
    for metric_name, metric_title, better in metrics_to_plot:
        balanced_value = metric_means['Balanced Cayley'][metric_name]
        uniform_value = metric_means['Uniform'][metric_name]
        
        # Skip if values are invalid
        if balanced_value == 0 or uniform_value == 0 or np.isinf(balanced_value) or np.isinf(uniform_value):
            continue
            
        # Calculate relative difference
        if better == 'Higher is better':
            # Positive means balanced is better, negative means uniform is better
            diff = (balanced_value - uniform_value) / max(abs(uniform_value), 1e-10)
        else:  # 'Lower is better'
            # Negative means balanced is better, positive means uniform is better
            diff = (uniform_value - balanced_value) / max(abs(uniform_value), 1e-10)
        
        relative_diff.append({
            'metric': metric_title,
            'relative_diff': diff
        })
    
    relative_df = pd.DataFrame(relative_diff)
    
    # Create barplot of relative differences
    colors = ['green' if x >= 0 else 'red' for x in relative_df['relative_diff']]
    ax = sns.barplot(
        data=relative_df,
        x='metric',
        y='relative_diff',
        palette=colors
    )
    
    # Add horizontal line at 0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add title and labels
    plt.title('Relative Improvement of Balanced Cayley over Uniform\n(Positive = Balanced is better)', fontsize=14)
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Relative Difference', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():.2f}', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', va='bottom' if p.get_height() >= 0 else 'top', 
            xytext=(0, 5 if p.get_height() >= 0 else -5), 
            textcoords='offset points'
        )
    
    # Save the summary plot
    plt.tight_layout()
    summary_file = os.path.join(output_dir, f"{dataset_name.lower()}_metrics_summary.png")
    plt.savefig(summary_file)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def analyze_all_datasets(k=3, hidden_dim=16, num_epochs=15, num_samples=5):
    """Run analysis on all three datasets"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"oversquashing_analysis/metrics_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create JSON to collect all results
    all_results = []
    
    # Run MUTAG experiment
    print("\n=== Running MUTAG Experiment ===")
    mutag_results = run_analysis(
        dataset_name="MUTAG",
        k=k,
        hidden_dim=hidden_dim,
        batch_size=4,
        num_epochs=num_epochs,
        num_samples=num_samples,
        seed=42,
        output_dir=output_dir
    )
    all_results.append(mutag_results)
    
    # Run PROTEINS experiment
    print("\n=== Running PROTEINS Experiment ===")
    proteins_results = run_analysis(
        dataset_name="PROTEINS",
        k=k,
        hidden_dim=hidden_dim,
        batch_size=4,
        num_epochs=num_epochs,
        num_samples=num_samples,
        seed=42,
        output_dir=output_dir
    )
    all_results.append(proteins_results)
    
    # Run ENZYMES experiment
    print("\n=== Running ENZYMES Experiment ===")
    enzymes_results = run_analysis(
        dataset_name="ENZYMES",
        k=k,
        hidden_dim=hidden_dim,
        batch_size=4,
        num_epochs=num_epochs,
        num_samples=num_samples,
        seed=42,
        output_dir=output_dir
    )
    all_results.append(enzymes_results)
    
    # Save combined results
    with open(os.path.join(output_dir, "all_dataset_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create cross-dataset comparison visualization
    create_cross_dataset_visualization(
        [mutag_results, proteins_results, enzymes_results],
        output_dir
    )
    
    print(f"\nAll analyses complete. Results saved to {output_dir}")

def create_cross_dataset_visualization(results_list, output_dir):
    """
    Create visualizations comparing metrics across different datasets.
    
    Args:
        results_list: List of results for different datasets
        output_dir: Directory to save visualizations
    """
    datasets = [r['dataset'] for r in results_list]
    
    # Extract key metrics for each dataset and model
    metric_data = []
    
    for result in results_list:
        dataset = result['dataset']
        
        # Extract metrics
        for model_type in ['balanced_cayley', 'uniform']:
            metrics_list = result['oversquashing_metrics'][model_type]
            
            if not metrics_list:
                continue
                
            # Calculate mean for each metric across graphs
            mean_metrics = {}
            for metric_name in ['mean_effective_resistance', 'cheeger_constant', 
                               'dirichlet_energy', 'conductance', 'spectral_gap', 
                               'algebraic_connectivity']:
                values = []
                for m in metrics_list:
                    if metric_name in m['metrics']:
                        val = m['metrics'][metric_name]
                        if not np.isinf(val) and not np.isnan(val):
                            values.append(val)
                
                if values:
                    mean_metrics[metric_name] = np.mean(values)
                else:
                    mean_metrics[metric_name] = 0
            
            # Add row for this dataset and model
            metric_data.append({
                'dataset': dataset,
                'model': 'Balanced Cayley' if model_type == 'balanced_cayley' else 'Uniform',
                'accuracy': result['accuracy'][model_type],
                **mean_metrics
            })
    
    # Create DataFrame
    df = pd.DataFrame(metric_data)
    
    # Create visualization
    plt.figure(figsize=(15, 20))
    
    # Metrics to plot
    metrics_to_plot = [
        ('mean_effective_resistance', 'Mean Effective Resistance'),
        ('cheeger_constant', 'Cheeger Constant'),
        ('dirichlet_energy', 'Dirichlet Energy'),
        ('conductance', 'Graph Conductance'),
        ('spectral_gap', 'Spectral Gap'),
        ('algebraic_connectivity', 'Algebraic Connectivity'),
        ('accuracy', 'Model Accuracy')
    ]
    
    # Create subplot for each metric
    for i, (metric_name, metric_title) in enumerate(metrics_to_plot):
        plt.subplot(len(metrics_to_plot), 1, i+1)
        
        # Create grouped bar chart
        sns.barplot(
            data=df,
            x='dataset',
            y=metric_name,
            hue='model',
            palette=['blue', 'orange']
        )
        
        # Add title and labels
        plt.title(f'{metric_title} Across Datasets', fontsize=14)
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel(metric_title, fontsize=12)
        plt.legend(title='Model')
    
    # Save visualization
    plt.tight_layout()
    vis_file = os.path.join(output_dir, "cross_dataset_comparison.png")
    plt.savefig(vis_file)
    plt.close()
    
    print(f"Cross-dataset visualization saved to {vis_file}")
    
    # Create markdown summary
    md_file = os.path.join(output_dir, "OVERSQUASHING_ANALYSIS_SUMMARY.md")
    with open(md_file, 'w') as f:
        f.write("# Comprehensive Oversquashing Analysis Summary\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Add explanation of metrics
        f.write("## Metrics Explanation\n\n")
        f.write("| Metric | Description | Interpretation |\n")
        f.write("|--------|-------------|---------------|\n")
        f.write("| **Effective Resistance** | Measures resistance between node pairs | Lower values indicate less oversquashing |\n")
        f.write("| **Cheeger Constant** | Measures bottlenecks in the graph | Higher values indicate fewer bottlenecks |\n")
        f.write("| **Dirichlet Energy** | Measures smoothness of message passing | Lower values indicate smoother message propagation |\n")
        f.write("| **Conductance** | Measures how well connected the graph is | Higher values indicate better connectivity |\n")
        f.write("| **Spectral Gap** | Difference between first and second eigenvalues | Higher values indicate faster mixing time |\n")
        f.write("| **Algebraic Connectivity** | Second smallest eigenvalue of Laplacian | Higher values indicate more robust connectivity |\n\n")
        
        # Add results tables for each dataset
        for dataset in datasets:
            f.write(f"\n## {dataset}\n\n")
            
            # Filter for this dataset
            dataset_df = df[df['dataset'] == dataset]
            
            # Create table for this dataset
            f.write("| Metric | Balanced Cayley | Uniform | Relative Difference |\n")
            f.write("|--------|----------------|---------|--------------------|\n")
            
            for metric_name, metric_title in metrics_to_plot:
                balanced_row = dataset_df[dataset_df['model'] == 'Balanced Cayley']
                uniform_row = dataset_df[dataset_df['model'] == 'Uniform']
                
                if len(balanced_row) > 0 and len(uniform_row) > 0:
                    balanced_val = balanced_row[metric_name].values[0]
                    uniform_val = uniform_row[metric_name].values[0]
                    
                    # Calculate relative difference
                    if uniform_val != 0:
                        rel_diff = (balanced_val - uniform_val) / abs(uniform_val)
                        rel_diff_str = f"{rel_diff:.2f}"
                    else:
                        rel_diff_str = "N/A"
                    
                    f.write(f"| {metric_title} | {balanced_val:.4f} | {uniform_val:.4f} | {rel_diff_str} |\n")
        
        # Add overall insights
        f.write("\n## Key Findings\n\n")
        
        # Calculate overall metrics
        balanced_overall = df[df['model'] == 'Balanced Cayley']
        uniform_overall = df[df['model'] == 'Uniform']
        
        # Compare each metric to determine which approach is better overall
        for metric_name, metric_title in metrics_to_plot:
            balanced_mean = balanced_overall[metric_name].mean()
            uniform_mean = uniform_overall[metric_name].mean()
            
            # Determine which is better based on the metric
            if metric_name in ['mean_effective_resistance', 'dirichlet_energy']:
                # Lower is better
                better = "Balanced Cayley" if balanced_mean < uniform_mean else "Uniform"
                direction = "lower"
            else:
                # Higher is better
                better = "Balanced Cayley" if balanced_mean > uniform_mean else "Uniform"
                direction = "higher"
            
            f.write(f"- For **{metric_title}**, the **{better}** approach shows {direction} values overall\n")
        
        # Add conclusion
        f.write("\n## Conclusion\n\n")
        
        # Determine overall winner based on accuracy
        balanced_acc = balanced_overall['accuracy'].mean()
        uniform_acc = uniform_overall['accuracy'].mean()
        winner = "Balanced Cayley" if balanced_acc > uniform_acc else "Uniform"
        
        f.write(f"The **{winner}** initialization demonstrates better overall performance ")
        f.write(f"with a mean accuracy of {max(balanced_acc, uniform_acc):.4f} compared to ")
        f.write(f"{min(balanced_acc, uniform_acc):.4f}.\n\n")
        
        # Count metrics where each approach is better
        balanced_wins = sum(1 for m, _ in metrics_to_plot if 
                           (m in ['mean_effective_resistance', 'dirichlet_energy'] and 
                            balanced_overall[m].mean() < uniform_overall[m].mean()) or
                           (m not in ['mean_effective_resistance', 'dirichlet_energy'] and 
                            balanced_overall[m].mean() > uniform_overall[m].mean()))
        
        uniform_wins = len(metrics_to_plot) - balanced_wins
        
        f.write(f"The Balanced Cayley approach performs better in {balanced_wins} metrics, ")
        f.write(f"while the Uniform approach performs better in {uniform_wins} metrics.\n\n")
        
        f.write("This suggests that ")
        if balanced_wins > uniform_wins:
            f.write("the **Balanced Cayley** initialization is more effective at mitigating oversquashing ")
            f.write("and improving message passing efficiency in graph neural networks.")
        elif uniform_wins > balanced_wins:
            f.write("the **Uniform** initialization is more effective at mitigating oversquashing ")
            f.write("and improving message passing efficiency in graph neural networks.")
        else:
            f.write("both approaches have comparable effects on mitigating oversquashing, ")
            f.write("with different strengths in different aspects of graph connectivity.")
    
    print(f"Markdown summary saved to {md_file}")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run oversquashing metrics analysis')
    parser.add_argument('--dataset', type=str, default='MUTAG', 
                       help='Dataset name (MUTAG, PROTEINS, ENZYMES)')
    parser.add_argument('--k', type=int, default=3, 
                       help='Number of top-k connections')
    parser.add_argument('--hidden', type=int, default=16, 
                       help='Hidden dimension size')
    parser.add_argument('--epochs', type=int, default=15, 
                       help='Number of training epochs')
    parser.add_argument('--samples', type=int, default=5, 
                       help='Number of test samples to analyze')
    parser.add_argument('--all', action='store_true', 
                       help='Run analysis on all three datasets')
    
    args = parser.parse_args()
    
    # Set up error detection for debugging
    torch.autograd.set_detect_anomaly(True)
    
    if args.all:
        # Run on all datasets
        analyze_all_datasets(
            k=args.k, 
            hidden_dim=args.hidden, 
            num_epochs=args.epochs,
            num_samples=args.samples
        )
    else:
        # Run on single dataset
        run_analysis(
            dataset_name=args.dataset,
            k=args.k,
            hidden_dim=args.hidden,
            num_epochs=args.epochs,
            num_samples=args.samples
        )
