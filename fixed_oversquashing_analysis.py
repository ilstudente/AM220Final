"""
Fixed oversquashing metrics analysis script for comparing balanced Cayley vs. uniform initialization.
This version properly uses the learned edge weights from each model to calculate metrics.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import networkx as nx
from tqdm import tqdm
import gc
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from scipy import sparse
from scipy.sparse.linalg import eigsh

from models.memory_saver_iprmpnn import MemorySaverIPRMPNNModel
from utils.oversquashing_metrics import compute_oversquashing_metric, compute_graph_connectivity_metrics

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def compute_cheeger_constant(adj_matrix):
    """
    Compute the Cheeger constant (conductance) of a graph.
    The Cheeger constant measures how well-connected a graph is.
    Lower values indicate potential bottlenecks for message passing.
    
    Args:
        adj_matrix: Adjacency matrix of the graph
        
    Returns:
        Cheeger constant value
    """
    # Convert to NetworkX graph
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()
        
    G = nx.from_numpy_array(adj_matrix)
    
    # If graph is not connected, return 0
    if not nx.is_connected(G):
        return 0.0
    
    n = G.number_of_nodes()
    
    # For small graphs, calculate exactly
    if n <= 15:
        min_conductance = 1.0
        for i in range(1, 2**(n-1)):
            # Create a binary representation of i
            binary = format(i, f'0{n}b')[-n:]
            set_s = {j for j in range(n) if binary[j] == '1'}
            
            # Skip if set_s is empty or contains all nodes
            if not set_s or len(set_s) == n:
                continue
                
            set_sc = set(range(n)) - set_s
            
            # Calculate cut size
            cut_size = sum(1 for u in set_s for v in set_sc if G.has_edge(u, v))
            
            # Calculate conductance
            vol_s = sum(dict(G.degree(set_s)).values())
            vol_sc = sum(dict(G.degree(set_sc)).values())
            conductance = cut_size / min(vol_s, vol_sc) if min(vol_s, vol_sc) > 0 else 1.0
            
            min_conductance = min(min_conductance, conductance)
        
        return float(min_conductance)
    else:
        # For larger graphs, use approximation via spectral gap
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

def compute_dirichlet_energy(adj_matrix, node_features):
    """
    Compute the Dirichlet energy which measures the smoothness of features across the graph.
    Higher Dirichlet energy indicates more feature variation across edges.
    
    Args:
        adj_matrix: Adjacency matrix of the graph
        node_features: Node feature matrix
        
    Returns:
        Dirichlet energy value
    """
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()
    
    if isinstance(node_features, torch.Tensor):
        node_features = node_features.cpu().numpy()
    
    # Ensure node_features is 2D
    if node_features.ndim == 1:
        node_features = node_features.reshape(-1, 1)
    
    # Create graph Laplacian
    n = adj_matrix.shape[0]
    degrees = np.sum(adj_matrix, axis=1)
    D = np.diag(degrees)
    L = D - adj_matrix
    
    # Calculate Dirichlet energy: f^T L f
    energy = 0
    for f in range(node_features.shape[1]):
        feature = node_features[:, f]
        energy += np.dot(feature, np.dot(L, feature))
    
    return float(energy)

def create_weighted_adjacency_from_model(model_weights, original_edge_index, num_nodes):
    """
    Create a properly weighted adjacency matrix from the model's learned weights.
    
    Args:
        model_weights: Edge weights from the model
        original_edge_index: Original edge index from the dataset
        num_nodes: Number of nodes in the graph
        
    Returns:
        Weighted adjacency matrix
    """
    # Start with the original graph structure
    device = original_edge_index.device
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    adj[original_edge_index[0], original_edge_index[1]] = 1
    
    # Get the matrix of edge weights from the model
    edge_weight_matrix = model_weights['edge_weights']  # Should be shape [num_nodes, num_virtual_nodes]
    
    # Create a virtual node enhanced adjacency matrix
    # This combines the original graph with the virtual node connections
    virtual_connections = torch.zeros(num_nodes, num_nodes, device=device)
    
    # Calculate virtual node connections: If two nodes are strongly connected to 
    # the same virtual nodes, they have a strong indirect connection
    virtual_connections = torch.mm(edge_weight_matrix, edge_weight_matrix.t())
    
    # Normalize the virtual connections
    max_val = virtual_connections.max()
    if max_val > 0:
        virtual_connections = virtual_connections / max_val
    
    # Combine original structure with virtual node-based connections
    combined_adj = adj + virtual_connections * 0.5  # Give virtual connections slightly less weight
    
    return combined_adj

def compute_metrics_for_model(edge_index, model_weights, node_features=None, num_nodes=None):
    """
    Compute oversquashing metrics for a specific model using its learned edge weights.
    
    Args:
        edge_index: Original edge index from the dataset
        model_weights: Edge weights from the model
        node_features: Node features for Dirichlet energy calculation
        num_nodes: Number of nodes in the graph
        
    Returns:
        Dict with all metrics
    """
    # If num_nodes not provided, infer from edge_index
    if num_nodes is None:
        num_nodes = max(edge_index[0].max().item(), edge_index[1].max().item()) + 1
    
    # Create weighted adjacency matrix
    weighted_adj = create_weighted_adjacency_from_model(model_weights, edge_index, num_nodes)
    
    # Compute standard oversquashing metrics using original edge_index but with model's virtual node influence
    standard_metrics = compute_oversquashing_metric(weighted_adj)
    connectivity_metrics = compute_graph_connectivity_metrics(weighted_adj)
    
    # Calculate additional metrics
    cheeger_constant = compute_cheeger_constant(weighted_adj)
    
    # Calculate Dirichlet energy if node features are provided
    dirichlet_energy = None
    if node_features is not None:
        dirichlet_energy = compute_dirichlet_energy(weighted_adj, node_features)
    
    # Combine all metrics
    metrics = {
        **standard_metrics,
        **connectivity_metrics,
        "cheeger_constant": cheeger_constant,
        "dirichlet_energy": dirichlet_energy
    }
    
    return metrics

def run_analysis(dataset_name, k=3, hidden_dim=16, batch_size=4, num_epochs=10, seed=42):
    """
    Run experiment with both balanced Cayley and uniform initialization,
    and analyze oversquashing metrics for the final learned graphs.
    
    Args:
        dataset_name: Name of the dataset (MUTAG, PROTEINS, ENZYMES)
        k: Number of top-k connections
        hidden_dim: Hidden dimension size
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        seed: Random seed
    
    Returns:
        Dict with analysis results
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"oversquashing_analysis/fixed_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    try:
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
                    
                    # Backward pass with gradient clipping
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
        
        # Initialize tracking
        balanced_train_losses = []
        balanced_train_accs = []
        balanced_test_accs = []
        uniform_train_losses = []
        uniform_train_accs = []
        uniform_test_accs = []
        
        # Training loop
        print(f"Training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            # Train balanced model
            balanced_loss, balanced_train_acc = train(balanced_model, balanced_optimizer, train_loader)
            balanced_train_losses.append(balanced_loss)
            balanced_train_accs.append(balanced_train_acc)
            
            # Train uniform model
            uniform_loss, uniform_train_acc = train(uniform_model, uniform_optimizer, train_loader)
            uniform_train_losses.append(uniform_loss)
            uniform_train_accs.append(uniform_train_acc)
            
            # Evaluate every few epochs or at the end
            if (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
                balanced_test_acc = evaluate(balanced_model, test_loader)
                balanced_test_accs.append(balanced_test_acc)
                
                uniform_test_acc = evaluate(uniform_model, test_loader)
                uniform_test_accs.append(uniform_test_acc)
                
                print(f"Epoch {epoch+1}/{num_epochs}: Balanced=[Loss: {balanced_loss:.4f}, Acc: {balanced_test_acc:.4f}], "
                      f"Uniform=[Loss: {uniform_loss:.4f}, Acc: {uniform_test_acc:.4f}]")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}: Balanced Loss: {balanced_loss:.4f}, Uniform Loss: {uniform_loss:.4f}")
        
        # Final evaluation and oversquashing analysis
        print("Final evaluation and oversquashing analysis...")
        
        # Enable oversquashing tracking for final analysis
        balanced_model.enable_oversquashing_tracking()
        uniform_model.enable_oversquashing_tracking()
        
        # Analyze oversquashing on a subset of test graphs
        num_samples = min(5, len(test_dataset))
        print(f"Analyzing {num_samples} graphs for oversquashing metrics")
        
        # Create analysis data loaders with batch size 1
        oversquashing_dataloader = DataLoader(
            test_dataset[:num_samples], 
            batch_size=1, 
            shuffle=False
        )
        
        # Collect metrics
        balanced_metrics = []
        uniform_metrics = []
        
        for idx, data in enumerate(tqdm(oversquashing_dataloader, desc="Analyzing graphs")):
            try:
                data = data.to(device)
                
                # Forward pass to collect edge weights
                with torch.no_grad():
                    # Process with balanced model
                    balanced_model(data)
                    balanced_weights = balanced_model.get_final_edge_weights(0)
                    
                    # Process with uniform model
                    uniform_model(data)
                    uniform_weights = uniform_model.get_final_edge_weights(0)
                    
                    if balanced_weights and uniform_weights:
                        try:
                            # Calculate metrics with CORRECT handling of edge weights
                            print(f"Computing metrics for graph {idx}...")
                            
                            # Balanced model metrics
                            balanced_computed_metrics = compute_metrics_for_model(
                                data.edge_index,
                                balanced_weights,
                                data.x,
                                balanced_weights['num_nodes']
                            )
                            
                            # Uniform model metrics
                            uniform_computed_metrics = compute_metrics_for_model(
                                data.edge_index,
                                uniform_weights,
                                data.x,
                                uniform_weights['num_nodes']
                            )
                            
                            # Store metrics
                            balanced_metrics.append({
                                'graph_idx': idx,
                                'num_nodes': balanced_weights['num_nodes'],
                                'num_virtual_nodes': balanced_weights['num_virtual_nodes'],
                                'metrics': balanced_computed_metrics
                            })
                            
                            uniform_metrics.append({
                                'graph_idx': idx,
                                'num_nodes': uniform_weights['num_nodes'],
                                'num_virtual_nodes': uniform_weights['num_virtual_nodes'],
                                'metrics': uniform_computed_metrics
                            })
                            
                            # Print a sample of the differences to verify they're different
                            if idx == 0:
                                print("\nMetrics comparison for first graph:")
                                print(f"Balanced Cheeger constant: {balanced_computed_metrics['cheeger_constant']:.4f}")
                                print(f"Uniform Cheeger constant: {uniform_computed_metrics['cheeger_constant']:.4f}")
                                print(f"Balanced mean effective resistance: {balanced_computed_metrics['mean_effective_resistance']:.4f}")
                                print(f"Uniform mean effective resistance: {uniform_computed_metrics['mean_effective_resistance']:.4f}")
                                print(f"Balanced avg path length: {balanced_computed_metrics['avg_path_length']:.4f}")
                                print(f"Uniform avg path length: {uniform_computed_metrics['avg_path_length']:.4f}")
                            
                            print(f"Graph {idx}: Metrics collected successfully")
                        except Exception as e:
                            print(f"Error calculating metrics for graph {idx}: {e}")
            except Exception as e:
                print(f"Error processing graph {idx}: {e}")
        
        # Disable tracking to save memory
        balanced_model.disable_oversquashing_tracking()
        uniform_model.disable_oversquashing_tracking()
        
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
        
        # Save results
        results_file = os.path.join(output_dir, f"{dataset_name.lower()}_fixed_analysis.json")
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
        
        # Visualize metrics
        visualize_metrics(balanced_metrics, uniform_metrics, dataset_name, output_dir)
        
        # Clean up memory
        del balanced_model, uniform_model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
    
    except Exception as e:
        print(f"Error in analysis: {e}")
        return None

def visualize_metrics(balanced_metrics, uniform_metrics, dataset_name, output_dir):
    """
    Visualize the oversquashing metrics comparison.
    
    Args:
        balanced_metrics: List of metrics dicts for balanced Cayley approach
        uniform_metrics: List of metrics dicts for uniform approach
        dataset_name: Name of the dataset
        output_dir: Directory to save visualizations
    """
    if not balanced_metrics or not uniform_metrics:
        print("No metrics to visualize")
        return
    
    # Extract key metrics
    balanced_resistance = [m['metrics']['mean_effective_resistance'] for m in balanced_metrics 
                          if 'metrics' in m and 'mean_effective_resistance' in m['metrics']]
    
    uniform_resistance = [m['metrics']['mean_effective_resistance'] for m in uniform_metrics
                         if 'metrics' in m and 'mean_effective_resistance' in m['metrics']]
    
    balanced_path_length = [m['metrics']['avg_path_length'] for m in balanced_metrics
                           if 'metrics' in m and 'avg_path_length' in m['metrics']
                           and m['metrics']['avg_path_length'] != float('inf')]
    
    uniform_path_length = [m['metrics']['avg_path_length'] for m in uniform_metrics
                          if 'metrics' in m and 'avg_path_length' in m['metrics']
                          and m['metrics']['avg_path_length'] != float('inf')]
    
    balanced_spectral_gap = [m['metrics']['spectral_gap'] for m in balanced_metrics
                            if 'metrics' in m and 'spectral_gap' in m['metrics']]
    
    uniform_spectral_gap = [m['metrics']['spectral_gap'] for m in uniform_metrics
                           if 'metrics' in m and 'spectral_gap' in m['metrics']]
    
    balanced_cheeger = [m['metrics']['cheeger_constant'] for m in balanced_metrics
                       if 'metrics' in m and 'cheeger_constant' in m['metrics']]
    
    uniform_cheeger = [m['metrics']['cheeger_constant'] for m in uniform_metrics
                      if 'metrics' in m and 'cheeger_constant' in m['metrics']]
    
    # Create visualization
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Effective resistance
    plt.subplot(2, 2, 1)
    idx = range(len(balanced_resistance))
    plt.bar([i - 0.2 for i in idx], balanced_resistance, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in idx], uniform_resistance, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title(f'Effective Resistance Comparison ({dataset_name})')
    plt.xlabel('Graph Index')
    plt.ylabel('Mean Effective Resistance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Path length
    plt.subplot(2, 2, 2)
    if balanced_path_length and uniform_path_length:
        plt.bar([i - 0.2 for i in range(len(balanced_path_length))], balanced_path_length, 
                width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
        plt.bar([i + 0.2 for i in range(len(uniform_path_length))], uniform_path_length, 
                width=0.4, label='Uniform', color='orange', alpha=0.7)
        plt.title(f'Average Path Length Comparison ({dataset_name})')
        plt.xlabel('Graph Index')
        plt.ylabel('Average Path Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No valid path length data', 
                horizontalalignment='center', verticalalignment='center')
        plt.title(f'Average Path Length Comparison ({dataset_name})')
    
    # Plot 3: Spectral gap
    plt.subplot(2, 2, 3)
    plt.bar([i - 0.2 for i in idx], balanced_spectral_gap, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in idx], uniform_spectral_gap, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title(f'Spectral Gap Comparison ({dataset_name})')
    plt.xlabel('Graph Index')
    plt.ylabel('Spectral Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Cheeger constant
    plt.subplot(2, 2, 4)
    plt.bar([i - 0.2 for i in range(len(balanced_cheeger))], balanced_cheeger, 
            width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in range(len(uniform_cheeger))], uniform_cheeger, 
            width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title(f'Cheeger Constant Comparison ({dataset_name})')
    plt.xlabel('Graph Index')
    plt.ylabel('Cheeger Constant')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save visualization
    plt.tight_layout()
    vis_file = os.path.join(output_dir, f"{dataset_name.lower()}_metrics_comparison.png")
    plt.savefig(vis_file)
    plt.close()
    
    print(f"Visualization saved to {vis_file}")
    
    # Also create a summary visualization with means
    plt.figure(figsize=(12, 6))
    
    metrics = [
        ('Mean Effective Resistance', np.mean(balanced_resistance) if balanced_resistance else 0, 
                                     np.mean(uniform_resistance) if uniform_resistance else 0),
        ('Average Path Length', np.mean(balanced_path_length) if balanced_path_length else 0, 
                               np.mean(uniform_path_length) if uniform_path_length else 0),
        ('Spectral Gap', np.mean(balanced_spectral_gap) if balanced_spectral_gap else 0, 
                        np.mean(uniform_spectral_gap) if uniform_spectral_gap else 0),
        ('Cheeger Constant', np.mean(balanced_cheeger) if balanced_cheeger else 0,
                           np.mean(uniform_cheeger) if uniform_cheeger else 0)
    ]
    
    labels = [m[0] for m in metrics]
    balanced_values = [m[1] for m in metrics]
    uniform_values = [m[2] for m in metrics]
    
    x = range(len(labels))
    
    plt.bar([i - 0.2 for i in x], balanced_values, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in x], uniform_values, width=0.4, label='Uniform', color='orange', alpha=0.7)
    
    plt.title(f'Oversquashing Metrics Summary ({dataset_name})')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.xticks(x, labels, rotation=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save summary visualization
    summary_file = os.path.join(output_dir, f"{dataset_name.lower()}_metrics_summary.png")
    plt.savefig(summary_file)
    plt.close()
    
    print(f"Summary visualization saved to {summary_file}")
    
    # Create a markdown summary
    summary_md = os.path.join(output_dir, f"{dataset_name.lower()}_metrics_summary.md")
    with open(summary_md, 'w') as f:
        f.write(f"# Oversquashing Metrics Analysis for {dataset_name}\n\n")
        
        f.write("## Summary of Metrics\n\n")
        f.write("| Metric | Balanced Cayley | Uniform | Difference |\n")
        f.write("|--------|----------------|---------|------------|\n")
        
        for label, bal_val, uni_val in metrics:
            diff = bal_val - uni_val
            better = "Balanced" if (label in ["Spectral Gap", "Cheeger Constant"] and diff > 0) or \
                                   (label in ["Mean Effective Resistance", "Average Path Length"] and diff < 0) \
                                else "Uniform"
            f.write(f"| {label} | {bal_val:.4f} | {uni_val:.4f} | {diff:.4f} ({better} better) |\n")
        
        f.write("\n## Interpretation\n\n")
        f.write("- **Lower effective resistance** indicates less oversquashing\n")
        f.write("- **Shorter average path length** indicates more efficient message passing\n")
        f.write("- **Larger spectral gap** indicates faster information mixing in the graph\n")
        f.write("- **Larger Cheeger constant** indicates better connectivity with fewer bottlenecks\n")
    
    print(f"Summary markdown saved to {summary_md}")

def run_all_datasets():
    """Run analysis on MUTAG, PROTEINS, and ENZYMES datasets"""
    for dataset in ["MUTAG", "PROTEINS", "ENZYMES"]:
        print(f"\n=== Running analysis for {dataset} ===\n")
        run_analysis(
            dataset_name=dataset,
            k=3,
            hidden_dim=16,
            batch_size=4,
            num_epochs=10,
            seed=42
        )

if __name__ == "__main__":
    # Start with just MUTAG which is smaller and faster to process
    run_analysis(
        dataset_name="MUTAG",
        k=3,
        hidden_dim=16,
        batch_size=4,
        num_epochs=3,  # Reduced epochs for faster testing
        seed=42
    )
    
    # Uncomment to run all datasets
    # run_all_datasets()
