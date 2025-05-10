"""
Run oversquashing analysis for balanced vs uniform initialization methods on multiple datasets.
This script compares the impact of different initialization methods on message passing efficiency
by analyzing metrics like Cheeger constant and Dirichlet energy.
"""

import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import networkx as nx
from tqdm import tqdm
import gc
import scipy.sparse as sp

from models.memory_saver_iprmpnn import MemorySaverIPRMPNNModel
from utils.oversquashing_metrics import compute_oversquashing_metric, compute_graph_connectivity_metrics

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_cheeger_constant(adjacency_matrix):
    """
    Compute the Cheeger constant (isoperimetric number) of a graph.
    Lower Cheeger constant indicates a bottleneck in the graph.
    
    Args:
        adjacency_matrix: Adjacency matrix (can be weighted)
        
    Returns:
        Cheeger constant value
    """
    try:
        # Convert to networkx graph
        if isinstance(adjacency_matrix, torch.Tensor):
            adjacency_matrix = adjacency_matrix.cpu().numpy()
            
        if isinstance(adjacency_matrix, np.ndarray):
            G = nx.from_numpy_array(adjacency_matrix)
        else:
            G = adjacency_matrix  # Assume it's already a networkx graph
            
        if not nx.is_connected(G):
            # For disconnected graphs, return a low value
            return 0.0
            
        # Calculate Cheeger constant
        n = G.number_of_nodes()
        min_expansion = float('inf')
        
        # Use spectral partitioning as an approximation
        laplacian = nx.normalized_laplacian_matrix(G).todense()
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        
        # Use the Fiedler vector (eigenvector corresponding to the second smallest eigenvalue)
        fiedler_vector = eigenvectors[:, 1]
        
        # Use median as the partition threshold
        median = np.median(fiedler_vector)
        
        # Create partitions
        S = set([i for i, val in enumerate(fiedler_vector) if val <= median])
        complement_S = set(range(n)) - S
        
        # Count edges between partitions
        cut_size = nx.cut_size(G, S, complement_S)
        
        # Return approximate Cheeger constant
        denominator = min(sum(G.degree(node) for node in S), 
                          sum(G.degree(node) for node in complement_S))
        
        if denominator > 0:
            return cut_size / denominator
        else:
            return 0.0
            
    except Exception as e:
        print(f"Error computing Cheeger constant: {e}")
        return 0.0

def compute_dirichlet_energy(adjacency_matrix, node_features):
    """
    Compute the Dirichlet energy of the graph with node features.
    Higher values indicate more feature variation across edges.
    
    Args:
        adjacency_matrix: Adjacency matrix (can be weighted)
        node_features: Node feature matrix
        
    Returns:
        Dirichlet energy value
    """
    try:
        # Convert to numpy if needed
        if isinstance(adjacency_matrix, torch.Tensor):
            adjacency_matrix = adjacency_matrix.cpu().numpy()
        if isinstance(node_features, torch.Tensor):
            node_features = node_features.cpu().numpy()
            
        # Normalize features
        if len(node_features.shape) == 1:
            node_features = node_features.reshape(-1, 1)
            
        # Normalize features to have unit norm
        norms = np.linalg.norm(node_features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_features = node_features / norms
        
        # Compute graph Laplacian
        if isinstance(adjacency_matrix, np.ndarray):
            adj_sparse = sp.csr_matrix(adjacency_matrix)
        else:
            adj_sparse = adjacency_matrix  # Assume it's already sparse
            
        diag = np.array(adj_sparse.sum(axis=1)).flatten()
        degree_mat = sp.diags(diag)
        laplacian = degree_mat - adj_sparse
        
        # Compute Dirichlet energy
        energy = 0
        
        # Use efficient matrix computation
        energy = 0.5 * np.sum(normalized_features.T @ laplacian @ normalized_features)
        
        return float(energy)
        
    except Exception as e:
        print(f"Error computing Dirichlet energy: {e}")
        return 0.0

def compute_extended_oversquashing_metrics(edge_index, edge_weights=None, node_features=None):
    """
    Compute extended oversquashing metrics including Cheeger constant and Dirichlet energy.
    
    Args:
        edge_index: Edge index tensor (2 x E)
        edge_weights: Optional edge weights
        node_features: Optional node features for Dirichlet energy
        
    Returns:
        Dict of metrics
    """
    # Create adjacency matrix from edge index
    if isinstance(edge_index, torch.Tensor):
        num_nodes = max(edge_index[0].max().item(), edge_index[1].max().item()) + 1
        device = edge_index.device
        
        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        adj[edge_index[0], edge_index[1]] = 1
        
        # Make undirected
        adj = (adj + adj.t()) > 0
        adj = adj.float()
        
        # Apply edge weights if provided
        if edge_weights is not None:
            rows, cols = adj.nonzero(as_tuple=True)
            if len(rows) > 0 and len(edge_weights) >= len(rows):
                adj_weighted = torch.zeros_like(adj)
                for i, (r, c) in enumerate(zip(rows, cols)):
                    if i < len(edge_weights):
                        adj_weighted[r, c] = edge_weights[i]
                adj = adj_weighted
    else:
        # Assume it's already an adjacency matrix
        adj = edge_index
        
    # Base oversquashing metrics
    base_metrics = compute_oversquashing_metric(adj)
    connectivity_metrics = compute_graph_connectivity_metrics(adj)
    
    # Extended metrics
    adj_np = adj.cpu().numpy() if isinstance(adj, torch.Tensor) else adj
    
    # Compute Cheeger constant
    cheeger = compute_cheeger_constant(adj_np)
    
    # Compute Dirichlet energy if node features provided
    dirichlet_energy = 0.0
    if node_features is not None:
        dirichlet_energy = compute_dirichlet_energy(adj_np, node_features)
        
    # Combine all metrics
    metrics = {
        **base_metrics,
        **connectivity_metrics,
        "cheeger_constant": float(cheeger),
        "dirichlet_energy": float(dirichlet_energy)
    }
    
    return metrics

def analyze_dataset(dataset_name, k=3, hidden_dim=16, num_epochs=10, batch_size=8, seed=42):
    """
    Run oversquashing analysis on the specified dataset.
    
    Args:
        dataset_name: Name of the dataset (MUTAG, PROTEINS, ENZYMES)
        k: Number of top-k connections
        hidden_dim: Hidden dimension size
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        seed: Random seed
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"oversquashing_analysis/balanced_{dataset_name.lower()}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    dataset = TUDataset(root='/workspaces/IPR-MPNN/datasets', name=dataset_name)
    
    # Get dataset stats
    print(f"\n=== Dataset: {dataset_name} ===")
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
    def train(model, optimizer, loader):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data in loader:
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
                
        return total_loss / len(loader.dataset), correct / total
    
    # Evaluation function
    def evaluate(model, loader):
        model.eval()
        correct = 0
        total = 0
        
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                out = model(data)
                pred = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                
        return correct / total
    
    # Training loop
    print(f"Training models for {num_epochs} epochs...")
    
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
        
        # Evaluate every few epochs
        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == num_epochs - 1:
            balanced_test_acc = evaluate(balanced_model, test_loader)
            balanced_test_accs.append(balanced_test_acc)
            
            uniform_test_acc = evaluate(uniform_model, test_loader)
            uniform_test_accs.append(uniform_test_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs}: Balanced=[Loss: {balanced_loss:.4f}, Acc: {balanced_test_acc:.4f}], "
                  f"Uniform=[Loss: {uniform_loss:.4f}, Acc: {uniform_test_acc:.4f}]")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}: Balanced Loss: {balanced_loss:.4f}, Uniform Loss: {uniform_loss:.4f}")
    
    # Oversquashing analysis
    print("Running oversquashing analysis...")
    
    # Enable weight tracking for analysis
    balanced_model.collect_oversquashing_metrics = True
    uniform_model.collect_oversquashing_metrics = True
    
    # Analyze a subset of test graphs
    num_samples = min(10, len(test_dataset))
    analysis_dataloader = DataLoader(test_dataset[:num_samples], batch_size=1, shuffle=False)
    
    balanced_metrics = []
    uniform_metrics = []
    
    for idx, data in enumerate(tqdm(analysis_dataloader, desc="Analyzing graphs")):
        try:
            data = data.to(device)
            
            # Process with models to collect edge weights
            with torch.no_grad():
                # Balanced model
                balanced_model(data)
                balanced_weights = balanced_model.get_final_edge_weights(0)
                
                # Uniform model
                uniform_model(data)
                uniform_weights = uniform_model.get_final_edge_weights(0)
                
                if balanced_weights and uniform_weights:
                    # Extract edge weights
                    balanced_edge_weights = balanced_weights.get('edge_weights', None)
                    uniform_edge_weights = uniform_weights.get('edge_weights', None)
                    
                    if balanced_edge_weights is not None and uniform_edge_weights is not None:
                        # Compute metrics using edge weights
                        balanced_metrics_dict = compute_extended_oversquashing_metrics(
                            data.edge_index, 
                            balanced_edge_weights,
                            data.x
                        )
                        
                        uniform_metrics_dict = compute_extended_oversquashing_metrics(
                            data.edge_index,
                            uniform_edge_weights,
                            data.x
                        )
                        
                        # Store metrics
                        balanced_metrics.append({
                            'graph_idx': idx,
                            'num_nodes': balanced_weights['num_nodes'],
                            'metrics': balanced_metrics_dict
                        })
                        
                        uniform_metrics.append({
                            'graph_idx': idx,
                            'num_nodes': uniform_weights['num_nodes'],
                            'metrics': uniform_metrics_dict
                        })
                        
                        if idx == 0:  # Print metrics for first graph
                            print(f"\nGraph 0 oversquashing metrics:")
                            print(f"  Balanced - Mean Effective Resistance: {balanced_metrics_dict['mean_effective_resistance']:.4f}")
                            print(f"  Uniform - Mean Effective Resistance: {uniform_metrics_dict['mean_effective_resistance']:.4f}")
                            print(f"  Balanced - Cheeger Constant: {balanced_metrics_dict['cheeger_constant']:.4f}")
                            print(f"  Uniform - Cheeger Constant: {uniform_metrics_dict['cheeger_constant']:.4f}")
                    else:
                        print(f"No edge weights found for graph {idx}")
                else:
                    print(f"Failed to get weights for graph {idx}")
        except Exception as e:
            print(f"Error analyzing graph {idx}: {e}")
    
    # Disable weight tracking
    balanced_model.collect_oversquashing_metrics = False
    uniform_model.collect_oversquashing_metrics = False
    
    # Final evaluation
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
    results_file = os.path.join(output_dir, f"{dataset_name.lower()}_analysis.json")
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
    
    # Create summary visualizations
    create_summary_visualization(balanced_metrics, uniform_metrics, dataset_name, output_dir)
    
    # Clean up memory
    del balanced_model, uniform_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

def create_summary_visualization(balanced_metrics, uniform_metrics, dataset_name, output_dir):
    """
    Create summary visualizations comparing balanced and uniform initialization.
    
    Args:
        balanced_metrics: List of metrics for balanced initialization
        uniform_metrics: List of metrics for uniform initialization
        dataset_name: Name of the dataset
        output_dir: Directory to save visualizations
    """
    if not balanced_metrics or not uniform_metrics:
        print("No metrics to visualize")
        return
    
    # Extract metrics
    balanced_resistance = [m['metrics']['mean_effective_resistance'] for m in balanced_metrics 
                          if 'metrics' in m and 'mean_effective_resistance' in m['metrics']]
    
    uniform_resistance = [m['metrics']['mean_effective_resistance'] for m in uniform_metrics
                         if 'metrics' in m and 'mean_effective_resistance' in m['metrics']]
    
    balanced_cheeger = [m['metrics']['cheeger_constant'] for m in balanced_metrics
                       if 'metrics' in m and 'cheeger_constant' in m['metrics']]
    
    uniform_cheeger = [m['metrics']['cheeger_constant'] for m in uniform_metrics
                      if 'metrics' in m and 'cheeger_constant' in m['metrics']]
    
    balanced_spectral = [m['metrics']['spectral_gap'] for m in balanced_metrics
                        if 'metrics' in m and 'spectral_gap' in m['metrics']]
    
    uniform_spectral = [m['metrics']['spectral_gap'] for m in uniform_metrics
                       if 'metrics' in m and 'spectral_gap' in m['metrics']]
    
    balanced_dirichlet = [m['metrics']['dirichlet_energy'] for m in balanced_metrics
                         if 'metrics' in m and 'dirichlet_energy' in m['metrics']]
    
    uniform_dirichlet = [m['metrics']['dirichlet_energy'] for m in uniform_metrics
                        if 'metrics' in m and 'dirichlet_energy' in m['metrics']]
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Effective resistance
    plt.subplot(2, 2, 1)
    idx = range(len(balanced_resistance))
    plt.bar([i - 0.2 for i in idx], balanced_resistance, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in idx], uniform_resistance, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title(f'Effective Resistance - {dataset_name}')
    plt.xlabel('Graph Index')
    plt.ylabel('Mean Effective Resistance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cheeger constant
    plt.subplot(2, 2, 2)
    plt.bar([i - 0.2 for i in idx], balanced_cheeger, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in idx], uniform_cheeger, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title(f'Cheeger Constant - {dataset_name}')
    plt.xlabel('Graph Index')
    plt.ylabel('Cheeger Constant')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Spectral gap
    plt.subplot(2, 2, 3)
    plt.bar([i - 0.2 for i in idx], balanced_spectral, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in idx], uniform_spectral, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title(f'Spectral Gap - {dataset_name}')
    plt.xlabel('Graph Index')
    plt.ylabel('Spectral Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Dirichlet energy
    plt.subplot(2, 2, 4)
    plt.bar([i - 0.2 for i in idx], balanced_dirichlet, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in idx], uniform_dirichlet, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title(f'Dirichlet Energy - {dataset_name}')
    plt.xlabel('Graph Index')
    plt.ylabel('Dirichlet Energy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name.lower()}_metrics_comparison.png"))
    plt.close()
    
    # Create summary boxplot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Effective resistance boxplot
    plt.subplot(2, 2, 1)
    plt.boxplot([balanced_resistance, uniform_resistance], labels=['Balanced', 'Uniform'])
    plt.title(f'Effective Resistance Distribution - {dataset_name}')
    plt.ylabel('Effective Resistance')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cheeger constant boxplot
    plt.subplot(2, 2, 2)
    plt.boxplot([balanced_cheeger, uniform_cheeger], labels=['Balanced', 'Uniform'])
    plt.title(f'Cheeger Constant Distribution - {dataset_name}')
    plt.ylabel('Cheeger Constant')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Spectral gap boxplot
    plt.subplot(2, 2, 3)
    plt.boxplot([balanced_spectral, uniform_spectral], labels=['Balanced', 'Uniform'])
    plt.title(f'Spectral Gap Distribution - {dataset_name}')
    plt.ylabel('Spectral Gap')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Dirichlet energy boxplot
    plt.subplot(2, 2, 4)
    plt.boxplot([balanced_dirichlet, uniform_dirichlet], labels=['Balanced', 'Uniform'])
    plt.title(f'Dirichlet Energy Distribution - {dataset_name}')
    plt.ylabel('Dirichlet Energy')
    plt.grid(True, alpha=0.3)
    
    # Save boxplot figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name.lower()}_metrics_boxplot.png"))
    plt.close()
    
    # Create summary table as markdown
    with open(os.path.join(output_dir, f"{dataset_name.lower()}_metrics_summary.md"), 'w') as f:
        f.write(f"# Oversquashing Metrics Summary for {dataset_name}\n\n")
        
        # Calculate mean metrics
        balanced_resistance_mean = np.mean(balanced_resistance) if balanced_resistance else float('nan')
        uniform_resistance_mean = np.mean(uniform_resistance) if uniform_resistance else float('nan')
        resistance_diff = balanced_resistance_mean - uniform_resistance_mean
        
        balanced_cheeger_mean = np.mean(balanced_cheeger) if balanced_cheeger else float('nan')
        uniform_cheeger_mean = np.mean(uniform_cheeger) if uniform_cheeger else float('nan')
        cheeger_diff = balanced_cheeger_mean - uniform_cheeger_mean
        
        balanced_spectral_mean = np.mean(balanced_spectral) if balanced_spectral else float('nan')
        uniform_spectral_mean = np.mean(uniform_spectral) if uniform_spectral else float('nan')
        spectral_diff = balanced_spectral_mean - uniform_spectral_mean
        
        balanced_dirichlet_mean = np.mean(balanced_dirichlet) if balanced_dirichlet else float('nan')
        uniform_dirichlet_mean = np.mean(uniform_dirichlet) if uniform_dirichlet else float('nan')
        dirichlet_diff = balanced_dirichlet_mean - uniform_dirichlet_mean
        
        # Write table
        f.write("## Mean Metrics\n\n")
        f.write("| Metric | Balanced Cayley | Uniform | Difference | Better Approach |\n")
        f.write("|--------|----------------|---------|------------|----------------|\n")
        
        # Effective resistance (lower is better)
        resistance_better = "Balanced" if resistance_diff < 0 else "Uniform"
        f.write(f"| Effective Resistance | {balanced_resistance_mean:.4f} | {uniform_resistance_mean:.4f} | {resistance_diff:.4f} | {resistance_better} |\n")
        
        # Cheeger constant (higher is better)
        cheeger_better = "Balanced" if cheeger_diff > 0 else "Uniform"
        f.write(f"| Cheeger Constant | {balanced_cheeger_mean:.4f} | {uniform_cheeger_mean:.4f} | {cheeger_diff:.4f} | {cheeger_better} |\n")
        
        # Spectral gap (higher is better)
        spectral_better = "Balanced" if spectral_diff > 0 else "Uniform"
        f.write(f"| Spectral Gap | {balanced_spectral_mean:.4f} | {uniform_spectral_mean:.4f} | {spectral_diff:.4f} | {spectral_better} |\n")
        
        # Dirichlet energy (depends on use case)
        dirichlet_better = "Balanced" if dirichlet_diff > 0 else "Uniform"
        f.write(f"| Dirichlet Energy | {balanced_dirichlet_mean:.4f} | {uniform_dirichlet_mean:.4f} | {dirichlet_diff:.4f} | {dirichlet_better} |\n")
        
        # Add interpretation
        f.write("\n## Interpretation\n\n")
        f.write("- **Lower effective resistance** indicates less oversquashing\n")
        f.write("- **Higher Cheeger constant** indicates better graph connectivity (fewer bottlenecks)\n")
        f.write("- **Higher spectral gap** indicates faster information mixing in the graph\n")
        f.write("- **Higher Dirichlet energy** indicates more pronounced feature differences across graph edges\n\n")
        
        # Overall assessment
        f.write("## Overall Assessment\n\n")
        f.write("Based on the metrics above:\n\n")
        
        count_balanced = 0
        count_uniform = 0
        
        if resistance_better == "Balanced":
            count_balanced += 1
            f.write("- **Balanced Cayley** shows less oversquashing (lower effective resistance)\n")
        else:
            count_uniform += 1
            f.write("- **Uniform** shows less oversquashing (lower effective resistance)\n")
            
        if cheeger_better == "Balanced":
            count_balanced += 1
            f.write("- **Balanced Cayley** has better connectivity (higher Cheeger constant)\n")
        else:
            count_uniform += 1
            f.write("- **Uniform** has better connectivity (higher Cheeger constant)\n")
            
        if spectral_better == "Balanced":
            count_balanced += 1
            f.write("- **Balanced Cayley** has faster information mixing (higher spectral gap)\n")
        else:
            count_uniform += 1
            f.write("- **Uniform** has faster information mixing (higher spectral gap)\n")
            
        if dirichlet_better == "Balanced":
            count_balanced += 1
            f.write("- **Balanced Cayley** has higher feature distinction across edges (higher Dirichlet energy)\n")
        else:
            count_uniform += 1
            f.write("- **Uniform** has higher feature distinction across edges (higher Dirichlet energy)\n")
            
        f.write(f"\n**Summary:** {count_balanced} metrics favor Balanced Cayley, {count_uniform} metrics favor Uniform initialization.\n")
        
        if count_balanced > count_uniform:
            f.write("\n**Overall, Balanced Cayley initialization appears to be more effective at reducing oversquashing for this dataset.**\n")
        elif count_uniform > count_balanced:
            f.write("\n**Overall, Uniform initialization appears to be more effective at reducing oversquashing for this dataset.**\n")
        else:
            f.write("\n**Both initialization methods show comparable performance in reducing oversquashing for this dataset.**\n")
    
    print(f"Visualization and summary saved to {output_dir}")

if __name__ == "__main__":
    # Set shorter epochs for testing
    num_epochs = 10  # Reduced for faster results
    
    # Analyze MUTAG dataset (small and fast)
    mutag_results = analyze_dataset(
        dataset_name="MUTAG",
        k=3,
        hidden_dim=16,
        num_epochs=num_epochs,
        batch_size=8,
        seed=42
    )
    
    # Analyze PROTEINS dataset (medium size)
    proteins_results = analyze_dataset(
        dataset_name="PROTEINS",
        k=3,
        hidden_dim=16,
        num_epochs=num_epochs,
        batch_size=8,
        seed=42
    )
    
    # Analyze ENZYMES dataset (larger)
    enzymes_results = analyze_dataset(
        dataset_name="ENZYMES",
        k=3,
        hidden_dim=16,
        num_epochs=num_epochs,
        batch_size=8,
        seed=42
    )
    
    print("\nAnalysis complete for all datasets!")
