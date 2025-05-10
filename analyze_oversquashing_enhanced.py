"""
Enhanced oversquashing metrics analysis script for comparing balanced Cayley vs. uniform initialization.
This script adds additional metrics like Cheeger constant and Dirichlet energy to evaluate 
graph connectivity and message passing efficiency.
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
    if n <= 20:
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

def compute_extended_oversquashing_metrics(edge_index, edge_weights=None, node_features=None):
    """
    Compute extended oversquashing metrics including Cheeger constant and Dirichlet energy.
    
    Args:
        edge_index: Edge index of the graph
        edge_weights: Optional edge weights
        node_features: Optional node features
        
    Returns:
        Dict with extended oversquashing metrics
    """
    # Get standard oversquashing metrics
    standard_metrics = compute_oversquashing_metric(edge_index, edge_weights)
    connectivity_metrics = compute_graph_connectivity_metrics(edge_index, edge_weights)
    
    # Convert edge_index to adjacency matrix for additional metrics
    if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 and edge_index.size(0) == 2:
        # Determine number of nodes
        num_nodes = max(edge_index[0].max().item(), edge_index[1].max().item()) + 1
        
        # Create adjacency matrix
        device = edge_index.device
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        adj[edge_index[0], edge_index[1]] = 1
        
        # Make undirected if it's not already
        adj = (adj + adj.t()) > 0
        adj = adj.float()
        
        # Apply edge weights if provided
        if edge_weights is not None and len(edge_weights) > 0:
            rows, cols = edge_index
            for i, (r, c) in enumerate(zip(rows, cols)):
                if i < len(edge_weights):
                    adj[r, c] = edge_weights[i]
    else:
        # Assume it's already an adjacency matrix
        adj = edge_index.float()
    
    # Calculate additional metrics
    cheeger_constant = compute_cheeger_constant(adj)
    
    # Calculate Dirichlet energy if node features are provided
    dirichlet_energy = 0.0
    if node_features is not None:
        dirichlet_energy = compute_dirichlet_energy(adj, node_features)
    
    # Combine all metrics
    extended_metrics = {
        **standard_metrics,
        **connectivity_metrics,
        "cheeger_constant": cheeger_constant,
        "dirichlet_energy": dirichlet_energy if node_features is not None else None
    }
    
    return extended_metrics

def analyze_initialization_oversquashing(dataset_name, k=3, hidden_dim=16, num_graphs=10, seed=42):
    """
    Analyze the oversquashing properties of balanced Cayley vs. uniform initialization
    without training, focusing on the initial connectivity patterns.
    
    Args:
        dataset_name: Name of the dataset (MUTAG, PROTEINS, ENZYMES)
        k: Number of top-k connections
        hidden_dim: Hidden dimension size
        num_graphs: Number of graphs to analyze
        seed: Random seed
    
    Returns:
        Dict with analysis results
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"oversquashing_analysis/init_analysis_{timestamp}"
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
    balanced_model.enable_oversquashing_tracking()
    uniform_model.enable_oversquashing_tracking()
    
    # Collect metrics
    balanced_metrics = []
    uniform_metrics = []
    
    # Process each graph
    for idx, data in enumerate(tqdm(data_loader, desc=f"Analyzing {dataset_name} graphs")):
        data = data.to(device)
        
        with torch.no_grad():
            # Forward pass through balanced model
            balanced_model(data)
            balanced_weights = balanced_model.get_final_edge_weights(0)
            
            # Forward pass through uniform model
            uniform_model(data)
            uniform_weights = uniform_model.get_final_edge_weights(0)
            
            if balanced_weights and uniform_weights:
                # Get edge weights
                balanced_edge_weights = balanced_weights.get('edge_weights', None)
                uniform_edge_weights = uniform_weights.get('edge_weights', None)
                
                # Calculate extended metrics
                try:
                    balanced_metrics.append({
                        'graph_idx': idx,
                        'num_nodes': balanced_weights['num_nodes'],
                        'num_virtual_nodes': balanced_weights['num_virtual_nodes'],
                        'metrics': compute_extended_oversquashing_metrics(
                            data.edge_index, 
                            balanced_edge_weights,
                            data.x
                        )
                    })
                    
                    uniform_metrics.append({
                        'graph_idx': idx,
                        'num_nodes': uniform_weights['num_nodes'],
                        'num_virtual_nodes': uniform_weights['num_virtual_nodes'],
                        'metrics': compute_extended_oversquashing_metrics(
                            data.edge_index,
                            uniform_edge_weights,
                            data.x
                        )
                    })
                    
                    print(f"Graph {idx}: Processed successfully")
                except Exception as e:
                    print(f"Error processing graph {idx}: {e}")
    
    # Disable tracking
    balanced_model.disable_oversquashing_tracking()
    uniform_model.disable_oversquashing_tracking()
    
    # Compile results
    results = {
        'dataset': dataset_name,
        'parameters': {
            'top_k': k,
            'hidden_dim': hidden_dim,
            'seed': seed,
            'num_graphs': num_graphs
        },
        'oversquashing_metrics': {
            'balanced_cayley': balanced_metrics,
            'uniform': uniform_metrics
        }
    }
    
    # Save results
    results_file = os.path.join(output_dir, f"{dataset_name.lower()}_init_analysis.json")
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
    
    return results

def run_oversquashing_analysis_after_training(dataset_name, k=3, hidden_dim=16, 
                                             batch_size=4, num_epochs=10, seed=42):
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
    output_dir = f"oversquashing_analysis/trained_analysis_{timestamp}"
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
                    # Get actual edge weights for analysis
                    balanced_edge_weights = balanced_weights.get('edge_weights', None)
                    uniform_edge_weights = uniform_weights.get('edge_weights', None)
                    
                    try:
                        # Calculate extended oversquashing metrics
                        balanced_extended_metrics = compute_extended_oversquashing_metrics(
                            data.edge_index, 
                            balanced_edge_weights,
                            data.x
                        )
                        
                        uniform_extended_metrics = compute_extended_oversquashing_metrics(
                            data.edge_index,
                            uniform_edge_weights,
                            data.x
                        )
                        
                        # Store metrics
                        balanced_metrics.append({
                            'graph_idx': idx,
                            'num_nodes': balanced_weights['num_nodes'],
                            'num_virtual_nodes': balanced_weights['num_virtual_nodes'],
                            'metrics': balanced_extended_metrics
                        })
                        
                        uniform_metrics.append({
                            'graph_idx': idx,
                            'num_nodes': uniform_weights['num_nodes'],
                            'num_virtual_nodes': uniform_weights['num_virtual_nodes'],
                            'metrics': uniform_extended_metrics
                        })
                        
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
    results_file = os.path.join(output_dir, f"{dataset_name.lower()}_trained_analysis.json")
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
    visualize_metrics(balanced_metrics, uniform_metrics, dataset_name, output_dir, trained=True)
    
    # Clean up memory
    del balanced_model, uniform_model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

def visualize_metrics(balanced_metrics, uniform_metrics, dataset_name, output_dir, trained=False):
    """
    Visualize the oversquashing metrics comparison.
    
    Args:
        balanced_metrics: List of metrics dicts for balanced Cayley approach
        uniform_metrics: List of metrics dicts for uniform approach
        dataset_name: Name of the dataset
        output_dir: Directory to save visualizations
        trained: Whether this is for trained or initial analysis
    """
    if not balanced_metrics or not uniform_metrics:
        print("No metrics to visualize")
        return
    
    # Extract key metrics
    balanced_resistance = [m['metrics']['mean_effective_resistance'] for m in balanced_metrics]
    uniform_resistance = [m['metrics']['mean_effective_resistance'] for m in uniform_metrics]
    
    balanced_cheeger = [m['metrics']['cheeger_constant'] for m in balanced_metrics]
    uniform_cheeger = [m['metrics']['cheeger_constant'] for m in uniform_metrics]
    
    balanced_spectral_gap = [m['metrics']['spectral_gap'] for m in balanced_metrics]
    uniform_spectral_gap = [m['metrics']['spectral_gap'] for m in uniform_metrics]
    
    balanced_path_length = [m['metrics']['avg_path_length'] for m in balanced_metrics 
                          if m['metrics']['avg_path_length'] != float('inf')]
    uniform_path_length = [m['metrics']['avg_path_length'] for m in uniform_metrics
                         if m['metrics']['avg_path_length'] != float('inf')]
    
    # Try to extract Dirichlet energy if available
    balanced_dirichlet = []
    uniform_dirichlet = []
    try:
        balanced_dirichlet = [m['metrics']['dirichlet_energy'] for m in balanced_metrics 
                             if m['metrics']['dirichlet_energy'] is not None]
        uniform_dirichlet = [m['metrics']['dirichlet_energy'] for m in uniform_metrics
                            if m['metrics']['dirichlet_energy'] is not None]
    except:
        print("Dirichlet energy not available")
    
    # Create visualization
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Effective resistance
    plt.subplot(3, 2, 1)
    idx = range(len(balanced_resistance))
    plt.bar([i - 0.2 for i in idx], balanced_resistance, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in idx], uniform_resistance, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title(f'Effective Resistance ({dataset_name})')
    plt.xlabel('Graph Index')
    plt.ylabel('Mean Effective Resistance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cheeger constant
    plt.subplot(3, 2, 2)
    plt.bar([i - 0.2 for i in idx], balanced_cheeger, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in idx], uniform_cheeger, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title(f'Cheeger Constant ({dataset_name})')
    plt.xlabel('Graph Index')
    plt.ylabel('Cheeger Constant')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Spectral gap
    plt.subplot(3, 2, 3)
    plt.bar([i - 0.2 for i in idx], balanced_spectral_gap, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in idx], uniform_spectral_gap, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title(f'Spectral Gap ({dataset_name})')
    plt.xlabel('Graph Index')
    plt.ylabel('Spectral Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Average path length
    plt.subplot(3, 2, 4)
    if balanced_path_length and uniform_path_length:
        idx_path = range(min(len(balanced_path_length), len(uniform_path_length)))
        plt.bar([i - 0.2 for i in idx_path], balanced_path_length[:len(idx_path)], width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
        plt.bar([i + 0.2 for i in idx_path], uniform_path_length[:len(idx_path)], width=0.4, label='Uniform', color='orange', alpha=0.7)
        plt.title(f'Average Path Length ({dataset_name})')
        plt.xlabel('Graph Index')
        plt.ylabel('Average Path Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "Insufficient path length data", horizontalalignment='center', verticalalignment='center')
    
    # Plot 5: Dirichlet energy if available
    plt.subplot(3, 2, 5)
    if balanced_dirichlet and uniform_dirichlet:
        idx_dir = range(min(len(balanced_dirichlet), len(uniform_dirichlet)))
        plt.bar([i - 0.2 for i in idx_dir], balanced_dirichlet[:len(idx_dir)], width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
        plt.bar([i + 0.2 for i in idx_dir], uniform_dirichlet[:len(idx_dir)], width=0.4, label='Uniform', color='orange', alpha=0.7)
        plt.title(f'Dirichlet Energy ({dataset_name})')
        plt.xlabel('Graph Index')
        plt.ylabel('Dirichlet Energy')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "Dirichlet energy not available", horizontalalignment='center', verticalalignment='center')
    
    # Plot 6: Summary boxplot
    plt.subplot(3, 2, 6)
    boxplot_data = [
        balanced_resistance,
        uniform_resistance
    ]
    plt.boxplot(boxplot_data, labels=['Balanced Cayley', 'Uniform'])
    plt.title(f'Effective Resistance Distribution ({dataset_name})')
    plt.ylabel('Effective Resistance')
    plt.grid(True, alpha=0.3)
    
    # Save visualization
    analysis_type = "trained" if trained else "init"
    plt.tight_layout()
    vis_file = os.path.join(output_dir, f"{dataset_name.lower()}_{analysis_type}_metrics.png")
    plt.savefig(vis_file)
    plt.close()
    
    print(f"Visualization saved to {vis_file}")
    
    # Also create a summary visualization with means
    plt.figure(figsize=(12, 8))
    
    metrics = [
        ('Effective Resistance', np.mean(balanced_resistance), np.mean(uniform_resistance)),
        ('Cheeger Constant', np.mean(balanced_cheeger), np.mean(uniform_cheeger)),
        ('Spectral Gap', np.mean(balanced_spectral_gap), np.mean(uniform_spectral_gap)),
        ('Avg Path Length', np.mean(balanced_path_length) if balanced_path_length else 0, 
                            np.mean(uniform_path_length) if uniform_path_length else 0)
    ]
    
    if balanced_dirichlet and uniform_dirichlet:
        metrics.append(('Dirichlet Energy', np.mean(balanced_dirichlet), np.mean(uniform_dirichlet)))
    
    labels = [m[0] for m in metrics]
    balanced_values = [m[1] for m in metrics]
    uniform_values = [m[2] for m in metrics]
    
    x = range(len(labels))
    
    plt.bar([i - 0.2 for i in x], balanced_values, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in x], uniform_values, width=0.4, label='Uniform', color='orange', alpha=0.7)
    
    plt.title(f'Oversquashing Metrics Summary ({dataset_name} - {"Trained" if trained else "Initial"})')
    plt.xlabel('Metric')
    plt.ylabel('Average Value')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save summary visualization
    summary_file = os.path.join(output_dir, f"{dataset_name.lower()}_{analysis_type}_summary.png")
    plt.tight_layout()
    plt.savefig(summary_file)
    plt.close()
    
    print(f"Summary visualization saved to {summary_file}")

def run_all_datasets():
    """Run analysis on all three datasets: MUTAG, PROTEINS, ENZYMES"""
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # First, run initialization analysis without training
    print("\n=== Running Initialization Oversquashing Analysis ===")
    for dataset in ["MUTAG", "PROTEINS", "ENZYMES"]:
        print(f"\nAnalyzing initialization for {dataset}...")
        analyze_initialization_oversquashing(
            dataset_name=dataset,
            k=3,
            hidden_dim=16,
            num_graphs=10,
            seed=seed
        )
    
    # Then, run analysis after training
    print("\n=== Running Trained Models Oversquashing Analysis ===")
    for dataset in ["MUTAG", "PROTEINS", "ENZYMES"]:
        print(f"\nTraining and analyzing {dataset}...")
        num_epochs = 10 if dataset == "MUTAG" else 8  # Fewer epochs for larger datasets
        run_oversquashing_analysis_after_training(
            dataset_name=dataset,
            k=3,
            hidden_dim=16,
            batch_size=4,
            num_epochs=num_epochs,
            seed=seed
        )

if __name__ == "__main__":
    # Run single dataset test with both analyses
    print("=== Running Oversquashing Analysis for MUTAG ===")
    
    # Analyze initial state (without training)
    mutag_init_results = analyze_initialization_oversquashing(
        dataset_name="MUTAG",
        k=3,
        hidden_dim=16,
        num_graphs=10,
        seed=42
    )
    
    # Analyze after training
    mutag_trained_results = run_oversquashing_analysis_after_training(
        dataset_name="MUTAG",
        k=3,
        hidden_dim=16,
        batch_size=4,
        num_epochs=8,  # Reduced epochs for faster testing
        seed=42
    )
    
    # To run on all datasets, uncomment this:
    # run_all_datasets()
