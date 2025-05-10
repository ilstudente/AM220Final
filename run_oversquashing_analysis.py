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

from models.memory_saver_iprmpnn import MemorySaverIPRMPNNModel
from utils.oversquashing_metrics import compute_oversquashing_metric, compute_graph_connectivity_metrics

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def summarize_oversquashing_comparison(balanced_metrics, uniform_metrics):
    """
    Summarize the oversquashing comparison between balanced Cayley and uniform methods.
    
    Args:
        balanced_metrics: List of metrics dicts for balanced Cayley approach
        uniform_metrics: List of metrics dicts for uniform approach
        
    Returns:
        Dict with summary metrics
    """
    if not balanced_metrics or not uniform_metrics:
        return {
            "error": "Insufficient metrics data for comparison"
        }
    
    # Extract key metrics
    balanced_resistance = [m['oversquashing']['mean_effective_resistance'] for m in balanced_metrics
                          if 'oversquashing' in m and 'mean_effective_resistance' in m['oversquashing']]
    
    uniform_resistance = [m['oversquashing']['mean_effective_resistance'] for m in uniform_metrics
                         if 'oversquashing' in m and 'mean_effective_resistance' in m['oversquashing']]
    
    balanced_path_length = [m['connectivity']['avg_path_length'] for m in balanced_metrics
                           if 'connectivity' in m and 'avg_path_length' in m['connectivity']
                           and m['connectivity']['avg_path_length'] != float('inf')]
    
    uniform_path_length = [m['connectivity']['avg_path_length'] for m in uniform_metrics
                          if 'connectivity' in m and 'avg_path_length' in m['connectivity']
                          and m['connectivity']['avg_path_length'] != float('inf')]
    
    balanced_spectral_gap = [m['connectivity']['spectral_gap'] for m in balanced_metrics
                            if 'connectivity' in m and 'spectral_gap' in m['connectivity']]
    
    uniform_spectral_gap = [m['connectivity']['spectral_gap'] for m in uniform_metrics
                           if 'connectivity' in m and 'spectral_gap' in m['connectivity']]
    
    # Compute means if data available
    balanced_mean_resistance = np.mean(balanced_resistance) if balanced_resistance else float('inf')
    uniform_mean_resistance = np.mean(uniform_resistance) if uniform_resistance else float('inf')
    
    balanced_mean_path_length = np.mean(balanced_path_length) if balanced_path_length else float('inf')
    uniform_mean_path_length = np.mean(uniform_path_length) if uniform_path_length else float('inf')
    
    balanced_mean_spectral_gap = np.mean(balanced_spectral_gap) if balanced_spectral_gap else 0
    uniform_mean_spectral_gap = np.mean(uniform_spectral_gap) if uniform_spectral_gap else 0
    
    # Calculate differences (positive means balanced is better for spectral gap, negative means balanced is better for others)
    resistance_diff = balanced_mean_resistance - uniform_mean_resistance
    path_length_diff = balanced_mean_path_length - uniform_mean_path_length
    spectral_gap_diff = balanced_mean_spectral_gap - uniform_mean_spectral_gap
    
    # Determine which approach is better for each metric
    resistance_winner = "balanced_cayley" if resistance_diff < 0 else "uniform"
    path_length_winner = "balanced_cayley" if path_length_diff < 0 else "uniform"
    spectral_gap_winner = "balanced_cayley" if spectral_gap_diff > 0 else "uniform"
    
    # Create summary
    summary = {
        "mean_effective_resistance": {
            "balanced_cayley": float(balanced_mean_resistance),
            "uniform": float(uniform_mean_resistance),
            "difference": float(resistance_diff),
            "better_approach": resistance_winner
        },
        "avg_path_length": {
            "balanced_cayley": float(balanced_mean_path_length),
            "uniform": float(uniform_mean_path_length),
            "difference": float(path_length_diff),
            "better_approach": path_length_winner
        },
        "spectral_gap": {
            "balanced_cayley": float(balanced_mean_spectral_gap),
            "uniform": float(uniform_mean_spectral_gap),
            "difference": float(spectral_gap_diff),
            "better_approach": spectral_gap_winner
        },
        "overall_assessment": {
            "less_oversquashing": resistance_winner,
            "better_connectivity": path_length_winner,
            "faster_mixing": spectral_gap_winner
        }
    }
    
    return summary

def run_experiment_with_oversquashing_analysis(dataset_name, k=3, hidden_dim=16, batch_size=4, 
                                              num_epochs=15, seed=42, output_dir=None):
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
        output_dir: Directory to save results
    
    Returns:
        Dict with results and oversquashing metrics
    """
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = f"oversquashing_analysis/analysis_{timestamp}"
    
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
            
            # Enable weight tracking for the last batch
            if data_loader.batch_size >= data.num_graphs:
                model.enable_oversquashing_tracking()
                
            try:
                # Forward pass
                out = model(data)
                loss = criterion(out, data.y)
                
                # Backward pass with gradient clipping to prevent exploding gradients
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
                
            # Disable weight tracking to save memory
            model.disable_oversquashing_tracking()
            
        return total_loss / len(data_loader.dataset), correct / total
    
    # Evaluation function
    def evaluate(model, data_loader):
        model.eval()
        correct = 0
        total = 0
        
        for data in data_loader:
            data = data.to(device)
            with torch.no_grad():
                # Enable weight tracking for the last batch only
                if data_loader.batch_size >= data.num_graphs:
                    model.enable_oversquashing_tracking()
                    
                out = model(data)
                pred = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                
                # Disable weight tracking
                model.disable_oversquashing_tracking()
                
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
        
        # Evaluate every few epochs
        if (epoch + 1) % 3 == 0 or epoch == 0 or epoch == num_epochs - 1:
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
    print("Enabling oversquashing tracking for both models...")
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
                print(f"Graph {idx} - Balanced weights available: {balanced_weights is not None}")
                if balanced_weights:
                    print(f"  Balanced keys: {balanced_weights.keys()}")
                
                # Process with uniform model
                uniform_model(data)
                uniform_weights = uniform_model.get_final_edge_weights(0)
                print(f"Graph {idx} - Uniform weights available: {uniform_weights is not None}")
                if uniform_weights:
                    print(f"  Uniform keys: {uniform_weights.keys()}")
                
                if balanced_weights and uniform_weights:
                    # Get actual edge weights for analysis
                    balanced_edge_weights = balanced_weights.get('edge_weights', None)
                    uniform_edge_weights = uniform_weights.get('edge_weights', None)
                    
                    # Get the original graph structure
                    edge_index = data.edge_index
                    
                    try:
                        # Calculate oversquashing metrics for both approaches
                        balanced_oversquashing = compute_oversquashing_metric(edge_index, balanced_edge_weights)
                        balanced_connectivity = compute_graph_connectivity_metrics(edge_index, balanced_edge_weights)
                        
                        uniform_oversquashing = compute_oversquashing_metric(edge_index, uniform_edge_weights)
                        uniform_connectivity = compute_graph_connectivity_metrics(edge_index, uniform_edge_weights)
                        
                        # Store metrics
                        balanced_metrics.append({
                            'graph_idx': idx,
                            'num_nodes': balanced_weights['num_nodes'],
                            'num_virtual_nodes': balanced_weights['num_virtual_nodes'],
                            'oversquashing': balanced_oversquashing,
                            'connectivity': balanced_connectivity
                        })
                        
                        uniform_metrics.append({
                            'graph_idx': idx,
                            'num_nodes': uniform_weights['num_nodes'],
                            'num_virtual_nodes': uniform_weights['num_virtual_nodes'], 
                            'oversquashing': uniform_oversquashing,
                            'connectivity': uniform_connectivity
                        })
                        
                        if idx == 0:  # Print detailed metrics for the first graph
                            print(f"\nDetailed metrics for graph 0:")
                            print(f"Balanced Cayley - Mean Effective Resistance: {balanced_oversquashing['mean_effective_resistance']:.4f}")
                            print(f"Uniform - Mean Effective Resistance: {uniform_oversquashing['mean_effective_resistance']:.4f}")
                            print(f"Balanced Cayley - Avg Path Length: {balanced_connectivity['avg_path_length']:.4f}")
                            print(f"Uniform - Avg Path Length: {uniform_connectivity['avg_path_length']:.4f}")
                            
                    except Exception as e:
                        print(f"Error calculating metrics for graph {idx}: {e}")
        except Exception as e:
            print(f"Error processing graph {idx}: {e}")
            continue
    
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
        },
        'analysis_summary': summarize_oversquashing_comparison(balanced_metrics, uniform_metrics)
    }
    
    # Save results to file
    results_file = os.path.join(output_dir, f"{dataset_name.lower()}_oversquashing_analysis.json")
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
    
    # Visualize oversquashing metrics
    visualize_oversquashing_metrics(balanced_metrics, uniform_metrics, dataset_name, output_dir)
    
    # Clean up memory
    del balanced_model, uniform_model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

def visualize_oversquashing_metrics(balanced_metrics, uniform_metrics, dataset_name, output_dir):
    """
    Visualize oversquashing metrics comparison between balanced Cayley and uniform approaches.
    
    Args:
        balanced_metrics: List of metrics dicts for balanced Cayley approach
        uniform_metrics: List of metrics dicts for uniform approach
        dataset_name: Name of the dataset
        output_dir: Directory to save visualization
    """
    if not balanced_metrics or not uniform_metrics:
        print("No metrics to visualize")
        return
    
    # Extract metrics
    balanced_resistance = [m['oversquashing']['mean_effective_resistance'] for m in balanced_metrics]
    uniform_resistance = [m['oversquashing']['mean_effective_resistance'] for m in uniform_metrics]
    
    balanced_path_length = [m['connectivity']['avg_path_length'] for m in balanced_metrics 
                          if m['connectivity']['avg_path_length'] != float('inf')]
    uniform_path_length = [m['connectivity']['avg_path_length'] for m in uniform_metrics
                         if m['connectivity']['avg_path_length'] != float('inf')]
    
    balanced_spectral_gap = [m['connectivity']['spectral_gap'] for m in balanced_metrics]
    uniform_spectral_gap = [m['connectivity']['spectral_gap'] for m in uniform_metrics]
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
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
        idx = range(min(len(balanced_path_length), len(uniform_path_length)))
        plt.bar([i - 0.2 for i in idx], balanced_path_length[:len(idx)], width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
        plt.bar([i + 0.2 for i in idx], uniform_path_length[:len(idx)], width=0.4, label='Uniform', color='orange', alpha=0.7)
        plt.title(f'Average Path Length Comparison ({dataset_name})')
        plt.xlabel('Graph Index')
        plt.ylabel('Average Path Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "Insufficient path length data", horizontalalignment='center', verticalalignment='center')
    
    # Plot 3: Spectral gap
    plt.subplot(2, 2, 3)
    plt.bar([i - 0.2 for i in idx], balanced_spectral_gap, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in idx], uniform_spectral_gap, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title(f'Spectral Gap Comparison ({dataset_name})')
    plt.xlabel('Graph Index')
    plt.ylabel('Spectral Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Boxplot comparison
    plt.subplot(2, 2, 4)
    plt.boxplot([balanced_resistance, uniform_resistance], labels=['Balanced Cayley', 'Uniform'])
    plt.title(f'Effective Resistance Distribution ({dataset_name})')
    plt.ylabel('Effective Resistance')
    plt.grid(True, alpha=0.3)
    
    # Save visualization
    plt.tight_layout()
    vis_file = os.path.join(output_dir, f"{dataset_name.lower()}_oversquashing_comparison.png")
    plt.savefig(vis_file)
    plt.close()
    
    print(f"Visualization saved to {vis_file}")
    
    # Also create a summary visualization with means
    plt.figure(figsize=(10, 6))
    
    metrics = [
        ('Mean Effective Resistance', np.mean(balanced_resistance), np.mean(uniform_resistance)),
        ('Average Path Length', np.mean(balanced_path_length) if balanced_path_length else 0, 
                            np.mean(uniform_path_length) if uniform_path_length else 0),
        ('Spectral Gap', np.mean(balanced_spectral_gap), np.mean(uniform_spectral_gap))
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
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save summary visualization
    summary_file = os.path.join(output_dir, f"{dataset_name.lower()}_metrics_summary.png")
    plt.savefig(summary_file)
    plt.close()
    
    print(f"Summary visualization saved to {summary_file}")
    
    # Add visualization of edge weight distributions if available
    visualize_edge_weight_distributions(balanced_metrics, uniform_metrics, dataset_name, output_dir)


def visualize_edge_weight_distributions(balanced_metrics, uniform_metrics, dataset_name, output_dir):
    """
    Visualize the distribution of edge weights in both initialization methods.
    
    Args:
        balanced_metrics: List of metrics dicts for balanced Cayley approach
        uniform_metrics: List of metrics dicts for uniform approach
        dataset_name: Name of the dataset
        output_dir: Directory to save visualization
    """
    # Extract edge weights if available in metrics
    balanced_weights = []
    uniform_weights = []
    
    for i, (balanced, uniform) in enumerate(zip(balanced_metrics, uniform_metrics)):
        if 'edge_weights' in balanced and 'edge_weights' in uniform:
            balanced_weights.append(balanced['edge_weights'].flatten().cpu().numpy())
            uniform_weights.append(uniform['edge_weights'].flatten().cpu().numpy())
    
    if not balanced_weights or not uniform_weights:
        print("No edge weight data available for visualization")
        return
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Concatenate all weights
    balanced_all = np.concatenate(balanced_weights)
    uniform_all = np.concatenate(uniform_weights)
    
    # Plot 1: Histogram of all edge weights
    plt.subplot(2, 2, 1)
    plt.hist(balanced_all, bins=30, alpha=0.7, label='Balanced Cayley', color='blue')
    plt.hist(uniform_all, bins=30, alpha=0.7, label='Uniform', color='orange')
    plt.title(f'Edge Weight Distribution ({dataset_name})')
    plt.xlabel('Edge Weight')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Boxplot of weights by graph
    plt.subplot(2, 2, 2)
    plt.boxplot(balanced_weights + uniform_weights,
               labels=[f'B{i}' for i in range(len(balanced_weights))] + 
                      [f'U{i}' for i in range(len(uniform_weights))])
    plt.title(f'Edge Weight Variation by Graph ({dataset_name})')
    plt.xlabel('Graph (B=Balanced, U=Uniform)')
    plt.ylabel('Edge Weight')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot 3: Mean weights by graph
    plt.subplot(2, 2, 3)
    balanced_means = [np.mean(w) for w in balanced_weights]
    uniform_means = [np.mean(w) for w in uniform_weights]
    idx = range(min(len(balanced_means), len(uniform_means)))
    plt.bar([i - 0.2 for i in idx], balanced_means[:len(idx)], width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in idx], uniform_means[:len(idx)], width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title(f'Mean Edge Weight by Graph ({dataset_name})')
    plt.xlabel('Graph Index')
    plt.ylabel('Mean Edge Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Weight variance by graph
    plt.subplot(2, 2, 4)
    balanced_var = [np.var(w) for w in balanced_weights]
    uniform_var = [np.var(w) for w in uniform_weights]
    plt.bar([i - 0.2 for i in idx], balanced_var[:len(idx)], width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in idx], uniform_var[:len(idx)], width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title(f'Edge Weight Variance by Graph ({dataset_name})')
    plt.xlabel('Graph Index')
    plt.ylabel('Edge Weight Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save visualization
    plt.tight_layout()
    vis_file = os.path.join(output_dir, f"{dataset_name.lower()}_edge_weight_analysis.png")
    plt.savefig(vis_file)
    plt.close()
    
    print(f"Edge weight visualization saved to {vis_file}")

def analyze_all_datasets():
    """Run analysis on all three datasets"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"oversquashing_analysis/analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create JSON to collect all results
    all_results = []
    
    # Run MUTAG experiment
    print("\n=== Running MUTAG Experiment ===")
    mutag_results = run_experiment_with_oversquashing_analysis(
        dataset_name="MUTAG",
        k=3,
        hidden_dim=16,
        batch_size=4,
        num_epochs=15,
        seed=42,
        output_dir=output_dir
    )
    all_results.append(mutag_results)
    
    # Run PROTEINS experiment
    print("\n=== Running PROTEINS Experiment ===")
    proteins_results = run_experiment_with_oversquashing_analysis(
        dataset_name="PROTEINS",
        k=3,
        hidden_dim=16,
        batch_size=4,
        num_epochs=15,
        seed=42,
        output_dir=output_dir
    )
    all_results.append(proteins_results)
    
    # Run ENZYMES experiment
    print("\n=== Running ENZYMES Experiment ===")
    enzymes_results = run_experiment_with_oversquashing_analysis(
        dataset_name="ENZYMES",
        k=3,
        hidden_dim=16,
        batch_size=4,
        num_epochs=15,
        seed=42,
        output_dir=output_dir
    )
    all_results.append(enzymes_results)
    
    # Save combined results
    with open(os.path.join(output_dir, "all_dataset_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create combined visualization
    create_combined_visualization(
        [mutag_results, proteins_results, enzymes_results],
        output_dir
    )
    
    print(f"\nAll analyses complete. Results saved to {output_dir}")

def create_combined_visualization(results_list, output_dir):
    """Create a combined visualization of oversquashing metrics across datasets"""
    datasets = [r['dataset'] for r in results_list]
    
    # Collect metrics
    balanced_resistance_means = []
    uniform_resistance_means = []
    balanced_path_length_means = []
    uniform_path_length_means = []
    balanced_spectral_gap_means = []
    uniform_spectral_gap_means = []
    
    for result in results_list:
        # Calculate mean metrics for each dataset
        balanced_metrics = result['oversquashing_metrics']['balanced_cayley']
        uniform_metrics = result['oversquashing_metrics']['uniform']
        
        if balanced_metrics and uniform_metrics:
            balanced_resistance = [m['oversquashing']['mean_effective_resistance'] for m in balanced_metrics]
            uniform_resistance = [m['oversquashing']['mean_effective_resistance'] for m in uniform_metrics]
            
            balanced_path_length = [m['connectivity']['avg_path_length'] for m in balanced_metrics 
                                   if m['connectivity']['avg_path_length'] != float('inf')]
            uniform_path_length = [m['connectivity']['avg_path_length'] for m in uniform_metrics
                                  if m['connectivity']['avg_path_length'] != float('inf')]
            
            balanced_spectral_gap = [m['connectivity']['spectral_gap'] for m in balanced_metrics]
            uniform_spectral_gap = [m['connectivity']['spectral_gap'] for m in uniform_metrics]
            
            balanced_resistance_means.append(np.mean(balanced_resistance))
            uniform_resistance_means.append(np.mean(uniform_resistance))
            
            if balanced_path_length:
                balanced_path_length_means.append(np.mean(balanced_path_length))
            else:
                balanced_path_length_means.append(0)
                
            if uniform_path_length:
                uniform_path_length_means.append(np.mean(uniform_path_length))
            else:
                uniform_path_length_means.append(0)
                
            balanced_spectral_gap_means.append(np.mean(balanced_spectral_gap))
            uniform_spectral_gap_means.append(np.mean(uniform_spectral_gap))
    
    # Create cross-dataset visualization
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Effective resistance across datasets
    plt.subplot(2, 2, 1)
    x = range(len(datasets))
    plt.bar([i - 0.2 for i in x], balanced_resistance_means, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in x], uniform_resistance_means, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title('Mean Effective Resistance Across Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Mean Effective Resistance')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Path length across datasets
    plt.subplot(2, 2, 2)
    plt.bar([i - 0.2 for i in x], balanced_path_length_means, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in x], uniform_path_length_means, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title('Average Path Length Across Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Average Path Length')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Spectral gap across datasets
    plt.subplot(2, 2, 3)
    plt.bar([i - 0.2 for i in x], balanced_spectral_gap_means, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in x], uniform_spectral_gap_means, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title('Spectral Gap Across Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Spectral Gap')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Model performance
    plt.subplot(2, 2, 4)
    balanced_acc = [r['accuracy']['balanced_cayley'] for r in results_list]
    uniform_acc = [r['accuracy']['uniform'] for r in results_list]
    plt.bar([i - 0.2 for i in x], balanced_acc, width=0.4, label='Balanced Cayley', color='blue', alpha=0.7)
    plt.bar([i + 0.2 for i in x], uniform_acc, width=0.4, label='Uniform', color='orange', alpha=0.7)
    plt.title('Model Accuracy Across Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Test Accuracy')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save combined visualization
    plt.tight_layout()
    combined_file = os.path.join(output_dir, "combined_dataset_comparison.png")
    plt.savefig(combined_file)
    plt.close()
    
    print(f"Combined visualization saved to {combined_file}")
    
    # Create markdown summary
    with open(os.path.join(output_dir, "OVERSQUASHING_ANALYSIS.md"), 'w') as f:
        f.write("# Oversquashing Analysis Summary\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Mean Effective Resistance\n\n")
        f.write("| Dataset | Balanced Cayley | Uniform | Difference |\n")
        f.write("|---------|----------------|---------|------------|\n")
        for i, dataset in enumerate(datasets):
            diff = balanced_resistance_means[i] - uniform_resistance_means[i]
            f.write(f"| {dataset} | {balanced_resistance_means[i]:.4f} | {uniform_resistance_means[i]:.4f} | {diff:.4f} |\n")
        
        f.write("\n## Average Path Length\n\n")
        f.write("| Dataset | Balanced Cayley | Uniform | Difference |\n")
        f.write("|---------|----------------|---------|------------|\n")
        for i, dataset in enumerate(datasets):
            diff = balanced_path_length_means[i] - uniform_path_length_means[i]
            f.write(f"| {dataset} | {balanced_path_length_means[i]:.4f} | {uniform_path_length_means[i]:.4f} | {diff:.4f} |\n")
        
        f.write("\n## Spectral Gap\n\n")
        f.write("| Dataset | Balanced Cayley | Uniform | Difference |\n")
        f.write("|---------|----------------|---------|------------|\n")
        for i, dataset in enumerate(datasets):
            diff = balanced_spectral_gap_means[i] - uniform_spectral_gap_means[i]
            f.write(f"| {dataset} | {balanced_spectral_gap_means[i]:.4f} | {uniform_spectral_gap_means[i]:.4f} | {diff:.4f} |\n")
        
        f.write("\n## Model Accuracy\n\n")
        f.write("| Dataset | Balanced Cayley | Uniform | Difference |\n")
        f.write("|---------|----------------|---------|------------|\n")
        for i, dataset in enumerate(datasets):
            diff = balanced_acc[i] - uniform_acc[i]
            f.write(f"| {dataset} | {balanced_acc[i]:.4f} | {uniform_acc[i]:.4f} | {diff:.4f} |\n")
            
        f.write("\n## Interpretation\n\n")
        f.write("- **Lower effective resistance** indicates less oversquashing\n")
        f.write("- **Shorter average path length** indicates more efficient message passing\n")
        f.write("- **Larger spectral gap** indicates faster information mixing in the graph\n\n")
        
        f.write("## Conclusion\n\n")
        
        # Calculate overall comparison
        resistance_diff = np.mean(balanced_resistance_means) - np.mean(uniform_resistance_means)
        path_diff = np.mean(balanced_path_length_means) - np.mean(uniform_path_length_means)
        spectral_diff = np.mean(balanced_spectral_gap_means) - np.mean(uniform_spectral_gap_means)
        acc_diff = np.mean(balanced_acc) - np.mean(uniform_acc)
        
        if resistance_diff < 0:
            f.write("- The Balanced Cayley approach shows **less oversquashing** (lower effective resistance)\n")
        else:
            f.write("- The Uniform approach shows **less oversquashing** (lower effective resistance)\n")
            
        if path_diff < 0:
            f.write("- The Balanced Cayley approach has **shorter average path lengths**\n")
        else:
            f.write("- The Uniform approach has **shorter average path lengths**\n")
            
        if spectral_diff > 0:
            f.write("- The Balanced Cayley approach has **larger spectral gap** (faster information mixing)\n")
        else:
            f.write("- The Uniform approach has **larger spectral gap** (faster information mixing)\n")
            
        if acc_diff > 0:
            f.write("- The Balanced Cayley approach achieved **higher accuracy** overall\n")
        else:
            f.write("- The Uniform approach achieved **higher accuracy** overall\n")

def run_simplified_test():
    """Run a very simplified test focusing only on the oversquashing metrics"""
    print("Running simplified oversquashing test...")
    
    # Create output directory
    output_dir = "oversquashing_analysis/simplified_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load a small dataset
    dataset = TUDataset(root='/workspaces/IPR-MPNN/datasets', name="MUTAG")
    
    # Take just 5 samples
    test_dataset = dataset[:5]
    
    # Create a dataloader with batch size 1
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize a model
    model = MemorySaverIPRMPNNModel(
        input_dim=dataset.num_features,
        hidden_dim=16,
        output_dim=dataset.num_classes,
        edge_init_type='cayley',
        top_k=3
    ).to(device)
    
    # Enable oversquashing tracking
    print("Enabling oversquashing tracking")
    model.collect_oversquashing_metrics = True
    
    # Process each graph and collect metrics
    metrics = []
    
    print(f"Processing {len(test_loader)} graphs...")
    for idx, data in enumerate(test_loader):
        print(f"\nProcessing graph {idx}")
        
        # Move data to device
        data = data.to(device)
        
        # Forward pass to collect edge weights
        with torch.no_grad():
            # Process with model
            model(data)
            
            # Get edge weights
            print(f"Getting weights for graph {idx}")
            weights = model.get_final_edge_weights(0)
            
            if weights:
                print(f"  Got weights with keys: {weights.keys()}")
                
                # Get the original graph structure
                edge_index = data.edge_index
                
                # Get actual edge weights for analysis
                edge_weights = weights.get('edge_weights', None)
                
                if edge_weights is not None:
                    print(f"  Edge weights shape: {edge_weights.shape}")
                    
                    try:
                        # Calculate oversquashing metrics
                        oversquashing = compute_oversquashing_metric(edge_index, edge_weights)
                        connectivity = compute_graph_connectivity_metrics(edge_index, edge_weights)
                        
                        # Store metrics
                        metrics.append({
                            'graph_idx': idx,
                            'num_nodes': weights['num_nodes'],
                            'num_virtual_nodes': weights['num_virtual_nodes'],
                            'oversquashing': oversquashing,
                            'connectivity': connectivity
                        })
                        
                        print(f"  Successfully calculated metrics for graph {idx}")
                        print(f"  Mean Effective Resistance: {oversquashing['mean_effective_resistance']:.4f}")
                        print(f"  Avg Path Length: {connectivity['avg_path_length']:.4f}")
                    except Exception as e:
                        print(f"  Error calculating metrics: {str(e)}")
                else:
                    print("  No edge weights found in the weights dict")
            else:
                print(f"  No weights retrieved for graph {idx}")
    
    # Save metrics
    if metrics:
        print(f"\nSuccessfully collected metrics for {len(metrics)} graphs")
        results_file = os.path.join(output_dir, "simplified_test_results.json")
        
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
            
            json.dump(convert_to_native(metrics), f, indent=2)
        
        print(f"Results saved to {results_file}")
    else:
        print("No metrics collected")

if __name__ == "__main__":
    # Set anomaly detection to diagnose gradient issues
    torch.autograd.set_detect_anomaly(True)
    
    # Run simplified test first
    run_simplified_test()
    
    # Run experiments with shorter epochs for debugging
    # run_with_debug = True
    
    # if run_with_debug:
    #     # Run shorter version of MUTAG only for debugging
    #     print("\n=== Running MUTAG Experiment (Debug Mode) ===")
    #     mutag_results = run_experiment_with_oversquashing_analysis(
    #         dataset_name="MUTAG",
    #         k=3,
    #         hidden_dim=16,
    #         batch_size=4,
    #         num_epochs=5,  # Shorter run for debugging
    #         seed=42,
    #         output_dir="oversquashing_analysis/debug_run"
    #     )
    # else:
    #     # Run all experiments
    #     analyze_all_datasets()
