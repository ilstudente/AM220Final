"""
Script to run multiple trials of experiments for statistical significance.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import gc
import json
import time
from datetime import datetime
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import warnings
from scipy import stats
warnings.filterwarnings('ignore')  # Suppress warnings

from models.memory_saver_iprmpnn import MemorySaverIPRMPNNModel

def run_trial(dataset_name, trial_idx, seed, k=3, batch_size=4, hidden_dim=16, num_epochs=20, 
            patience=5, convergence_delta=0.01):
    """
    Run a single trial of the experiment with a specific random seed.
    Includes early stopping for convergence and returns full learning curves.
    
    Args:
        dataset_name: Name of the dataset to use
        trial_idx: Index of the current trial
        seed: Random seed for reproducibility
        k: Number of top connections to keep
        batch_size: Training batch size
        hidden_dim: Size of hidden dimensions
        num_epochs: Maximum number of epochs to train
        patience: Number of epochs to wait for improvement before early stopping
        convergence_delta: Minimum improvement to be considered significant
        
    Returns:
        Dictionary with results including accuracies and learning curves
    """
    print(f"\nTrial {trial_idx+1} (Seed {seed})...")
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset with minimal memory usage
    dataset = TUDataset(root='/workspaces/IPR-MPNN/datasets', name=dataset_name)
    
    # Basic dataset stats
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    
    # Split into train and test (80/20) with specific seed
    torch.manual_seed(seed)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Use small batch size to reduce memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models with memory-efficient settings
    balanced_model = MemorySaverIPRMPNNModel(
        input_dim=num_features, 
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        edge_init_type='cayley',
        top_k=k
    ).to(device)
    
    uniform_model = MemorySaverIPRMPNNModel(
        input_dim=num_features, 
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        edge_init_type='uniform',
        top_k=k
    ).to(device)
    
    # Use smaller weight decay to reduce regularization computation
    balanced_optimizer = torch.optim.Adam(balanced_model.parameters(), lr=0.005, weight_decay=1e-5)
    uniform_optimizer = torch.optim.Adam(uniform_model.parameters(), lr=0.005, weight_decay=1e-5)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training function with memory optimization
    def train(model, optimizer, data_loader):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data in data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data)
            loss = criterion(out, data.y)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate training accuracy
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
            
            total_loss += loss.item() * data.num_graphs
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
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
                
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return correct / total
    
    # Training loop 
    balanced_train_losses = []
    balanced_train_accs = []
    balanced_test_accs = []
    uniform_train_losses = []
    uniform_train_accs = []
    uniform_test_accs = []
    
    # For early stopping
    best_balanced_acc = 0
    best_uniform_acc = 0
    balanced_patience_counter = 0
    uniform_patience_counter = 0
    balanced_converged = False
    uniform_converged = False
    
    for epoch in range(num_epochs):
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Train balanced model
        balanced_loss, balanced_train_acc = train(balanced_model, balanced_optimizer, train_loader)
        balanced_train_losses.append(balanced_loss)
        balanced_train_accs.append(balanced_train_acc)
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Train uniform model
        uniform_loss, uniform_train_acc = train(uniform_model, uniform_optimizer, train_loader)
        uniform_train_losses.append(uniform_loss)
        uniform_train_accs.append(uniform_train_acc)
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Evaluate models every few epochs
        if (epoch + 1) % 3 == 0 or epoch == 0 or epoch == num_epochs - 1:
            # Evaluate balanced model
            balanced_acc = evaluate(balanced_model, test_loader)
            balanced_test_accs.append(balanced_acc)
            
            # Check for early stopping for balanced model
            if balanced_acc > best_balanced_acc + convergence_delta:
                best_balanced_acc = balanced_acc
                balanced_patience_counter = 0
            else:
                balanced_patience_counter += 1
                
            if balanced_patience_counter >= patience:
                balanced_converged = True
                
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Evaluate uniform model
            uniform_acc = evaluate(uniform_model, test_loader)
            uniform_test_accs.append(uniform_acc)
            
            # Check for early stopping for uniform model
            if uniform_acc > best_uniform_acc + convergence_delta:
                best_uniform_acc = uniform_acc
                uniform_patience_counter = 0
            else:
                uniform_patience_counter += 1
                
            if uniform_patience_counter >= patience:
                uniform_converged = True
            
            print(f"  Epoch {epoch+1}/{num_epochs}: B-Acc={balanced_acc:.4f}, U-Acc={uniform_acc:.4f}")
        else:
            # Just append None for non-evaluated epochs to keep indexing consistent
            balanced_test_accs.append(None)
            uniform_test_accs.append(None)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: B-Loss={balanced_loss:.4f}, U-Loss={uniform_loss:.4f}")
            
        # Check if both models have converged
        if balanced_converged and uniform_converged:
            print(f"  Both models converged at epoch {epoch+1}. Early stopping.")
            break
    
    # Final evaluation
    final_balanced_acc = evaluate(balanced_model, test_loader)
    final_uniform_acc = evaluate(uniform_model, test_loader)
    
    print(f"  Results: Balanced={final_balanced_acc:.4f}, Uniform={final_uniform_acc:.4f}")
    
    # Create convergence visualization for this trial
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(balanced_train_losses, label='Balanced Cayley')
    plt.plot(uniform_train_losses, label='Uniform')
    plt.title(f'Trial {trial_idx+1} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot test accuracy for evaluated epochs
    plt.subplot(1, 2, 2)
    eval_epochs = [i for i, acc in enumerate(balanced_test_accs) if acc is not None]
    balanced_eval_accs = [acc for acc in balanced_test_accs if acc is not None]
    uniform_eval_accs = [acc for acc in uniform_test_accs if acc is not None]
    
    plt.plot(eval_epochs, balanced_eval_accs, 'o-', label='Balanced Cayley')
    plt.plot(eval_epochs, uniform_eval_accs, 'o-', label='Uniform')
    plt.title(f'Trial {trial_idx+1} Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Create results directory
    results_dir = f"statistical_results/{dataset_name.lower()}/trials"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/trial_{trial_idx+1}_convergence.png")
    plt.close()
    
    return {
        'balanced_cayley': final_balanced_acc,
        'uniform': final_uniform_acc,
        'seed': seed,
        'epochs_trained': epoch + 1,
        'balanced_converged': balanced_converged,
        'uniform_converged': uniform_converged,
        'learning_curves': {
            'balanced': {
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

def run_statistical_experiment(dataset_name='MUTAG', num_trials=10, 
                              k=3, batch_size=4, hidden_dim=16, num_epochs=20, 
                              patience=5, convergence_delta=0.01, start_seed=42):
    """
    Run multiple trials of the experiment and compute statistics.
    
    Args:
        dataset_name: Name of the dataset to use
        num_trials: Number of trials to run with different seeds
        k: Number of top connections to keep
        batch_size: Training batch size
        hidden_dim: Size of hidden dimensions
        num_epochs: Maximum number of epochs to train per trial
        patience: Number of epochs to wait for improvement before early stopping
        convergence_delta: Minimum improvement to be considered significant
        start_seed: Starting random seed
    """
    print(f"\n==================================================")
    print(f"Statistical Experiment: {dataset_name}")
    print(f"Trials: {num_trials}, Max Epochs: {num_epochs}, Top-k: {k}")
    print(f"Early stopping patience: {patience}, Delta: {convergence_delta}")
    print(f"==================================================")
    
    # Create results directory
    results_dir = f"statistical_results/{dataset_name.lower()}"
    os.makedirs(results_dir, exist_ok=True)
    
    trial_results = []
    balanced_accuracies = []
    uniform_accuracies = []
    
    # Run the specified number of trials
    for i in tqdm(range(num_trials), desc="Running trials"):
        # Use different seeds for each trial
        seed = start_seed + i
        result = run_trial(dataset_name, i, seed, k, batch_size, hidden_dim, num_epochs, patience, convergence_delta)
        
        trial_results.append(result)
        balanced_accuracies.append(result['balanced_cayley'])
        uniform_accuracies.append(result['uniform'])
        
        # Clear memory between trials
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Calculate statistics
    balanced_mean = np.mean(balanced_accuracies)
    balanced_std = np.std(balanced_accuracies)
    balanced_ci = stats.t.interval(0.95, len(balanced_accuracies)-1, 
                                   loc=balanced_mean, 
                                   scale=stats.sem(balanced_accuracies))
    
    uniform_mean = np.mean(uniform_accuracies)
    uniform_std = np.std(uniform_accuracies)
    uniform_ci = stats.t.interval(0.95, len(uniform_accuracies)-1, 
                                 loc=uniform_mean, 
                                 scale=stats.sem(uniform_accuracies))
    
    # Run t-test to check if difference is statistically significant
    t_stat, p_value = stats.ttest_rel(balanced_accuracies, uniform_accuracies)
    
    # Print summary
    print("\nSummary of Results:")
    print(f"Balanced Cayley: Mean={balanced_mean:.4f}, Std={balanced_std:.4f}")
    print(f"95% Confidence Interval: [{balanced_ci[0]:.4f}, {balanced_ci[1]:.4f}]")
    print(f"Uniform: Mean={uniform_mean:.4f}, Std={uniform_std:.4f}")
    print(f"95% Confidence Interval: [{uniform_ci[0]:.4f}, {uniform_ci[1]:.4f}]")
    print(f"T-test: t={t_stat:.4f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        print("The difference is statistically significant (p < 0.05)")
    else:
        print("The difference is not statistically significant (p >= 0.05)")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Individual trial results
    plt.subplot(1, 2, 1)
    x = np.arange(num_trials)
    plt.plot(x, balanced_accuracies, 'bo-', label='Balanced Cayley')
    plt.plot(x, uniform_accuracies, 'ro-', label='Uniform')
    plt.xlabel('Trial')
    plt.ylabel('Test Accuracy')
    plt.title(f'{dataset_name} - Individual Trial Results')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Box plot
    plt.subplot(1, 2, 2)
    data = [balanced_accuracies, uniform_accuracies]
    plt.boxplot(data, labels=['Balanced Cayley', 'Uniform'])
    plt.ylabel('Test Accuracy')
    plt.title(f'{dataset_name} - Accuracy Distribution')
    
    # Add means as points
    positions = [1, 2]
    means = [balanced_mean, uniform_mean]
    plt.plot(positions, means, 'rs', markersize=8)
    
    # Add p-value annotation
    plt.annotate(f'p-value: {p_value:.4f}', xy=(1.5, max(balanced_mean, uniform_mean) + 0.05),
                ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/statistical_comparison.png")
    print(f"Saved statistical visualization to {results_dir}/statistical_comparison.png")
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results = {
        'dataset': dataset_name,
        'num_trials': num_trials,
        'parameters': {
            'top_k': k,
            'batch_size': batch_size,
            'hidden_dim': hidden_dim,
            'max_epochs': num_epochs,
            'patience': patience,
            'convergence_delta': convergence_delta,
            'start_seed': start_seed
        },
        'balanced_cayley': {
            'mean': float(balanced_mean),
            'std': float(balanced_std),
            'ci_lower': float(balanced_ci[0]),
            'ci_upper': float(balanced_ci[1]),
            'individual_accuracies': [float(acc) for acc in balanced_accuracies]
        },
        'uniform': {
            'mean': float(uniform_mean),
            'std': float(uniform_std),
            'ci_lower': float(uniform_ci[0]),
            'ci_upper': float(uniform_ci[1]),
            'individual_accuracies': [float(acc) for acc in uniform_accuracies]
        },
        't_test': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        },
        'convergence': {
            'balanced_trials_converged': sum(1 for result in trial_results if result.get('balanced_converged', False)),
            'uniform_trials_converged': sum(1 for result in trial_results if result.get('uniform_converged', False)),
            'avg_epochs_to_train': sum(result.get('epochs_trained', num_epochs) for result in trial_results) / num_trials
        },
        'trial_results': trial_results
    }
    
    with open(f"{results_dir}/statistical_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary to markdown
    with open(f"{results_dir}/summary.md", 'w') as f:
        f.write(f"# Statistical Experiment: {dataset_name}\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Parameters\n")
        f.write(f"- Number of trials: {num_trials}\n")
        f.write(f"- Epochs per trial: {num_epochs}\n")
        f.write(f"- Top-k connections: {k}\n")
        f.write(f"- Batch size: {batch_size}\n")
        f.write(f"- Hidden dimensions: {hidden_dim}\n\n")
        
        f.write(f"## Results\n\n")
        f.write(f"### Balanced Cayley Model\n")
        f.write(f"- Mean accuracy: {balanced_mean:.4f}\n")
        f.write(f"- Standard deviation: {balanced_std:.4f}\n")
        f.write(f"- 95% Confidence interval: [{balanced_ci[0]:.4f}, {balanced_ci[1]:.4f}]\n\n")
        
        f.write(f"### Uniform Model\n")
        f.write(f"- Mean accuracy: {uniform_mean:.4f}\n")
        f.write(f"- Standard deviation: {uniform_std:.4f}\n")
        f.write(f"- 95% Confidence interval: [{uniform_ci[0]:.4f}, {uniform_ci[1]:.4f}]\n\n")
        
        f.write(f"### Statistical Significance\n")
        f.write(f"- T-statistic: {t_stat:.4f}\n")
        f.write(f"- P-value: {p_value:.4f}\n")
        if p_value < 0.05:
            f.write(f"- The difference is statistically significant (p < 0.05)\n\n")
        else:
            f.write(f"- The difference is not statistically significant (p >= 0.05)\n\n")
        
        f.write(f"### Individual Trial Results\n\n")
        f.write("| Trial | Balanced Cayley | Uniform |\n")
        f.write("|-------|----------------|--------|\n")
        for i in range(num_trials):
            f.write(f"| {i+1} | {balanced_accuracies[i]:.4f} | {uniform_accuracies[i]:.4f} |\n")
    
    print(f"Saved detailed results to {results_dir}/")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run statistical experiments')
    parser.add_argument('--dataset', type=str, default='MUTAG', choices=['MUTAG', 'ENZYMES', 'PROTEINS'],
                        help='Dataset to use (default: MUTAG)')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials to run')
    parser.add_argument('--epochs', type=int, default=20, help='Maximum number of epochs per trial')
    parser.add_argument('--k', type=int, default=3, help='Top-k connections')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--dim', type=int, default=16, help='Hidden dimension')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--delta', type=float, default=0.01, help='Improvement threshold for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Starting seed')
    
    args = parser.parse_args()
    
    # Record start time
    start_time = time.time()
    
    run_statistical_experiment(
        dataset_name=args.dataset, 
        num_trials=args.trials,
        k=args.k,
        batch_size=args.batch,
        hidden_dim=args.dim,
        num_epochs=args.epochs,
        patience=args.patience,
        convergence_delta=args.delta,
        start_seed=args.seed
    )
    
    # Calculate and print elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total experiment time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
