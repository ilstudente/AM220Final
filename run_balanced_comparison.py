"""
Script to run comparison experiments with the balanced IPR-MPNN model.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from custom_aligned_comparison import load_dataset
from models.balanced_iprmpnn import BalancedIPRMPNNModel

def run_comparison(dataset_name='ENZYMES', num_epochs=100, k=5):
    """
    Run a comparison between balanced Cayley, standard Cayley, and uniform initializations.
    """
    print(f"Running comparison on {dataset_name} dataset with top-{k} connections...")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset
    data_loaders = load_dataset(dataset_name)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    num_features = data_loaders['num_features']
    num_classes = data_loaders['num_classes']
    
    # Create models with different initializations
    balanced_model = BalancedIPRMPNNModel(
        input_dim=num_features, 
        hidden_dim=64, 
        output_dim=num_classes,
        edge_init_type='cayley',
        top_k=k
    ).to(device)
    
    standard_cayley_model = BalancedIPRMPNNModel(
        input_dim=num_features, 
        hidden_dim=64, 
        output_dim=num_classes,
        edge_init_type='cayley',
        top_k=k
    ).to(device)
    
    uniform_model = BalancedIPRMPNNModel(
        input_dim=num_features, 
        hidden_dim=64, 
        output_dim=num_classes,
        edge_init_type='uniform',
        top_k=k
    ).to(device)
    
    # Initialize optimizers
    balanced_optimizer = torch.optim.Adam(balanced_model.parameters(), lr=0.01, weight_decay=5e-4)
    cayley_optimizer = torch.optim.Adam(standard_cayley_model.parameters(), lr=0.01, weight_decay=5e-4)
    uniform_optimizer = torch.optim.Adam(uniform_model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training and evaluation functions
    def train(model, optimizer, data_loader):
        model.train()
        total_loss = 0
        
        for data in data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            
        return total_loss / len(data_loader.dataset)
    
    def evaluate(model, data_loader):
        model.eval()
        correct = 0
        
        for data in data_loader:
            data = data.to(device)
            with torch.no_grad():
                out = model(data)
                pred = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
                
        return correct / len(data_loader.dataset)
    
    # Training loop
    print("Training models...")
    balanced_train_acc = []
    balanced_val_acc = []
    
    cayley_train_acc = []
    cayley_val_acc = []
    
    uniform_train_acc = []
    uniform_val_acc = []
    
    for epoch in tqdm(range(1, num_epochs + 1)):
        # Train all models
        balanced_loss = train(balanced_model, balanced_optimizer, data_loaders['train'])
        cayley_loss = train(standard_cayley_model, cayley_optimizer, data_loaders['train'])
        uniform_loss = train(uniform_model, uniform_optimizer, data_loaders['train'])
        
        # Evaluate
        balanced_tr_acc = evaluate(balanced_model, data_loaders['train'])
        balanced_v_acc = evaluate(balanced_model, data_loaders['val'])
        
        cayley_tr_acc = evaluate(standard_cayley_model, data_loaders['train'])
        cayley_v_acc = evaluate(standard_cayley_model, data_loaders['val'])
        
        uniform_tr_acc = evaluate(uniform_model, data_loaders['train'])
        uniform_v_acc = evaluate(uniform_model, data_loaders['val'])
        
        # Record metrics
        balanced_train_acc.append(balanced_tr_acc)
        balanced_val_acc.append(balanced_v_acc)
        
        cayley_train_acc.append(cayley_tr_acc)
        cayley_val_acc.append(cayley_v_acc)
        
        uniform_train_acc.append(uniform_tr_acc)
        uniform_val_acc.append(uniform_v_acc)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Balanced Cayley - Train: {balanced_tr_acc:.4f}, Val: {balanced_v_acc:.4f}")
            print(f"  Standard Cayley - Train: {cayley_tr_acc:.4f}, Val: {cayley_v_acc:.4f}")
            print(f"  Uniform - Train: {uniform_tr_acc:.4f}, Val: {uniform_v_acc:.4f}")
    
    # Final evaluation on test set
    balanced_test_acc = evaluate(balanced_model, data_loaders['test'])
    cayley_test_acc = evaluate(standard_cayley_model, data_loaders['test'])
    uniform_test_acc = evaluate(uniform_model, data_loaders['test'])
    
    print("\nFinal Test Accuracy:")
    print(f"  Balanced Cayley: {balanced_test_acc:.4f}")
    print(f"  Standard Cayley: {cayley_test_acc:.4f}")
    print(f"  Uniform: {uniform_test_acc:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(balanced_train_acc, label='Balanced Cayley')
    plt.plot(cayley_train_acc, label='Standard Cayley')
    plt.plot(uniform_train_acc, label='Uniform')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.legend()
    plt.title('Training Accuracy vs. Epoch')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(balanced_val_acc, label='Balanced Cayley')
    plt.plot(cayley_val_acc, label='Standard Cayley')
    plt.plot(uniform_val_acc, label='Uniform')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.title('Validation Accuracy vs. Epoch')
    plt.grid(True, alpha=0.3)
    
    # Create output directory if it doesn't exist
    os.makedirs(f'balanced_comparison_results/{dataset_name.lower()}', exist_ok=True)
    
    # Save the plots
    plt.tight_layout()
    plt.savefig(f'balanced_comparison_results/{dataset_name.lower()}/training_curves_topk{k}.png')
    
    # Plot test accuracy comparison
    plt.figure(figsize=(8, 6))
    models = ['Balanced Cayley', 'Standard Cayley', 'Uniform']
    test_accs = [balanced_test_acc, cayley_test_acc, uniform_test_acc]
    
    plt.bar(models, test_accs)
    plt.ylim(0.5, 1.0)  # Adjust as needed for the dataset
    plt.ylabel('Test Accuracy')
    plt.title(f'Test Accuracy Comparison on {dataset_name} (top-{k})')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add text on bars
    for i, acc in enumerate(test_accs):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.savefig(f'balanced_comparison_results/{dataset_name.lower()}/test_accuracy_topk{k}.png')
    
    # Return final test accuracies
    return {
        'balanced_cayley': balanced_test_acc,
        'standard_cayley': cayley_test_acc,
        'uniform': uniform_test_acc
    }

if __name__ == "__main__":
    # Make sure output directory exists
    os.makedirs('balanced_comparison_results', exist_ok=True)
    
    # Run comparisons on different datasets
    datasets = ['ENZYMES', 'MUTAG']
    k_values = [5]
    
    results = {}
    for dataset in datasets:
        results[dataset] = {}
        for k in k_values:
            print(f"\n{'=' * 50}")
            print(f"Dataset: {dataset}, top-k: {k}")
            print(f"{'=' * 50}")
            
            results[dataset][k] = run_comparison(dataset_name=dataset, num_epochs=100, k=k)
    
    # Print summary of results
    print("\n\nSummary of Test Accuracies:")
    print("=" * 60)
    print(f"{'Dataset':<10} | {'K':<3} | {'Balanced':<10} | {'Standard':<10} | {'Uniform':<10}")
    print("-" * 60)
    
    for dataset in results:
        for k in results[dataset]:
            bal_acc = results[dataset][k]['balanced_cayley']
            std_acc = results[dataset][k]['standard_cayley']
            uni_acc = results[dataset][k]['uniform']
            print(f"{dataset:<10} | {k:<3} | {bal_acc:.4f}      | {std_acc:.4f}      | {uni_acc:.4f}")
    
    print("=" * 60)
