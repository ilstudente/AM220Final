"""
Script to run a quick comparison experiment with the balanced IPR-MPNN model.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from custom_aligned_comparison import load_dataset
from models.balanced_iprmpnn import BalancedIPRMPNNModel

def run_quick_comparison(dataset_name='MUTAG', num_epochs=30, k=5):
    """
    Run a quick comparison between balanced Cayley and uniform initializations.
    """
    print(f"\n==================================================")
    print(f"Dataset: {dataset_name}, top-k: {k}, epochs: {num_epochs}")
    print(f"==================================================")
    
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
    
    uniform_model = BalancedIPRMPNNModel(
        input_dim=num_features, 
        hidden_dim=64, 
        output_dim=num_classes,
        edge_init_type='uniform',
        top_k=k
    ).to(device)
    
    # Initialize optimizers
    balanced_optimizer = torch.optim.Adam(balanced_model.parameters(), lr=0.01, weight_decay=5e-4)
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
    train_losses_balanced = []
    train_losses_uniform = []
    test_accuracies_balanced = []
    test_accuracies_uniform = []
    
    print("Training models...")
    for epoch in tqdm(range(num_epochs)):
        # Train balanced model
        balanced_loss = train(balanced_model, balanced_optimizer, data_loaders['train'])
        train_losses_balanced.append(balanced_loss)
        
        # Train uniform model
        uniform_loss = train(uniform_model, uniform_optimizer, data_loaders['train'])
        train_losses_uniform.append(uniform_loss)
        
        # Evaluate models
        balanced_acc = evaluate(balanced_model, data_loaders['test'])
        test_accuracies_balanced.append(balanced_acc)
        
        uniform_acc = evaluate(uniform_model, data_loaders['test'])
        test_accuracies_uniform.append(uniform_acc)
    
    # Print final results
    print(f"\nFinal results after {num_epochs} epochs:")
    print(f"Balanced Cayley Test Accuracy: {test_accuracies_balanced[-1]:.4f}")
    print(f"Uniform Test Accuracy: {test_accuracies_uniform[-1]:.4f}")
    
    # Create results directory if it doesn't exist
    os.makedirs(f"balanced_comparison_results/{dataset_name.lower()}", exist_ok=True)
    
    # Plot and save training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_balanced, label='Balanced Cayley')
    plt.plot(train_losses_uniform, label='Uniform')
    plt.title(f'Training Loss ({dataset_name}, top-{k})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies_balanced, label='Balanced Cayley')
    plt.plot(test_accuracies_uniform, label='Uniform')
    plt.title(f'Test Accuracy ({dataset_name}, top-{k})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"balanced_comparison_results/{dataset_name.lower()}/training_curves_topk{k}.png")
    print(f"Saved training curves to balanced_comparison_results/{dataset_name.lower()}/training_curves_topk{k}.png")
    
    # Create bar chart of final test accuracies
    plt.figure(figsize=(8, 6))
    models = ['Balanced Cayley', 'Uniform']
    accuracies = [test_accuracies_balanced[-1], test_accuracies_uniform[-1]]
    
    plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e'])
    plt.title(f'Final Test Accuracy ({dataset_name}, top-{k})')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"balanced_comparison_results/{dataset_name.lower()}/test_accuracy_topk{k}.png")
    print(f"Saved test accuracy comparison to balanced_comparison_results/{dataset_name.lower()}/test_accuracy_topk{k}.png")
    
    return {
        'balanced_cayley': {
            'train_losses': train_losses_balanced,
            'test_accuracies': test_accuracies_balanced,
            'final_accuracy': test_accuracies_balanced[-1]
        },
        'uniform': {
            'train_losses': train_losses_uniform,
            'test_accuracies': test_accuracies_uniform,
            'final_accuracy': test_accuracies_uniform[-1]
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run balanced model comparison')
    parser.add_argument('--dataset', type=str, default='MUTAG', choices=['MUTAG', 'ENZYMES', 'DD'],
                        help='Dataset to use (default: MUTAG)')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--topk', type=int, default=5,
                        help='Number of connections to keep per base node (default: 5)')
    
    args = parser.parse_args()
    
    run_quick_comparison(dataset_name=args.dataset, num_epochs=args.epochs, k=args.topk)
