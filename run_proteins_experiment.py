"""
Script to run balanced IPR-MPNN model on the PROTEINS dataset.
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
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

from models.balanced_iprmpnn import BalancedIPRMPNNModel

def load_proteins_dataset():
    """
    Load the PROTEINS dataset with proper train/test split.
    """
    print("Loading PROTEINS dataset...")
    
    # Load the dataset
    dataset = TUDataset(root='/workspaces/IPR-MPNN/datasets', name='PROTEINS')
    
    # Get statistics
    num_graphs = len(dataset)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    
    # Calculate average number of nodes
    avg_nodes = sum(data.num_nodes for data in dataset) / num_graphs
    
    print(f"Dataset has {num_graphs} graphs with avg {avg_nodes:.2f} nodes per graph")
    
    # Split into train and test
    torch.manual_seed(42)
    train_idx = torch.randperm(len(dataset))[:int(0.8 * len(dataset))]
    test_idx = torch.tensor([i for i in range(len(dataset)) if i not in train_idx])
    
    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return {
        'train': train_loader,
        'test': test_loader,
        'num_features': num_features,
        'num_classes': num_classes
    }

def run_proteins_experiment(num_epochs=50, k=5, batch_size=32):
    """
    Run balanced model comparison on PROTEINS dataset.
    """
    print(f"\n==================================================")
    print(f"Dataset: PROTEINS, top-k: {k}, epochs: {num_epochs}")
    print(f"==================================================")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset
    data_loaders = load_proteins_dataset()
    
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
        correct = 0
        total = 0
        
        for data in data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
            total_loss += loss.item() * data.num_graphs
            
        train_acc = correct / total
        avg_loss = total_loss / len(data_loader.dataset)
        return avg_loss, train_acc
    
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
    
    # Training loop
    train_losses_balanced = []
    train_losses_uniform = []
    train_accs_balanced = []
    train_accs_uniform = []
    test_accs_balanced = []
    test_accs_uniform = []
    
    print("Training models...")
    for epoch in tqdm(range(num_epochs)):
        # Train balanced model
        balanced_loss, balanced_train_acc = train(balanced_model, balanced_optimizer, data_loaders['train'])
        train_losses_balanced.append(balanced_loss)
        train_accs_balanced.append(balanced_train_acc)
        
        # Train uniform model
        uniform_loss, uniform_train_acc = train(uniform_model, uniform_optimizer, data_loaders['train'])
        train_losses_uniform.append(uniform_loss)
        train_accs_uniform.append(uniform_train_acc)
        
        # Evaluate models
        balanced_acc = evaluate(balanced_model, data_loaders['test'])
        test_accs_balanced.append(balanced_acc)
        
        uniform_acc = evaluate(uniform_model, data_loaders['test'])
        test_accs_uniform.append(uniform_acc)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Balanced test acc: {balanced_acc:.4f}, Uniform test acc: {uniform_acc:.4f}")
    
    # Print final results
    print(f"\nFinal results after {num_epochs} epochs:")
    print(f"Balanced Cayley Test Accuracy: {test_accs_balanced[-1]:.4f}")
    print(f"Uniform Test Accuracy: {test_accs_uniform[-1]:.4f}")
    
    # Create results directory if it doesn't exist
    os.makedirs("balanced_comparison_results/proteins", exist_ok=True)
    
    # Plot and save training curves
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses_balanced, label='Balanced Cayley')
    plt.plot(train_losses_uniform, label='Uniform')
    plt.title('Training Loss (PROTEINS)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training accuracy
    plt.subplot(2, 2, 2)
    plt.plot(train_accs_balanced, label='Balanced Cayley')
    plt.plot(train_accs_uniform, label='Uniform')
    plt.title('Training Accuracy (PROTEINS)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot test accuracy
    plt.subplot(2, 2, 3)
    plt.plot(test_accs_balanced, label='Balanced Cayley')
    plt.plot(test_accs_uniform, label='Uniform')
    plt.title('Test Accuracy (PROTEINS)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot final test accuracy comparison
    plt.subplot(2, 2, 4)
    models = ['Balanced Cayley', 'Uniform']
    accuracies = [test_accs_balanced[-1], test_accs_uniform[-1]]
    plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e'])
    plt.title('Final Test Accuracy (PROTEINS)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"balanced_comparison_results/proteins/proteins_results_topk{k}.png")
    print(f"Saved results to balanced_comparison_results/proteins/proteins_results_topk{k}.png")
    
    return {
        'balanced_cayley': {
            'train_losses': train_losses_balanced,
            'train_accs': train_accs_balanced,
            'test_accs': test_accs_balanced,
            'final_accuracy': test_accs_balanced[-1]
        },
        'uniform': {
            'train_losses': train_losses_uniform,
            'train_accs': train_accs_uniform,
            'test_accs': test_accs_uniform,
            'final_accuracy': test_accs_uniform[-1]
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run balanced model on PROTEINS dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--topk', type=int, default=5, help='Number of connections per node (default: 5)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    
    args = parser.parse_args()
    
    run_proteins_experiment(num_epochs=args.epochs, k=args.topk, batch_size=args.batch_size)
