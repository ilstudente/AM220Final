"""
Script to run a lightweight version of the balanced IPR-MPNN model on the PROTEINS dataset.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

from models.balanced_iprmpnn import BalancedIPRMPNNModel

def run_lightweight_proteins_experiment(num_epochs=20, k=5, batch_size=16, hidden_dim=32):
    """
    Run a lightweight balanced model comparison on PROTEINS dataset.
    """
    print(f"\n==================================================")
    print(f"Dataset: PROTEINS, top-k: {k}, epochs: {num_epochs}")
    print(f"Using smaller batch size ({batch_size}) and reduced model complexity")
    print(f"==================================================")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset
    print("Loading PROTEINS dataset...")
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
    perm = torch.randperm(len(dataset))
    train_idx = perm[:int(0.8 * len(dataset))]
    test_idx = perm[int(0.8 * len(dataset)):]
    
    train_dataset = dataset[train_idx.tolist()]
    test_dataset = dataset[test_idx.tolist()]
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create smaller models with reduced hidden dimensions
    balanced_model = BalancedIPRMPNNModel(
        input_dim=num_features, 
        hidden_dim=hidden_dim,  # Reduced from 64 to 32
        output_dim=num_classes,
        edge_init_type='cayley',
        top_k=k
    ).to(device)
    
    uniform_model = BalancedIPRMPNNModel(
        input_dim=num_features, 
        hidden_dim=hidden_dim,  # Reduced from 64 to 32
        output_dim=num_classes,
        edge_init_type='uniform',
        top_k=k
    ).to(device)
    
    # Print model size
    balanced_params = sum(p.numel() for p in balanced_model.parameters())
    print(f"Balanced model parameters: {balanced_params:,}")
    
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
        balanced_loss, balanced_train_acc = train(balanced_model, balanced_optimizer, train_loader)
        train_losses_balanced.append(balanced_loss)
        train_accs_balanced.append(balanced_train_acc)
        
        # Train uniform model
        uniform_loss, uniform_train_acc = train(uniform_model, uniform_optimizer, train_loader)
        train_losses_uniform.append(uniform_loss)
        train_accs_uniform.append(uniform_train_acc)
        
        # Evaluate models
        balanced_acc = evaluate(balanced_model, test_loader)
        test_accs_balanced.append(balanced_acc)
        
        uniform_acc = evaluate(uniform_model, test_loader)
        test_accs_uniform.append(uniform_acc)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Balanced test acc: {balanced_acc:.4f}, Uniform test acc: {uniform_acc:.4f}")
    
    # Print final results
    print(f"\nFinal results after {num_epochs} epochs:")
    print(f"Balanced Cayley Test Accuracy: {test_accs_balanced[-1]:.4f}")
    print(f"Uniform Test Accuracy: {test_accs_uniform[-1]:.4f}")
    
    # Create results directory if it doesn't exist
    os.makedirs("balanced_comparison_results/proteins", exist_ok=True)
    
    # Save the results to a text file
    with open(f"balanced_comparison_results/proteins/results_topk{k}.txt", "w") as f:
        f.write(f"PROTEINS Dataset Results (top-k={k}, epochs={num_epochs})\n")
        f.write("="*50 + "\n")
        f.write(f"Balanced Cayley Final Test Accuracy: {test_accs_balanced[-1]:.4f}\n")
        f.write(f"Uniform Final Test Accuracy: {test_accs_uniform[-1]:.4f}\n")
        f.write("\n")
        f.write("Epoch-by-epoch Test Accuracies:\n")
        for i in range(num_epochs):
            f.write(f"Epoch {i+1}: Balanced = {test_accs_balanced[i]:.4f}, Uniform = {test_accs_uniform[i]:.4f}\n")
    
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
            'final_accuracy': test_accs_balanced[-1],
            'test_accs': test_accs_balanced
        },
        'uniform': {
            'final_accuracy': test_accs_uniform[-1],
            'test_accs': test_accs_uniform
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run lightweight balanced model on PROTEINS dataset')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (default: 20)')
    parser.add_argument('--topk', type=int, default=5, help='Number of connections per node (default: 5)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension size (default: 32)')
    
    args = parser.parse_args()
    
    run_lightweight_proteins_experiment(
        num_epochs=args.epochs, 
        k=args.topk, 
        batch_size=args.batch_size, 
        hidden_dim=args.hidden_dim
    )
