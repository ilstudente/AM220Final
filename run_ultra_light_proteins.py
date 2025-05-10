"""
Script to run ultra-lightweight model on PROTEINS dataset with minimal memory usage.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import gc
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

from models.memory_saver_iprmpnn import MemorySaverIPRMPNNModel

def run_ultra_light_proteins(num_epochs=15, k=3, batch_size=8, hidden_dim=16):
    """
    Run ultra-lightweight model on PROTEINS dataset with aggressive memory optimization.
    """
    print(f"\n==================================================")
    print(f"Ultra-Lightweight PROTEINS Experiment")
    print(f"Epochs: {num_epochs}, Top-k: {k}, Batch size: {batch_size}, Hidden dim: {hidden_dim}")
    print(f"==================================================")
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset with minimal memory usage
    print("Loading PROTEINS dataset...")
    # Set follow_batch to None to reduce memory usage
    dataset = TUDataset(root='/workspaces/IPR-MPNN/datasets', name='PROTEINS')
    
    # Basic dataset stats
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    print(f"Dataset loaded: {len(dataset)} graphs, {num_features} features, {num_classes} classes")
    
    # Split into train and test (80/20)
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Use small batch size to reduce memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Train set: {len(train_dataset)} graphs, Test set: {len(test_dataset)} graphs")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models with memory-efficient settings
    balanced_model = MemorySaverIPRMPNNModel(
        input_dim=num_features, 
        hidden_dim=hidden_dim,  # Small hidden dimension
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
            # Process in smaller chunks
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
    
    # Training loop with memory tracking
    train_losses_balanced = []
    train_accs_balanced = []
    train_losses_uniform = []
    train_accs_uniform = []
    test_accs_balanced = []
    test_accs_uniform = []
    
    print("\nTraining models with memory optimization...")
    for epoch in range(num_epochs):
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Train balanced model
        balanced_loss, balanced_train_acc = train(balanced_model, balanced_optimizer, train_loader)
        train_losses_balanced.append(balanced_loss)
        train_accs_balanced.append(balanced_train_acc)
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Train uniform model
        uniform_loss, uniform_train_acc = train(uniform_model, uniform_optimizer, train_loader)
        train_losses_uniform.append(uniform_loss)
        train_accs_uniform.append(uniform_train_acc)
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Evaluate models (only every few epochs to save memory)
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            balanced_acc = evaluate(balanced_model, test_loader)
            test_accs_balanced.append(balanced_acc)
            
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            uniform_acc = evaluate(uniform_model, test_loader)
            test_accs_uniform.append(uniform_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                 f"B-Loss={balanced_loss:.4f}, B-TrainAcc={balanced_train_acc:.4f}, B-TestAcc={balanced_acc:.4f} | "
                 f"U-Loss={uniform_loss:.4f}, U-TrainAcc={uniform_train_acc:.4f}, U-TestAcc={uniform_acc:.4f}")
        else:
            # Just show training progress
            print(f"Epoch {epoch+1}/{num_epochs}: "
                 f"B-Loss={balanced_loss:.4f}, B-TrainAcc={balanced_train_acc:.4f} | "
                 f"U-Loss={uniform_loss:.4f}, U-TrainAcc={uniform_train_acc:.4f}")
            
            # Fill with None for non-evaluated epochs
            if epoch > 0:
                test_accs_balanced.append(None)
                test_accs_uniform.append(None)
    
    # Evaluate final models
    final_balanced_acc = evaluate(balanced_model, test_loader)
    final_uniform_acc = evaluate(uniform_model, test_loader)
    
    print(f"\nFinal Test Accuracy:")
    print(f"Balanced Cayley: {final_balanced_acc:.4f}")
    print(f"Uniform: {final_uniform_acc:.4f}")
    
    # Create results directory
    os.makedirs('ultra_light_results', exist_ok=True)
    
    # Save training curves
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_balanced, label='Balanced Cayley')
    plt.plot(train_losses_uniform, label='Uniform')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Plot only non-None accuracy values
    x_balanced = [i*5 for i in range(len([acc for acc in test_accs_balanced if acc is not None]))]
    y_balanced = [acc for acc in test_accs_balanced if acc is not None]
    
    x_uniform = [i*5 for i in range(len([acc for acc in test_accs_uniform if acc is not None]))]
    y_uniform = [acc for acc in test_accs_uniform if acc is not None]
    
    plt.plot(x_balanced, y_balanced, 'o-', label='Balanced Cayley')
    plt.plot(x_uniform, y_uniform, 'o-', label='Uniform')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ultra_light_results/proteins_training_curves.png')
    print("Saved training curves to ultra_light_results/proteins_training_curves.png")
    
    # Save final accuracy comparison
    plt.figure(figsize=(6, 4))
    plt.bar(['Balanced Cayley', 'Uniform'], [final_balanced_acc, final_uniform_acc])
    plt.title('Final Test Accuracy')
    plt.ylim(0, 1)
    
    # Add values on top of bars
    plt.text(0, final_balanced_acc + 0.02, f'{final_balanced_acc:.4f}', ha='center')
    plt.text(1, final_uniform_acc + 0.02, f'{final_uniform_acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('ultra_light_results/proteins_final_accuracy.png')
    print("Saved final accuracy to ultra_light_results/proteins_final_accuracy.png")
    
    return {
        'balanced_cayley': final_balanced_acc,
        'uniform': final_uniform_acc
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ultra-lightweight PROTEINS experiment')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--k', type=int, default=3, help='Top-k connections')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--dim', type=int, default=16, help='Hidden dimension')
    
    args = parser.parse_args()
    
    run_ultra_light_proteins(
        num_epochs=args.epochs, 
        k=args.k, 
        batch_size=args.batch, 
        hidden_dim=args.dim
    )
