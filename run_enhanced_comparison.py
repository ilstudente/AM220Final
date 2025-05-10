"""
Test script to compare standard and enhanced IPR-MPNN models on the ENZYMES dataset.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from custom_aligned_comparison import IPRMPNNModel, load_dataset
from models.enhanced_iprmpnn import EnhancedIPRMPNNModel

def run_test_comparison(dataset_name='ENZYMES', num_epochs=20, top_k=5):
    """
    Run a quick comparison between standard and enhanced IPR-MPNN models.
    
    Args:
        dataset_name (str): Name of TU dataset to use
        num_epochs (int): Number of epochs to train
        top_k (int): Number of connections to keep per base node
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Running comparison on {dataset_name} dataset with top-{top_k} connectivity...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    data_loaders = load_dataset(dataset_name)
    
    # Create models
    models = {
        'Standard Cayley': IPRMPNNModel(
            input_dim=data_loaders['num_features'],
            hidden_dim=64,
            output_dim=data_loaders['num_classes'],
            edge_init_type='cayley',
            top_k=top_k
        ).to(device),
        
        'Enhanced Cayley': EnhancedIPRMPNNModel(
            input_dim=data_loaders['num_features'],
            hidden_dim=64,
            output_dim=data_loaders['num_classes'],
            edge_init_type='cayley',
            top_k=top_k
        ).to(device),
        
        'Uniform': IPRMPNNModel(
            input_dim=data_loaders['num_features'],
            hidden_dim=64,
            output_dim=data_loaders['num_classes'],
            edge_init_type='uniform',
            top_k=top_k
        ).to(device)
    }
    
    # Training functions
    criterion = torch.nn.CrossEntropyLoss()
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=0.01)
        for name, model in models.items()
    }
    
    # Track metrics
    train_losses = {name: [] for name in models}
    train_accs = {name: [] for name in models}
    val_losses = {name: [] for name in models}
    val_accs = {name: [] for name in models}
    
    # Training loop
    for epoch in range(num_epochs):
        # Train phase
        for name, model in models.items():
            model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch in data_loaders['train']:
                batch = batch.to(device)
                optimizers[name].zero_grad()
                
                out = model(batch)
                loss = criterion(out, batch.y)
                
                loss.backward()
                optimizers[name].step()
                
                epoch_loss += loss.item() * batch.num_graphs
                pred = out.argmax(dim=1)
                correct += int((pred == batch.y).sum())
                total += batch.num_graphs
            
            train_losses[name].append(epoch_loss / len(data_loaders['train'].dataset))
            train_accs[name].append(correct / total)
        
        # Validation phase
        for name, model in models.items():
            model.eval()
            epoch_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in data_loaders['val']:
                    batch = batch.to(device)
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    
                    epoch_loss += loss.item() * batch.num_graphs
                    pred = out.argmax(dim=1)
                    correct += int((pred == batch.y).sum())
                    total += batch.num_graphs
            
            val_losses[name].append(epoch_loss / len(data_loaders['val'].dataset))
            val_accs[name].append(correct / total)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            for name in models:
                print(f"  {name}: Train Loss={train_losses[name][-1]:.4f}, "
                      f"Train Acc={train_accs[name][-1]:.4f}, "
                      f"Val Loss={val_losses[name][-1]:.4f}, "
                      f"Val Acc={val_accs[name][-1]:.4f}")
    
    # Test phase
    test_metrics = {}
    for name, model in models.items():
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loaders['test']:
                batch = batch.to(device)
                out = model(batch)
                pred = out.argmax(dim=1)
                correct += int((pred == batch.y).sum())
                total += batch.num_graphs
        
        test_acc = correct / total
        test_metrics[name] = test_acc
        print(f"{name} Test Accuracy: {test_acc:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    # Training losses
    plt.subplot(2, 2, 1)
    for name, losses in train_losses.items():
        plt.plot(losses, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # Validation losses
    plt.subplot(2, 2, 2)
    for name, losses in val_losses.items():
        plt.plot(losses, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    
    # Training accuracies
    plt.subplot(2, 2, 3)
    for name, accs in train_accs.items():
        plt.plot(accs, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    # Validation accuracies
    plt.subplot(2, 2, 4)
    for name, accs in val_accs.items():
        plt.plot(accs, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    output_dir = f"enhanced_comparison_results/{dataset_name.lower()}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/training_curves_topk{top_k}.png")
    print(f"Training curves saved to {output_dir}/training_curves_topk{top_k}.png")
    
    # Plot test accuracies
    plt.figure(figsize=(8, 6))
    names = list(test_metrics.keys())
    accs = [test_metrics[name] for name in names]
    plt.bar(names, accs)
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy')
    plt.title(f'Test Accuracy Comparison on {dataset_name}')
    plt.ylim(0, 1)
    for i, acc in enumerate(accs):
        plt.text(i, acc + 0.02, f"{acc:.4f}", ha='center')
    plt.savefig(f"{output_dir}/test_accuracy_topk{top_k}.png")
    print(f"Test accuracy plot saved to {output_dir}/test_accuracy_topk{top_k}.png")
    
    return test_metrics

if __name__ == "__main__":
    # Test on ENZYMES dataset
    run_test_comparison(dataset_name='ENZYMES', num_epochs=20, top_k=5)
    
    # Test on MUTAG dataset
    run_test_comparison(dataset_name='MUTAG', num_epochs=20, top_k=5)
