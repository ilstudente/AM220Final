"""
Script to analyze and visualize edge weight learning in the balanced IPR-MPNN model.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import copy

from custom_aligned_comparison import load_dataset
from models.balanced_iprmpnn import BalancedIPRMPNNModel

def visualize_edge_weight_learning(dataset_name='MUTAG', num_epochs=20, k=5):
    """
    Analyze how edge weights change during training for both balanced Cayley and uniform models.
    """
    print(f"\nAnalyzing edge weight learning for {dataset_name} dataset...")
    
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
    
    # Get a batch to analyze
    for batch in data_loaders['train']:
        first_batch = batch.to(device)
        if first_batch.num_graphs >= 3:  # Make sure we have at least 3 graphs in the batch
            break
    
    # Record edge weights during training
    def record_edge_weights(model, batch):
        # Do a forward pass to register graphs
        model(batch)
        
        # Save a copy of edge weights for the first 3 graphs in the batch
        edge_weights = {}
        for i in range(min(3, batch.num_graphs)):
            graph_mask = (batch.batch == i)
            num_nodes = graph_mask.sum().item()
            graph_id = f"{num_nodes}_{i}"
            
            if graph_id in model.graph_edge_weights:
                weights = model.graph_edge_weights[graph_id].detach().cpu().clone()
                edge_weights[graph_id] = weights
                
        return edge_weights
    
    # Record initial weights
    print("Recording initial edge weights...")
    init_balanced_weights = record_edge_weights(balanced_model, first_batch)
    init_uniform_weights = record_edge_weights(uniform_model, first_batch)
    
    # Store weights over time
    balanced_weights_over_time = {graph_id: [init_balanced_weights[graph_id]] for graph_id in init_balanced_weights}
    uniform_weights_over_time = {graph_id: [init_uniform_weights[graph_id]] for graph_id in init_uniform_weights}
    
    # Training function
    def train(model, optimizer, data_loader, weights_over_time):
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
            
        # Record edge weights
        curr_weights = record_edge_weights(model, first_batch)
        for graph_id in curr_weights:
            if graph_id in weights_over_time:
                weights_over_time[graph_id].append(curr_weights[graph_id])
                
        return total_loss / len(data_loader.dataset)
    
    # Training loop
    print("Training models and recording edge weights...")
    for epoch in tqdm(range(num_epochs)):
        # Train balanced model
        balanced_loss = train(balanced_model, balanced_optimizer, data_loaders['train'], balanced_weights_over_time)
        
        # Train uniform model
        uniform_loss = train(uniform_model, uniform_optimizer, data_loaders['train'], uniform_weights_over_time)
    
    # Create results directory
    os.makedirs('edge_weight_analysis', exist_ok=True)
    
    # Analyze and visualize edge weight changes for a couple of sample graphs
    for graph_id in balanced_weights_over_time:
        balanced_weights = balanced_weights_over_time[graph_id]
        uniform_weights = uniform_weights_over_time[graph_id]
        
        num_nodes, num_virtual = balanced_weights[0].shape
        
        # Calculate weight changes over time (mean absolute change)
        balanced_changes = []
        uniform_changes = []
        
        for i in range(1, len(balanced_weights)):
            balanced_change = torch.abs(balanced_weights[i] - balanced_weights[i-1]).mean().item()
            uniform_change = torch.abs(uniform_weights[i] - uniform_weights[i-1]).mean().item()
            
            balanced_changes.append(balanced_change)
            uniform_changes.append(uniform_change)
        
        # Calculate cumulative change from initialization
        balanced_cumulative = []
        uniform_cumulative = []
        
        for i in range(1, len(balanced_weights)):
            balanced_cum = torch.abs(balanced_weights[i] - balanced_weights[0]).mean().item()
            uniform_cum = torch.abs(uniform_weights[i] - uniform_weights[0]).mean().item()
            
            balanced_cumulative.append(balanced_cum)
            uniform_cumulative.append(uniform_cum)
        
        # Calculate weight entropy (measure of how distributed weights are)
        balanced_entropy = []
        uniform_entropy = []
        
        for i in range(len(balanced_weights)):
            # For each base node, calculate entropy of its connection distribution
            node_balanced_entropy = []
            node_uniform_entropy = []
            
            for n in range(num_nodes):
                b_weights = balanced_weights[i][n]
                u_weights = uniform_weights[i][n]
                
                # Only consider non-zero weights for entropy calculation
                b_nonzero = b_weights[b_weights > 0]
                u_nonzero = u_weights[u_weights > 0]
                
                if len(b_nonzero) > 0:
                    b_entropy = -(b_nonzero * torch.log(b_nonzero + 1e-10)).sum().item()
                    node_balanced_entropy.append(b_entropy)
                
                if len(u_nonzero) > 0:
                    u_entropy = -(u_nonzero * torch.log(u_nonzero + 1e-10)).sum().item()
                    node_uniform_entropy.append(u_entropy)
            
            balanced_entropy.append(np.mean(node_balanced_entropy))
            uniform_entropy.append(np.mean(node_uniform_entropy))
        
        # Plot weight change metrics
        plt.figure(figsize=(15, 10))
        
        # Plot per-epoch weight changes
        plt.subplot(2, 2, 1)
        plt.plot(balanced_changes, label='Balanced Cayley')
        plt.plot(uniform_changes, label='Uniform')
        plt.title(f'Mean Weight Change per Epoch (Graph {graph_id})')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Change')
        plt.legend()
        
        # Plot cumulative changes
        plt.subplot(2, 2, 2)
        plt.plot(balanced_cumulative, label='Balanced Cayley')
        plt.plot(uniform_cumulative, label='Uniform')
        plt.title(f'Cumulative Weight Change from Initialization (Graph {graph_id})')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Change')
        plt.legend()
        
        # Plot entropy over time
        plt.subplot(2, 2, 3)
        plt.plot(balanced_entropy, label='Balanced Cayley')
        plt.plot(uniform_entropy, label='Uniform')
        plt.title(f'Weight Distribution Entropy (Graph {graph_id})')
        plt.xlabel('Epoch')
        plt.ylabel('Entropy')
        plt.legend()
        
        # Visualize final weight distributions for a sample node
        sample_node = min(3, num_nodes - 1)
        plt.subplot(2, 2, 4)
        
        x = np.arange(num_virtual)
        width = 0.35
        
        balanced_final = balanced_weights[-1][sample_node].numpy()
        uniform_final = uniform_weights[-1][sample_node].numpy()
        
        plt.bar(x - width/2, balanced_final, width, label='Balanced Cayley')
        plt.bar(x + width/2, uniform_final, width, label='Uniform')
        plt.title(f'Final Weight Distribution for Node {sample_node} (Graph {graph_id})')
        plt.xlabel('Virtual Node Index')
        plt.ylabel('Weight')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'edge_weight_analysis/weight_learning_{dataset_name}_{graph_id}.png')
    
    print(f"Analysis complete. Visualizations saved to edge_weight_analysis/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze edge weight learning')
    parser.add_argument('--dataset', type=str, default='MUTAG', choices=['MUTAG', 'ENZYMES', 'DD'],
                        help='Dataset to use (default: MUTAG)')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--topk', type=int, default=5,
                        help='Number of connections to keep per base node (default: 5)')
    
    args = parser.parse_args()
    
    visualize_edge_weight_learning(dataset_name=args.dataset, num_epochs=args.epochs, k=args.topk)
