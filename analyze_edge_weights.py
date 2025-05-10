"""
Script to analyze differences in how the model learns edge weights for different datasets.
This script will examine if the model is truly learning different edge weights for different datasets.
"""

import os
import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from custom_aligned_comparison import IPRMPNNModel, load_dataset

def analyze_edge_weights(dataset_name='ENZYMES', num_epochs=5):
    """
    Train a model briefly and check if edge weights are changing (being learned).
    """
    print(f"Analyzing learnable edge weights for {dataset_name} dataset...")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Load dataset
    data_loaders = load_dataset(dataset_name)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    num_features = data_loaders['num_features']
    num_classes = data_loaders['num_classes']
    
    # Create both uniform and cayley models
    uniform_model = IPRMPNNModel(
        input_dim=num_features, 
        hidden_dim=64, 
        output_dim=num_classes,
        edge_init_type='uniform'
    ).to(device)
    
    cayley_model = IPRMPNNModel(
        input_dim=num_features, 
        hidden_dim=64, 
        output_dim=num_classes,
        edge_init_type='cayley'
    ).to(device)
    
    # Initialize optimizers
    uniform_optimizer = torch.optim.Adam(uniform_model.parameters(), lr=0.01)
    cayley_optimizer = torch.optim.Adam(cayley_model.parameters(), lr=0.01)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Get a single batch to analyze before training
    for batch in data_loaders['train']:
        first_batch = batch.to(device)
        break
    
    # Do a forward pass to initialize edge weights
    print("\nInitializing models with first batch...")
    _ = uniform_model(first_batch)
    _ = cayley_model(first_batch)
    
    # Store initial weights for a single graph
    uniform_init_weights = {}
    cayley_init_weights = {}
    
    print("\nChecking initial edge weights for first few graphs:")
    for i, graph_id in enumerate(list(uniform_model.graph_edge_weights.keys())[:3]):
        if graph_id in uniform_model.graph_edge_weights and graph_id in cayley_model.graph_edge_weights:
            uniform_weights = uniform_model.graph_edge_weights[graph_id].detach().cpu().clone()
            cayley_weights = cayley_model.graph_edge_weights[graph_id].detach().cpu().clone()
            
            uniform_init_weights[graph_id] = uniform_weights
            cayley_init_weights[graph_id] = cayley_weights
            
            print(f"\nGraph {i+1} (id: {graph_id}):")
            print(f"  Uniform - shape: {uniform_weights.shape}, requires_grad: {uniform_model.graph_edge_weights[graph_id].requires_grad}")
            print(f"    Min: {uniform_weights.min().item():.6f}, Max: {uniform_weights.max().item():.6f}")
            
            # Count non-zero connections
            uniform_nonzero = (uniform_weights > 0).sum().item()
            uniform_total = uniform_weights.numel()
            print(f"    Nonzero: {uniform_nonzero}/{uniform_total} ({uniform_nonzero/uniform_total*100:.2f}%)")
            
            print(f"  Cayley - shape: {cayley_weights.shape}, requires_grad: {cayley_model.graph_edge_weights[graph_id].requires_grad}")
            print(f"    Min: {cayley_weights.min().item():.6f}, Max: {cayley_weights.max().item():.6f}")
            
            # Count non-zero connections
            cayley_nonzero = (cayley_weights > 0).sum().item()
            cayley_total = cayley_weights.numel()
            print(f"    Nonzero: {cayley_nonzero}/{cayley_total} ({cayley_nonzero/cayley_total*100:.2f}%)")
    
    # Train for a few epochs to see if weights change
    print(f"\nTraining for {num_epochs} epochs to check if edge weights change...")
    for epoch in range(num_epochs):
        # Train the uniform model
        uniform_model.train()
        for batch in data_loaders['train']:
            batch = batch.to(device)
            uniform_optimizer.zero_grad()
            out = uniform_model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            uniform_optimizer.step()
        
        # Train the cayley model
        cayley_model.train()
        for batch in data_loaders['train']:
            batch = batch.to(device)
            cayley_optimizer.zero_grad()
            out = cayley_model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            cayley_optimizer.step()
        
        print(f"  Epoch {epoch+1}/{num_epochs} complete")
    
    # Check final weights for the same graphs
    print("\nChecking final edge weights after training:")
    for i, graph_id in enumerate(list(uniform_init_weights.keys())):
        if graph_id in uniform_model.graph_edge_weights and graph_id in cayley_model.graph_edge_weights:
            uniform_final = uniform_model.graph_edge_weights[graph_id].detach().cpu()
            cayley_final = cayley_model.graph_edge_weights[graph_id].detach().cpu()
            
            # Calculate the magnitude of change
            uniform_change = torch.abs(uniform_final - uniform_init_weights[graph_id]).mean().item()
            cayley_change = torch.abs(cayley_final - cayley_init_weights[graph_id]).mean().item()
            
            print(f"\nGraph {i+1} (id: {graph_id}):")
            print(f"  Uniform - Avg absolute change: {uniform_change:.6f}")
            print(f"    Initial Min: {uniform_init_weights[graph_id].min().item():.6f}, Max: {uniform_init_weights[graph_id].max().item():.6f}")
            print(f"    Final Min: {uniform_final.min().item():.6f}, Max: {uniform_final.max().item():.6f}")
            
            # Count non-zero connections in final weights
            uniform_final_nonzero = (uniform_final > 0).sum().item()
            uniform_total = uniform_final.numel()
            print(f"    Final Nonzero: {uniform_final_nonzero}/{uniform_total} ({uniform_final_nonzero/uniform_total*100:.2f}%)")
            
            print(f"  Cayley - Avg absolute change: {cayley_change:.6f}")
            print(f"    Initial Min: {cayley_init_weights[graph_id].min().item():.6f}, Max: {cayley_init_weights[graph_id].max().item():.6f}")
            print(f"    Final Min: {cayley_final.min().item():.6f}, Max: {cayley_final.max().item():.6f}")
            
            # Count non-zero connections in final weights
            cayley_final_nonzero = (cayley_final > 0).sum().item()
            cayley_total = cayley_final.numel()
            print(f"    Final Nonzero: {cayley_final_nonzero}/{cayley_total} ({cayley_final_nonzero/cayley_total*100:.2f}%)")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # Analyze ENZYMES
    analyze_edge_weights('ENZYMES', num_epochs=3)
    
    # For comparison, analyze PROTEINS which showed better performance for Cayley
    analyze_edge_weights('PROTEINS', num_epochs=3)
