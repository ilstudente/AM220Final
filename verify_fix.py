"""
Script to verify that edge weights are properly being learned with the fixed implementation.
"""

import torch
import numpy as np
from custom_aligned_comparison import IPRMPNNModel, load_dataset

def verify_learnable_weights():
    """Test if edge weights are actually being learned."""
    print("Testing learnable edge weights with fixed implementation...")
    
    # Load a small dataset
    data_loaders = load_dataset('MUTAG')
    
    # Get sample graph
    for batch in data_loaders['train']:
        sample_graph = batch
        break
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = IPRMPNNModel(
        input_dim=data_loaders['num_features'], 
        hidden_dim=64, 
        output_dim=data_loaders['num_classes'],
        edge_init_type='cayley'
    ).to(device)
    
    # Do a forward pass to initialize edge weights
    sample_graph = sample_graph.to(device)
    _ = model(sample_graph)
    
    # Save initial weights
    initial_weights = {}
    for graph_id, param in model.graph_edge_weights.items():
        initial_weights[graph_id] = param.detach().clone()
        print(f"Graph {graph_id}: Initial weight shape: {param.shape}, requires_grad: {param.requires_grad}")
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Run a few training steps
    print("\nTraining for 3 steps...")
    for step in range(3):
        optimizer.zero_grad()
        output = model(sample_graph)
        loss = criterion(output, sample_graph.y)
        loss.backward()
        optimizer.step()
        print(f"Step {step+1}: Loss: {loss.item():.4f}")
    
    # Check if weights changed
    print("\nChecking weight changes:")
    for graph_id, initial_weight in initial_weights.items():
        if graph_id in model.graph_edge_weights:
            current_weight = model.graph_edge_weights[graph_id].detach()
            weight_diff = torch.abs(current_weight - initial_weight).mean().item()
            print(f"Graph {graph_id}: Average weight change: {weight_diff:.6f}")
            
            if weight_diff > 0:
                print("✅ Edge weights ARE being learned!")
            else:
                print("❌ Edge weights are NOT being learned!")

if __name__ == "__main__":
    verify_learnable_weights()
