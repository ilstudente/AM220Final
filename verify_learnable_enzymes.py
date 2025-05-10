"""
Script to verify if edge weights are learnable in the IPR-MPNN model for ENZYMES dataset.
"""

import torch
from torch_geometric.datasets import TUDataset
from custom_aligned_comparison import IPRMPNNModel, load_dataset
import os

def verify_learnable_weights():
    print("Verifying learnable edge weights for ENZYMES dataset...")
    
    # Load ENZYMES dataset
    data_loaders = load_dataset('ENZYMES')
    
    # Get sample graph
    for batch in data_loaders['train']:
        sample_graph = batch
        break
    
    # Create model with both initialization types
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_features = data_loaders['num_features']
    num_classes = data_loaders['num_classes']
    
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
    
    # Verify parameters
    print("\nChecking Uniform Model Parameters:")
    total_params = 0
    edge_weight_params = 0
    for name, param in uniform_model.named_parameters():
        if "edge_weights_" in name:
            print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")
            edge_weight_params += 1
        total_params += 1
    
    print(f"  Total parameters: {total_params}")
    print(f"  Edge weight parameters: {edge_weight_params}")
    
    print("\nChecking Cayley Model Parameters:")
    total_params = 0
    edge_weight_params = 0
    for name, param in cayley_model.named_parameters():
        if "edge_weights_" in name:
            print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")
            edge_weight_params += 1
        total_params += 1
    
    print(f"  Total parameters: {total_params}")
    print(f"  Edge weight parameters: {edge_weight_params}")
    
    # Convert the batch to a dictionary format that the model expects
    input_data = {
        'x': sample_graph.x.to(device),
        'edge_index': sample_graph.edge_index.to(device),
        'batch': sample_graph.batch.to(device)
    }
    
    # Run a sample forward pass to create the edge weight parameters
    cayley_model(input_data)
    
    # Check if graph_edge_weights dictionary is populated
    print("\nVerifying edge weights dictionary:")
    print(f"  Number of graphs with edge weights: {len(cayley_model.graph_edge_weights)}")
    
    # Check optimizer includes the edge weights
    optimizer = torch.optim.Adam(cayley_model.parameters(), lr=0.01)
    
    print("\nVerifying optimizer includes edge weights:")
    edge_weight_in_optimizer = False
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            for name, model_param in cayley_model.named_parameters():
                if param is model_param and "edge_weights_" in name:
                    edge_weight_in_optimizer = True
                    print(f"  Edge weight parameter {name} is in optimizer")
    
    if not edge_weight_in_optimizer:
        print("  WARNING: No edge weight parameters found in optimizer!")
    
    print("\nLearnable edge weights verification complete!")

if __name__ == "__main__":
    verify_learnable_weights()
