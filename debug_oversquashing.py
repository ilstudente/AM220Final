"""
Debug script for oversquashing analysis
"""
import os
import torch
import numpy as np
import json
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from models.memory_saver_iprmpnn import MemorySaverIPRMPNNModel
from utils.oversquashing_metrics import compute_oversquashing_metric, compute_graph_connectivity_metrics

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def debug_oversquashing_tracking():
    """Debug the oversquashing tracking functionality"""
    print("Running oversquashing tracking debug...")
    
    # Create output directory
    output_dir = "oversquashing_analysis/debug_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load a small dataset
    try:
        dataset = TUDataset(root='/workspaces/IPR-MPNN/datasets', name="MUTAG")
        print(f"Dataset loaded: {len(dataset)} graphs")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Take just a single sample
    try:
        sample_data = dataset[0].to(device)
        print(f"Sample data: {sample_data}")
        print(f"Number of nodes: {sample_data.num_nodes}")
        print(f"Number of edges: {sample_data.edge_index.shape[1]}")
    except Exception as e:
        print(f"Error getting sample data: {e}")
        return
    
    # Initialize a model
    try:
        model = MemorySaverIPRMPNNModel(
            input_dim=dataset.num_features,
            hidden_dim=16,
            output_dim=dataset.num_classes,
            edge_init_type='cayley',
            top_k=3
        ).to(device)
        print("Model initialized")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Enable oversquashing tracking
    try:
        print("Enabling oversquashing tracking")
        model.collect_oversquashing_metrics = True
    except Exception as e:
        print(f"Error enabling tracking: {e}")
        return
    
    # Forward pass
    try:
        print("Running forward pass")
        with torch.no_grad():
            out = model(sample_data)
            print(f"Forward pass successful, output shape: {out.shape}")
    except Exception as e:
        print(f"Error in forward pass: {e}")
        return
    
    # Check for stored weights
    try:
        print("Getting edge weights")
        weights = model.get_final_edge_weights(0)
        if weights:
            print(f"Weights retrieved, keys: {weights.keys()}")
            if 'edge_weights' in weights:
                edge_weights = weights['edge_weights']
                print(f"Edge weights shape: {edge_weights.shape}")
            else:
                print("No edge_weights key in weights dict")
        else:
            print("No weights retrieved")
    except Exception as e:
        print(f"Error getting weights: {e}")
        return
    
    # Calculate metrics
    if weights and 'edge_weights' in weights:
        try:
            print("Calculating oversquashing metrics")
            edge_weights = weights['edge_weights']
            edge_index = sample_data.edge_index
            
            oversquashing = compute_oversquashing_metric(edge_index, edge_weights)
            print(f"Oversquashing metrics: {oversquashing}")
            
            connectivity = compute_graph_connectivity_metrics(edge_index, edge_weights)
            print(f"Connectivity metrics: {connectivity}")
            
            # Save metrics
            results = {
                'graph_idx': 0,
                'num_nodes': weights['num_nodes'],
                'num_virtual_nodes': weights['num_virtual_nodes'],
                'oversquashing': oversquashing,
                'connectivity': connectivity
            }
            
            results_file = os.path.join(output_dir, "debug_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to {results_file}")
        except Exception as e:
            print(f"Error calculating metrics: {e}")
    
if __name__ == "__main__":
    debug_oversquashing_tracking()
