"""
Custom implementation for IPR-MPNN model with Cayley-aligned virtual nodes.
This is a simplified version that focuses on comparing uniform vs Cayley initialization.
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from datetime import datetime

from models.cayley_utils import cayley_initialize_edge_weight, calculate_optimal_virtual_nodes, get_cayley_n
from models.oversquashing_metrics import evaluate_with_virtual_nodes, plot_oversquashing_metrics

class IPRMPNNModel(torch.nn.Module):
    """
    IPR-MPNN model with learnable edge weights between base nodes and virtual nodes.
    Weights are initialized based on Cayley graph or uniform pattern and then optimized during training.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, max_virtual_nodes=None, edge_init_type='uniform', top_k=None):
        super(IPRMPNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_virtual_nodes = max_virtual_nodes
        self.edge_init_type = edge_init_type
        self.top_k = top_k  # If specified, will keep only top-k connections per virtual node
        
        # Node embedding layers
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Learnable affinity parameter for edge weights between base and virtual nodes
        # We'll initialize this in the forward pass to accommodate variable-sized graphs
        self.affinity_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Virtual node MLP
        self.virtual_node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Prediction head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Dictionary to store the learnable edge weights for each graph
        # Each entry will be a parameter of shape [num_base_nodes, num_virtual_nodes]
        self.graph_edge_weights = {}
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Embed nodes
        x = self.node_embedding(x)
        
        # Initial graph convolution
        x = F.relu(self.conv1(x, edge_index))
        
        # Get graph information
        num_graphs = batch.max().item() + 1
        device = x.device
        
        # Process each graph separately as each will have its own virtual node structure
        # Create a list to store the feature vectors for each graph
        graph_features = []
        
        for graph_idx in range(num_graphs):
            # Get nodes for this graph
            graph_mask = (batch == graph_idx)
            num_nodes = graph_mask.sum().item()
            graph_x = x[graph_mask]
            
            # Generate a unique identifier for this graph
            # In a real implementation, you'd use a more stable identifier
            graph_id = f"{num_nodes}_{graph_idx}"
            
            # For each graph, calculate the optimal number of virtual nodes based on Cayley structure
            if graph_id not in self.graph_edge_weights:
                # First time seeing this graph, initialize edge weights
                
                if self.edge_init_type == 'cayley':
                    # Calculate optimal number of virtual nodes for this specific graph
                    num_virtual_nodes, cayley_n = calculate_optimal_virtual_nodes(
                        num_base_nodes=num_nodes, 
                        verbose=False
                    )
                    
                    # Use Cayley graph initialization with the graph-specific n parameter
                    init_weights = cayley_initialize_edge_weight(
                        num_base_nodes=num_nodes, 
                        num_virtual_nodes=num_virtual_nodes,
                        cayley_n=cayley_n,
                        verbose=False
                    ).to(device)
                else:
                    # Use uniform initialization but with same number of virtual nodes for fair comparison
                    num_virtual_nodes, _ = calculate_optimal_virtual_nodes(
                        num_base_nodes=num_nodes, 
                        verbose=False
                    )
                        
                    # Use uniform initialization
                    init_weights = torch.ones(num_nodes, num_virtual_nodes, device=device) / num_virtual_nodes
                
                # Create learnable parameter starting from the initialization
                self.graph_edge_weights[graph_id] = nn.Parameter(init_weights.clone()).to(device)
                
                # Register the parameter so it's included in optimizer
                self.register_parameter(f"edge_weights_{graph_id}", self.graph_edge_weights[graph_id])
            
            # Get the learnable edge weights for this graph
            edge_weights = self.graph_edge_weights[graph_id]
            num_virtual_nodes = edge_weights.size(1)
            
            # Apply learnable transformation to the edge weights
            # This allows the model to learn which connections are more important
            
            # Compute affinity scores between base node features and virtual nodes
            affinity_features = self.affinity_mlp(graph_x)  # [num_nodes, hidden_dim]
            
            # Create virtual node embeddings (initially random)
            virtual_node_embeddings = torch.randn(num_virtual_nodes, self.hidden_dim, device=device)
            
            # Compute attention scores between base nodes and virtual nodes
            # Reshape for broadcasting: [num_nodes, 1, hidden_dim] and [1, num_virtual_nodes, hidden_dim]
            base_features = affinity_features.unsqueeze(1)  # [num_nodes, 1, hidden_dim]
            virtual_features = virtual_node_embeddings.unsqueeze(0)  # [1, num_virtual_nodes, hidden_dim]
            
            # Compute dot product attention scores
            attention_scores = torch.sum(base_features * virtual_features, dim=2)  # [num_nodes, num_virtual_nodes]
            
            # Combine initial edge weights with learned attention scores
            edge_weights = edge_weights * (1.0 + attention_scores)
            
            # If using top-k connectivity, apply pruning
            if self.top_k is not None and self.top_k < num_virtual_nodes:
                # Get the top-k indices per base node
                _, top_indices = torch.topk(edge_weights, k=self.top_k, dim=1)
                
                # Create a mask for the top-k connections
                mask = torch.zeros_like(edge_weights)
                for i in range(num_nodes):
                    mask[i, top_indices[i]] = 1.0
                
                # Apply the mask
                edge_weights = edge_weights * mask
            
            # Normalize weights to sum to 1 for each base node
            edge_weights = F.softmax(edge_weights, dim=1)
            
            # Create virtual nodes for this graph
            virtual_nodes = torch.zeros(num_virtual_nodes, self.hidden_dim, device=device)
            
            # Update virtual nodes based on the learned weights
            for v_idx in range(num_virtual_nodes):
                # Get weights for this virtual node
                weights = edge_weights[:, v_idx].reshape(-1, 1)
                
                # Weighted aggregation from base nodes to virtual node
                aggregated = (graph_x * weights).sum(dim=0)
                
                # Update the virtual node
                virtual_nodes[v_idx] = aggregated
            
            # Apply MLP to virtual nodes
            virtual_nodes = self.virtual_node_mlp(virtual_nodes)
            
            # Pool virtual nodes for this graph
            graph_feature = virtual_nodes.mean(dim=0)
            graph_features.append(graph_feature)
        
        # Stack graph features
        graph_features = torch.stack(graph_features, dim=0)
        
        # Final prediction
        out = self.mlp(graph_features)
        
        return out

def load_dataset(dataset_name='MUTAG'):
    """Load a TU dataset and prepare dataloaders.
    
    Args:
        dataset_name (str): Name of the TU dataset to load (e.g., 'MUTAG', 'PROTEINS', 'ENZYMES')
        
    Returns:
        dict: Dictionary containing the dataloaders and dataset info
    """
    print(f"Loading {dataset_name} dataset...")
    dataset = TUDataset(root='datasets/TUDataset', name=dataset_name)
    
    # Convert node features to float tensors
    for graph in dataset:
        if hasattr(graph, 'x') and graph.x is not None:
            graph.x = graph.x.float()
        else:
            # If node features don't exist, create simple one-hot encoding of node degrees
            degrees = torch.zeros(graph.num_nodes, dtype=torch.long)
            for i in range(graph.edge_index.size(1)):
                degrees[graph.edge_index[0, i]] += 1
            max_degree = max(degrees).item()
            one_hot = torch.zeros(graph.num_nodes, max_degree + 1)
            for i, deg in enumerate(degrees):
                one_hot[i, deg] = 1.0
            graph.x = one_hot
    
    # Calculate the average number of nodes per graph
    total_nodes = sum(data.num_nodes for data in dataset)
    avg_nodes = total_nodes / len(dataset)
    print(f"Dataset has {len(dataset)} graphs with avg {avg_nodes:.2f} nodes per graph")
    
    # Split dataset into train, validation, test
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size+val_size]
    test_dataset = dataset[train_size+val_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Return loaders and dataset info
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'num_features': dataset.num_features if hasattr(dataset, 'num_features') else max_degree + 1,
        'num_classes': dataset.num_classes,
        'avg_nodes': avg_nodes,
        'name': dataset_name
    }

def train_model(model, data_loaders, device, num_epochs=50):
    """Train and evaluate the model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Metrics storage
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data in data_loaders['train']:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.num_graphs
            pred = output.argmax(dim=1)
            train_correct += (pred == data.y).sum().item()
            train_total += data.num_graphs
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data in data_loaders['val']:
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data.y)
                
                val_loss += loss.item() * data.num_graphs
                pred = output.argmax(dim=1)
                val_correct += (pred == data.y).sum().item()
                val_total += data.num_graphs
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    
    # Evaluate test accuracy
    with torch.no_grad():
        for data in data_loaders['test']:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            test_correct += (pred == data.y).sum().item()
            test_total += data.num_graphs
    
    # Compute oversquashing metrics with virtual nodes
    oversquashing_metrics = evaluate_with_virtual_nodes(data_loaders['test'], model, device)
    
    test_acc = test_correct / test_total
    print(f'Test Accuracy: {test_acc:.4f}')
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'oversquashing_metrics': oversquashing_metrics
    }

def run_comparison(num_epochs=50, output_dir='custom_comparison_results', dataset_name='MUTAG'):
    """Run comparison between uniform and Cayley initialization.
    
    Args:
        num_epochs (int): Number of epochs to train for
        output_dir (str): Directory to save results to
        dataset_name (str): Name of TU dataset to use (e.g., 'MUTAG', 'PROTEINS', 'ENZYMES')
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    data_loaders = load_dataset(dataset_name)
    avg_nodes = data_loaders['avg_nodes']
    print(f"Average nodes per graph: {avg_nodes:.2f}")
    
    # For reference, calculate the average virtual node count
    avg_virtual_nodes, avg_cayley_n = calculate_optimal_virtual_nodes(int(avg_nodes))
    print(f"Reference: For average of {avg_nodes:.2f} base nodes, would use {avg_virtual_nodes} virtual nodes with Cayley n={avg_cayley_n}")
    print(f"But we'll be calculating optimal virtual nodes for each graph individually instead")
    
    # Create models
    uniform_model = IPRMPNNModel(
        input_dim=data_loaders['num_features'],
        hidden_dim=64,
        output_dim=data_loaders['num_classes'],
        max_virtual_nodes=None,
        edge_init_type='uniform',
        top_k=5  # Keep top-5 connections per virtual node after learning
    )
    
    # For the Cayley model, we will calculate optimal number of virtual nodes for each graph
    # dynamically in the forward pass
    cayley_model = IPRMPNNModel(
        input_dim=data_loaders['num_features'],
        hidden_dim=64,
        output_dim=data_loaders['num_classes'],
        max_virtual_nodes=None,
        edge_init_type='cayley',
        top_k=5  # Keep top-5 connections per virtual node after learning
    )
    
    # Train both models
    print("Training uniform initialization model...")
    uniform_metrics = train_model(uniform_model, data_loaders, device, num_epochs)
    
    print("\nTraining Cayley initialization model...")
    cayley_metrics = train_model(cayley_model, data_loaders, device, num_epochs)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comparison plots
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(uniform_metrics['train_losses'], label='Uniform')
    plt.plot(cayley_metrics['train_losses'], label='Cayley')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation loss
    plt.subplot(2, 2, 2)
    plt.plot(uniform_metrics['val_losses'], label='Uniform')
    plt.plot(cayley_metrics['val_losses'], label='Cayley')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training accuracy
    plt.subplot(2, 2, 3)
    plt.plot(uniform_metrics['train_accs'], label='Uniform')
    plt.plot(cayley_metrics['train_accs'], label='Cayley')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(2, 2, 4)
    plt.plot(uniform_metrics['val_accs'], label='Uniform')
    plt.plot(cayley_metrics['val_accs'], label='Cayley')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_curves_{timestamp}.png"))
    plt.close()
    
    # Plot test accuracies
    plt.figure(figsize=(8, 6))
    test_metrics = {
        'Test Accuracy': {
            'Uniform': uniform_metrics['test_acc'],
            'Cayley': cayley_metrics['test_acc']
        }
    }
    
    x = np.arange(len(test_metrics))
    width = 0.35
    
    plt.bar(x - width/2, [test_metrics[k]['Uniform'] for k in test_metrics], width, label='Uniform')
    plt.bar(x + width/2, [test_metrics[k]['Cayley'] for k in test_metrics], width, label='Cayley')
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Test Metrics: Uniform vs. Cayley Initialization')
    plt.xticks(x, list(test_metrics.keys()))
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"test_metrics_{timestamp}.png"))
    plt.close()
    
    # Plot oversquashing metrics
    plot_oversquashing_metrics(
        uniform_metrics['oversquashing_metrics'],
        cayley_metrics['oversquashing_metrics'],
        output_dir,
        timestamp
    )
    
    # Generate report
    with open(os.path.join(output_dir, f"comparison_report_{timestamp}.txt"), 'w') as f:
        f.write("Comparison Report: Uniform vs. Cayley Initialization\n")
        f.write("=====================================================\n\n")
        
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Average nodes per graph: {avg_nodes:.2f}\n")
        f.write(f"Reference - Cayley-aligned virtual nodes: {avg_virtual_nodes}\n")
        f.write(f"Reference - Cayley graph parameter n: {avg_cayley_n}\n\n")
        
        f.write("Test Metrics:\n")
        for key, values in test_metrics.items():
            f.write(f"  {key}:\n")
            f.write(f"    Uniform: {values['Uniform']:.4f}\n")
            f.write(f"    Cayley:  {values['Cayley']:.4f}\n")
            
            # Calculate improvement percentage
            improvement = (values['Cayley'] - values['Uniform']) / values['Uniform'] * 100
            f.write(f"    Improvement: {improvement:.2f}%\n\n")
        
        f.write("Final Training Loss:\n")
        f.write(f"  Uniform: {uniform_metrics['train_losses'][-1]:.4f}\n")
        f.write(f"  Cayley:  {cayley_metrics['train_losses'][-1]:.4f}\n\n")
        
        f.write("Final Validation Loss:\n")
        f.write(f"  Uniform: {uniform_metrics['val_losses'][-1]:.4f}\n")
        f.write(f"  Cayley:  {cayley_metrics['val_losses'][-1]:.4f}\n\n")
        
        f.write("Oversquashing Metrics:\n")
        for key in uniform_metrics['oversquashing_metrics']:
            if key.startswith('avg_'):
                metric_name = key.replace('avg_', '').replace('_', ' ').title()
                uniform_value = uniform_metrics['oversquashing_metrics'][key]
                cayley_value = cayley_metrics['oversquashing_metrics'][key]
                
                f.write(f"  {metric_name}:\n")
                f.write(f"    Uniform: {uniform_value:.4f}\n")
                f.write(f"    Cayley:  {cayley_value:.4f}\n")
                
                # For Cheeger constant, higher is better
                if 'cheeger' in key:
                    improvement = (cayley_value - uniform_value) / max(0.0001, uniform_value) * 100
                    better = "better" if cayley_value > uniform_value else "worse"
                # For other metrics, lower is better
                else:
                    improvement = (uniform_value - cayley_value) / max(0.0001, uniform_value) * 100
                    better = "better" if cayley_value < uniform_value else "worse"
                    
                f.write(f"    Cayley is {abs(improvement):.2f}% {better}\n\n")
        
        # Add connectivity statistics to the report
        f.write("Connectivity Statistics (After Learning):\n")
        for key in ['nonzero_ratio', 'avg_connections_per_base', 'avg_connections_per_virtual']:
            if key in uniform_metrics['oversquashing_metrics'] and key in cayley_metrics['oversquashing_metrics']:
                metric_name = key.replace('_', ' ').title()
                uniform_value = uniform_metrics['oversquashing_metrics'][key]
                cayley_value = cayley_metrics['oversquashing_metrics'][key]
                
                f.write(f"  {metric_name}:\n")
                f.write(f"    Uniform: {uniform_value:.2f}\n")
                f.write(f"    Cayley:  {cayley_value:.2f}\n\n")
        
        f.write("Analysis:\n")
        f.write("  The Cayley initialization uses a mathematical structure based on the Cayley graph\n")
        f.write("  to connect base nodes to virtual nodes. This creates a sparse but structured connectivity\n")
        f.write("  pattern, which may help the model propagate information more effectively across\n")
        f.write("  distant parts of the graph.\n\n")
        
        f.write("  In contrast, the uniform initialization connects each base node to every virtual node\n")
        f.write("  with equal weights, which may lead to over-smoothing or less effective message passing.\n\n")
        
        if cayley_metrics['test_acc'] > uniform_metrics['test_acc']:
            f.write("  The Cayley initialization achieved higher test accuracy, suggesting that the structured\n")
            f.write("  sparse connectivity pattern is beneficial for the MUTAG classification task.\n")
        else:
            f.write("  The Uniform initialization achieved higher test accuracy. This could be because\n")
            f.write("  the MUTAG dataset's graph structure doesn't benefit from the specific connectivity\n")
            f.write("  pattern created by the Cayley graph initialization.\n")
    
    print(f"Comparison complete! Results saved to {output_dir}/")
    
    return uniform_metrics, cayley_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare uniform and Cayley initialization for IPR-MPNN')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--output-dir', type=str, default='custom_comparison_results', help='Output directory')
    parser.add_argument('--dataset', type=str, default='MUTAG', help='TU dataset to use (e.g., MUTAG, PROTEINS, ENZYMES)')
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dataset-specific output directory
    output_dir = os.path.join(args.output_dir, args.dataset.lower())
    os.makedirs(output_dir, exist_ok=True)
    
    # Run comparison
    run_comparison(num_epochs=args.epochs, output_dir=output_dir, dataset_name=args.dataset)
