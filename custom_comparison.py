"""
Focused comparison script for uniform vs. Cayley initialization.
This script creates synthetic data and trains a smaller model to compare 
the two initialization approaches.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Batch
from torch_geometric.utils import erdos_renyi_graph
import torch.nn.functional as F
from datetime import datetime
import logging

from models.cayley_utils import cayley_initialize_edge_weight

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class SimpleHybridModel(torch.nn.Module):
    """
    A simplified hybrid model that uses either uniform or Cayley initialization
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_virtual_nodes, edge_init_type='uniform'):
        super(SimpleHybridModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_virtual_nodes = num_virtual_nodes
        self.edge_init_type = edge_init_type
        
        # Node embedding layers
        self.node_encoder = torch.nn.Linear(input_dim, hidden_dim)
        
        # Virtual node layers
        self.virtual_node_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Prediction head
        self.pred_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Get number of nodes in each graph
        unique_batch, counts = torch.unique(batch, return_counts=True)
        batch_sizes = counts.tolist()
        num_graphs = len(batch_sizes)
        
        # Encode nodes
        node_features = F.relu(self.node_encoder(x))
        
        # Create virtual nodes
        virtual_nodes = torch.zeros(num_graphs * self.num_virtual_nodes, self.hidden_dim, device=device)
        
        # Calculate weights between base nodes and virtual nodes
        all_weights = []
        offset = 0
        
        for graph_idx, num_nodes in enumerate(batch_sizes):
            if self.edge_init_type == 'cayley':
                # Use Cayley initialization
                graph_weights = cayley_initialize_edge_weight(
                    num_base_nodes=num_nodes,
                    num_virtual_nodes=self.num_virtual_nodes
                ).to(device)
            else:
                # Use uniform initialization
                graph_weights = torch.ones(num_nodes, self.num_virtual_nodes, device=device) / self.num_virtual_nodes
            
            all_weights.append(graph_weights)
            
            # Update virtual nodes using weighted aggregation
            for v_idx in range(self.num_virtual_nodes):
                # Extract weights for this virtual node
                weights = graph_weights[:, v_idx]
                
                # Calculate weighted feature for this virtual node
                node_indices = torch.where(batch == graph_idx)[0]
                weighted_features = node_features[node_indices] * weights.unsqueeze(1)
                
                # Aggregate features to virtual node
                aggregated = weighted_features.sum(dim=0)
                virtual_idx = graph_idx * self.num_virtual_nodes + v_idx
                virtual_nodes[virtual_idx] = aggregated
        
        # Process virtual nodes
        virtual_nodes = self.virtual_node_mlp(virtual_nodes)
        
        # Aggregate virtual nodes per graph
        graph_features = torch.zeros(num_graphs, self.hidden_dim, device=device)
        for graph_idx in range(num_graphs):
            start_idx = graph_idx * self.num_virtual_nodes
            end_idx = start_idx + self.num_virtual_nodes
            graph_features[graph_idx] = virtual_nodes[start_idx:end_idx].mean(dim=0)
        
        # Make predictions
        predictions = self.pred_mlp(graph_features)
        
        return predictions

def generate_synthetic_data(num_graphs=100, min_nodes=10, max_nodes=20, node_features=3, edge_prob=0.3):
    """Generate synthetic graph classification data"""
    dataset = []
    
    for i in range(num_graphs):
        # Random number of nodes
        num_nodes = np.random.randint(min_nodes, max_nodes + 1)
        
        # Generate random features
        x = torch.randn(num_nodes, node_features)
        
        # Generate random edges (Erdős–Rényi graph)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob)
        
        # Generate label (binary classification)
        # Use a simple rule: if more than half of the node features are positive on average, label as 1
        label = 1 if x.mean() > 0 else 0
        y = torch.tensor([label], dtype=torch.long)
        
        # Create graph data object
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
    
    return dataset

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.y.size(0)
    
    accuracy = 100. * correct / total
    return total_loss / len(train_loader), accuracy

def evaluate(model, loader, criterion):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss += criterion(out, data.y).item()
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.y.size(0)
    
    accuracy = 100. * correct / total
    return loss / len(loader), accuracy

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    num_epochs = 30
    batch_size = 32
    learning_rate = 0.001
    num_virtual_nodes = 8
    input_dim = 3
    hidden_dim = 64
    output_dim = 2  # Binary classification
    
    # Generate synthetic data
    logger.info("Generating synthetic graph dataset...")
    dataset = generate_synthetic_data(num_graphs=200, node_features=input_dim)
    
    # Split dataset
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    train_loader = [Batch.from_data_list(train_data[i:i+batch_size]) for i in range(0, len(train_data), batch_size)]
    val_loader = [Batch.from_data_list(val_data[i:i+batch_size]) for i in range(0, len(val_data), batch_size)]
    test_loader = [Batch.from_data_list(test_data[i:i+batch_size]) for i in range(0, len(test_data), batch_size)]
    
    logger.info(f"Dataset splits: Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create output directory
    output_dir = 'custom_comparison_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Train models with different initialization types
    initialization_types = ['uniform', 'cayley']
    results = {}
    
    for init_type in initialization_types:
        logger.info(f"Training model with {init_type} initialization...")
        
        # Create model
        model = SimpleHybridModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_virtual_nodes=num_virtual_nodes,
            edge_init_type=init_type
        ).to(device)
        
        # Loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion)
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Store results
        results[init_type] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch
        }
    
    # Plot results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Plot training and validation losses
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for init_type in initialization_types:
        plt.plot(results[init_type]['train_losses'], label=f"{init_type.capitalize()}")
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for init_type in initialization_types:
        plt.plot(results[init_type]['val_losses'], label=f"{init_type.capitalize()}")
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    for init_type in initialization_types:
        plt.plot(results[init_type]['train_accs'], label=f"{init_type.capitalize()}")
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    for init_type in initialization_types:
        plt.plot(results[init_type]['val_accs'], label=f"{init_type.capitalize()}")
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_curves_{timestamp}.png"))
    plt.close()
    
    # Plot test performance
    plt.figure(figsize=(10, 6))
    
    test_metrics = {
        'Test Accuracy': [results[init_type]['test_acc'] for init_type in initialization_types],
        'Test Loss': [results[init_type]['test_loss'] for init_type in initialization_types]
    }
    
    x = np.arange(len(initialization_types))
    width = 0.35
    
    plt.subplot(1, 2, 1)
    plt.bar(x, test_metrics['Test Accuracy'], width)
    plt.xticks(x, [init_type.capitalize() for init_type in initialization_types])
    plt.title('Test Accuracy')
    plt.ylabel('Accuracy (%)')
    
    plt.subplot(1, 2, 2)
    plt.bar(x, test_metrics['Test Loss'], width)
    plt.xticks(x, [init_type.capitalize() for init_type in initialization_types])
    plt.title('Test Loss')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"test_metrics_{timestamp}.png"))
    plt.close()
    
    # Generate comparison report
    with open(os.path.join(output_dir, f"comparison_report_{timestamp}.txt"), "w") as f:
        f.write("Comparison Report: Uniform vs. Cayley Initialization\n")
        f.write("==================================================\n\n")
        
        f.write("Test Performance:\n")
        for init_type in initialization_types:
            f.write(f"  {init_type.capitalize()}:\n")
            f.write(f"    Test Accuracy: {results[init_type]['test_acc']:.2f}%\n")
            f.write(f"    Test Loss: {results[init_type]['test_loss']:.4f}\n")
            f.write(f"    Best Validation Loss: {results[init_type]['best_val_loss']:.4f} (Epoch {results[init_type]['best_epoch']+1})\n\n")
        
        # Compare uniform vs cayley
        uniform_acc = results['uniform']['test_acc']
        cayley_acc = results['cayley']['test_acc']
        acc_diff = cayley_acc - uniform_acc
        
        uniform_loss = results['uniform']['test_loss']
        cayley_loss = results['cayley']['test_loss']
        loss_diff = uniform_loss - cayley_loss  # Lower loss is better
        
        f.write("Comparative Analysis:\n")
        f.write(f"  Accuracy Difference (Cayley - Uniform): {acc_diff:.2f}%\n")
        if acc_diff > 0:
            f.write("  The Cayley initialization achieved higher accuracy.\n")
        elif acc_diff < 0:
            f.write("  The Uniform initialization achieved higher accuracy.\n")
        else:
            f.write("  Both initializations achieved the same accuracy.\n")
        
        f.write(f"  Loss Difference (Uniform - Cayley): {loss_diff:.4f}\n")
        if loss_diff > 0:
            f.write("  The Cayley initialization achieved lower loss.\n")
        elif loss_diff < 0:
            f.write("  The Uniform initialization achieved lower loss.\n")
        else:
            f.write("  Both initializations achieved the same loss.\n")
        
        f.write("\nConclusions:\n")
        if cayley_acc > uniform_acc and cayley_loss < uniform_loss:
            f.write("  The Cayley initialization outperformed uniform initialization in both accuracy and loss.\n")
            f.write("  This suggests that the structured sparsity provided by Cayley graphs improves model performance.\n")
        elif uniform_acc > cayley_acc and uniform_loss < cayley_loss:
            f.write("  The Uniform initialization outperformed Cayley initialization in both accuracy and loss.\n")
            f.write("  This suggests that for this dataset, a dense connectivity pattern is more effective.\n")
        else:
            f.write("  The results are mixed, with one initialization performing better on some metrics.\n")
            if cayley_acc > uniform_acc:
                f.write("  The Cayley initialization achieved higher accuracy, which may be more important for classification tasks.\n")
            elif uniform_loss < cayley_loss:
                f.write("  The Uniform initialization achieved lower loss, indicating better overall fit to the data.\n")
    
    logger.info(f"Results saved to {output_dir}/ directory")

if __name__ == "__main__":
    main()
