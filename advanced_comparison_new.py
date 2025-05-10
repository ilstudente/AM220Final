"""
Advanced comparison of uniform vs. Cayley initialization in IPR-MPNN.

This script:
1. Creates two versions of the MUTAG dataset config - one with uniform and one with Cayley initialization
2. Trains IPR-MPNN models with both initializations
3. Compares the performance metrics and generates visualizations
"""

import os
import sys
import json
import time
import yaml
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict, deque
import torch

# For utility functions
from models.cayley_utils import cayley_initialize_edge_weight 

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_cayley_graph(n):
    """
    Simple implementation of Cayley graph generation.
    """
    generators = np.array([
        [[1, 1], [0, 1]],
        [[1, n-1], [0, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [n-1, 1]]])
    ind = 1

    queue = deque([np.array([[1, 0], [0, 1]])])
    nodes = {(1, 0, 0, 1): 0}

    senders = []
    receivers = []

    while queue:
        x = queue.popleft()
        x_flat = (x[0][0], x[0][1], x[1][0], x[1][1])
        ind_x = nodes[x_flat]
        for i in range(4):
            try:
                tx = np.matmul(x, generators[i])
                tx = np.mod(tx, n)
                tx_flat = (tx[0][0], tx[0][1], tx[1][0], tx[1][1])
                if tx_flat not in nodes:
                    nodes[tx_flat] = ind
                    ind += 1
                    queue.append(tx)
                ind_tx = nodes[tx_flat]

                senders.append(ind_x)
                receivers.append(ind_tx)
            except Exception as e:
                print(f"Error in Cayley graph generation: {e}")
            
    return torch.tensor([senders, receivers]), len(nodes)

def cayley_init_weights(num_base_nodes, num_virtual_nodes, n=2):
    """Initialize edge weights using Cayley graph"""
    # Generate Cayley graph
    edge_index, num_nodes = simple_cayley_graph(n)
    
    # Create edge weight matrix
    edge_weights = torch.zeros(num_base_nodes, num_virtual_nodes)
    
    # Assign weights based on graph structure
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        
        # Check if the edge connects a base node to a virtual node
        if src < num_base_nodes and dst >= num_base_nodes and dst < num_base_nodes + num_virtual_nodes:
            virtual_idx = dst - num_base_nodes
            edge_weights[src, virtual_idx] = 1.0
        elif dst < num_base_nodes and src >= num_base_nodes and src < num_base_nodes + num_virtual_nodes:
            virtual_idx = src - num_base_nodes
            edge_weights[dst, virtual_idx] = 1.0
    
    # Ensure each base node has at least one connection
    row_sums = edge_weights.sum(dim=1)
    zero_rows = torch.where(row_sums == 0)[0]
    
    if len(zero_rows) > 0:
        print(f"{len(zero_rows)} base nodes have no connections. Adding random connections.")
        # Add random connections for nodes with zero connections
        for row in zero_rows:
            virtual_idx = torch.randint(0, num_virtual_nodes, (1,))
            edge_weights[row, virtual_idx] = 1.0
    
    return edge_weights

def uniform_init_weights(num_base_nodes, num_virtual_nodes):
    """Initialize edge weights uniformly"""
    # Create a uniform distribution of weights
    edge_weights = torch.ones(num_base_nodes, num_virtual_nodes) / num_virtual_nodes
    return edge_weights

def create_modified_config(edge_init_type, output_path, epochs=100, num_runs=5):
    """Create a modified config file with the specified edge initialization type."""
    # Load the original config
    with open('configs/tudatasets/mutag.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Add edge_init_type and adjust other parameters
    config['edge_init_type'] = edge_init_type
    config['wandb'] = {
        'use_wandb': False,  # Disable wandb for now
        'project': 'TUDatasets-IPR-MPNN-Compare',
        'name': f"MUTAG_{edge_init_type}_init",
        'entity': 'mls-stuttgart'  # Use your own entity or remove this line
    }
    config['max_epoch'] = epochs
    config['num_runs'] = num_runs
    config['patience'] = 30
    
    # Save the modified config
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created modified config at {output_path} with {edge_init_type} initialization")
    return output_path

def run_experiment(config_path, log_dir):
    """Run an experiment using the specified config file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "run.log")
    
    # Run the experiment using subprocess to capture all output
    cmd = [sys.executable, 'run.py', '--cfg', config_path]
    logger.info(f"Running experiment with command: {' '.join(cmd)}")
    
    with open(log_file, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    logger.info(f"Experiment completed. Log saved to {log_file}")
    return log_file

def parse_wandb_logs(log_file):
    """Parse the log file to extract wandb logs."""
    metrics = defaultdict(list)
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if "val_loss" in line and "train_loss" in line:
                # Try to extract the metrics
                try:
                    # The line format might be like: {"train_loss": 0.1, "val_loss": 0.2, ...}
                    start_idx = line.find('{')
                    end_idx = line.rfind('}')
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = line[start_idx:end_idx+1]
                        data = json.loads(json_str)
                        
                        for key, value in data.items():
                            metrics[key].append(value)
                except:
                    pass
    except Exception as e:
        logger.error(f"Error parsing log file: {e}")
    
    return metrics

def extract_final_metrics(log_file):
    """Extract the final metrics from a log file."""
    metrics = {}
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Look for best validation metric
        if "Best val metric:" in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "Best val metric:" in line:
                    val_metric = float(line.split(':')[-1].strip())
                    metrics["best_val_metric"] = val_metric
                
                if "test metric:" in line:
                    test_metric = float(line.split(':')[-1].strip())
                    metrics["test_metric"] = test_metric
                    
                if "dirichlet energy:" in line:
                    dirichlet_energy = float(line.split(':')[-1].strip())
                    metrics["dirichlet_energy"] = dirichlet_energy
    except Exception as e:
        logger.error(f"Error extracting final metrics: {e}")
    
    return metrics

def extract_results_from_logs(uniform_log_file, cayley_log_file):
    """Extract and compare results from both log files."""
    uniform_metrics = parse_wandb_logs(uniform_log_file)
    cayley_metrics = parse_wandb_logs(cayley_log_file)
    
    # If parsing failed, try to read the whole log and look for the final metrics
    if not uniform_metrics or not cayley_metrics:
        logger.warning("Failed to parse logs as wandb format. Trying to extract final metrics...")
        
        uniform_metrics = extract_final_metrics(uniform_log_file)
        cayley_metrics = extract_final_metrics(cayley_log_file)
    
    return uniform_metrics, cayley_metrics

def plot_comparison(uniform_metrics, cayley_metrics, output_dir):
    """Generate comparison plots for the two initialization methods."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Check if we have time series data
    has_timeseries = (
        'train_loss' in uniform_metrics and 
        'val_loss' in uniform_metrics and 
        'train_loss' in cayley_metrics and 
        'val_loss' in cayley_metrics and
        isinstance(uniform_metrics['train_loss'], list) and
        len(uniform_metrics['train_loss']) > 1
    )
    
    if has_timeseries:
        # Plot training curves
        plt.figure(figsize=(15, 10))
        
        # Plot training loss
        plt.subplot(2, 2, 1)
        plt.plot(uniform_metrics['train_loss'], label='Uniform')
        plt.plot(cayley_metrics['train_loss'], label='Cayley')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot validation loss
        plt.subplot(2, 2, 2)
        plt.plot(uniform_metrics['val_loss'], label='Uniform')
        plt.plot(cayley_metrics['val_loss'], label='Cayley')
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot training metrics if available
        if 'train_metric_acc' in uniform_metrics and 'train_metric_acc' in cayley_metrics:
            plt.subplot(2, 2, 3)
            plt.plot(uniform_metrics['train_metric_acc'], label='Uniform')
            plt.plot(cayley_metrics['train_metric_acc'], label='Cayley')
            plt.title('Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
        
        # Plot validation metrics if available
        if 'val_metric_acc' in uniform_metrics and 'val_metric_acc' in cayley_metrics:
            plt.subplot(2, 2, 4)
            plt.plot(uniform_metrics['val_metric_acc'], label='Uniform')
            plt.plot(cayley_metrics['val_metric_acc'], label='Cayley')
            plt.title('Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"training_curves_{timestamp}.png"))
        plt.close()
        
        logger.info(f"Training curves saved to {output_dir}/training_curves_{timestamp}.png")
    
    # Create a bar chart comparing the final metrics
    final_metrics = {}
    for key in ['best_val_metric', 'test_metric', 'dirichlet_energy']:
        if key in uniform_metrics and key in cayley_metrics:
            if isinstance(uniform_metrics[key], list):
                final_metrics[key] = {
                    'Uniform': uniform_metrics[key][-1],
                    'Cayley': cayley_metrics[key][-1]
                }
            else:
                final_metrics[key] = {
                    'Uniform': uniform_metrics[key],
                    'Cayley': cayley_metrics[key]
                }
    
    if final_metrics:
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(final_metrics))
        width = 0.35
        
        plt.bar(x - width/2, [final_metrics[k]['Uniform'] for k in final_metrics], width, label='Uniform')
        plt.bar(x + width/2, [final_metrics[k]['Cayley'] for k in final_metrics], width, label='Cayley')
        
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Comparison of Final Metrics: Uniform vs. Cayley Initialization')
        plt.xticks(x, list(final_metrics.keys()))
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"final_metrics_{timestamp}.png"))
        plt.close()
        
        logger.info(f"Final metrics chart saved to {output_dir}/final_metrics_{timestamp}.png")
    
    # Save the metrics for future reference
    with open(os.path.join(output_dir, f"uniform_metrics_{timestamp}.json"), 'w') as f:
        json.dump(uniform_metrics, f, indent=2)
    
    with open(os.path.join(output_dir, f"cayley_metrics_{timestamp}.json"), 'w') as f:
        json.dump(cayley_metrics, f, indent=2)
    
    # Generate a text report
    with open(os.path.join(output_dir, f"comparison_report_{timestamp}.txt"), 'w') as f:
        f.write("Comparison Report: Uniform vs. Cayley Initialization\n")
        f.write("=====================================================\n\n")
        
        if final_metrics:
            f.write("Final Metrics:\n")
            for key, values in final_metrics.items():
                f.write(f"  {key}:\n")
                f.write(f"    Uniform: {values['Uniform']}\n")
                f.write(f"    Cayley:  {values['Cayley']}\n")
                
                # Calculate improvement percentage
                if values['Uniform'] != 0 and key != 'dirichlet_energy':  # For metrics where higher is better
                    improvement = (values['Cayley'] - values['Uniform']) / abs(values['Uniform']) * 100
                    f.write(f"    Improvement: {improvement:.2f}%\n")
                elif values['Uniform'] != 0:  # For dirichlet energy, lower is better
                    improvement = (values['Uniform'] - values['Cayley']) / abs(values['Uniform']) * 100
                    f.write(f"    Improvement: {improvement:.2f}%\n")
                f.write("\n")
        
        f.write("Analysis:\n")
        f.write("  The Cayley initialization uses a mathematical structure based on the Cayley graph\n")
        f.write("  to connect base nodes to virtual nodes. This creates a sparse but structured connectivity\n")
        f.write("  pattern, which may help the model propagate information more effectively across\n")
        f.write("  distant parts of the graph.\n\n")
        
        f.write("  In contrast, the uniform initialization connects each base node to every virtual node\n")
        f.write("  with equal weights, which may lead to over-smoothing or less effective message passing.\n\n")
        
        if 'dirichlet_energy' in final_metrics:
            if final_metrics['dirichlet_energy']['Cayley'] < final_metrics['dirichlet_energy']['Uniform']:
                f.write("  The lower Dirichlet energy in the Cayley initialization suggests that it helps\n")
                f.write("  reduce oversquashing, allowing for better information flow through the graph.\n\n")
            else:
                f.write("  The higher Dirichlet energy in the Cayley initialization suggests that it may\n")
                f.write("  lead to more oversquashing compared to uniform initialization. This could be\n")
                f.write("  because the specific graph structure doesn't align well with the Cayley pattern.\n\n")
    
    logger.info(f"Comparison report saved to {output_dir}/comparison_report_{timestamp}.txt")

def main():
    parser = argparse.ArgumentParser(description='Comparison of initialization methods for IPR-MPNN')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of runs for each experiment')
    parser.add_argument('--output-dir', type=str, default='comparison_results', help='Output directory')
    parser.add_argument('--force-run', action='store_true', help='Force rerun experiments even if logs exist')
    parser.add_argument('--simulate', action='store_true', help='Simulate initialization only without running experiments')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.simulate:
        # Just run a simulation to compare the initialization methods
        logger.info("Running simulation mode to compare initialization patterns")
        
        # Define test parameters
        num_base_nodes = 30
        num_virtual_nodes = 15
        
        # Initialize with both methods
        uniform_weights = uniform_init_weights(num_base_nodes, num_virtual_nodes)
        cayley_weights = cayley_init_weights(num_base_nodes, num_virtual_nodes)
        
        # Visualize the weights
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Plot weight matrices
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(uniform_weights.numpy(), cmap='Blues')
        plt.title('Uniform Initialization')
        plt.xlabel('Virtual Nodes')
        plt.ylabel('Base Nodes')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(cayley_weights.numpy(), cmap='Blues')
        plt.title('Cayley Initialization')
        plt.xlabel('Virtual Nodes')
        plt.ylabel('Base Nodes')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"weight_visualization_{timestamp}.png"))
        plt.close()
        
        # Compute statistics
        uniform_stats = {
            "mean": uniform_weights.mean().item(),
            "std": uniform_weights.std().item(),
            "min": uniform_weights.min().item(),
            "max": uniform_weights.max().item(),
            "nonzero": (uniform_weights > 0).sum().item(),
            "sparsity": 1.0 - (uniform_weights > 0).sum().item() / uniform_weights.numel(),
            "avg_connections_per_base_node": uniform_weights.sum(dim=1).mean().item() * uniform_weights.shape[1]
        }
        
        cayley_stats = {
            "mean": cayley_weights.mean().item(),
            "std": cayley_weights.std().item(),
            "min": cayley_weights.min().item(),
            "max": cayley_weights.max().item(),
            "nonzero": (cayley_weights > 0).sum().item(),
            "sparsity": 1.0 - (cayley_weights > 0).sum().item() / cayley_weights.numel(),
            "avg_connections_per_base_node": cayley_weights.sum(dim=1).mean().item()
        }
        
        # Save statistics to file
        with open(os.path.join(vis_dir, f"statistics_{timestamp}.txt"), "w") as f:
            f.write("Comparison of Initialization Methods\n")
            f.write("==================================\n\n")
            f.write(f"Base Nodes: {num_base_nodes}\n")
            f.write(f"Virtual Nodes: {num_virtual_nodes}\n\n")
            
            f.write("Uniform Initialization:\n")
            for key, value in uniform_stats.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\nCayley Initialization:\n")
            for key, value in cayley_stats.items():
                f.write(f"  {key}: {value}\n")
                
        logger.info(f"Simulation results saved to {vis_dir}")
        return
    
    # Create modified configs for IPR-MPNN experiments
    uniform_config = create_modified_config(
        'uniform', 
        f'{args.output_dir}/mutag_uniform.yaml',
        epochs=args.epochs,
        num_runs=args.num_runs
    )
    
    cayley_config = create_modified_config(
        'cayley', 
        f'{args.output_dir}/mutag_cayley.yaml',
        epochs=args.epochs,
        num_runs=args.num_runs
    )
    
    # Define log directories
    uniform_log_dir = os.path.join(args.output_dir, 'uniform')
    cayley_log_dir = os.path.join(args.output_dir, 'cayley')
    uniform_log_file = os.path.join(uniform_log_dir, 'run.log')
    cayley_log_file = os.path.join(cayley_log_dir, 'run.log')
    
    # Run experiments if needed
    if args.force_run or not os.path.exists(uniform_log_file):
        logger.info("Starting experiment with uniform initialization...")
        run_experiment(uniform_config, uniform_log_dir)
    else:
        logger.info(f"Uniform initialization log already exists at {uniform_log_file}")
    
    if args.force_run or not os.path.exists(cayley_log_file):
        logger.info("Starting experiment with Cayley initialization...")
        run_experiment(cayley_config, cayley_log_dir)
    else:
        logger.info(f"Cayley initialization log already exists at {cayley_log_file}")
    
    # Extract and compare results
    logger.info("Extracting results from logs...")
    uniform_metrics, cayley_metrics = extract_results_from_logs(uniform_log_file, cayley_log_file)
    
    # Generate comparison plots
    logger.info("Generating comparison plots...")
    plot_comparison(uniform_metrics, cayley_metrics, args.output_dir)
    
    logger.info("Comparison complete!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
