"""
Comparison script for uniform vs. Cayley initialization in IPR-MPNN.
This script creates modified MUTAG configs for both initialization methods
and runs experiments to compare performance.
"""

import os
import json
import time
import yaml
import subprocess
import logging
import sys
import matplotlib.pyplot as plt
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_modified_config(edge_init_type, output_path, max_epoch=50, num_runs=1):
    """Create a modified config file with the specified edge initialization type."""
    # Load the original config
    with open('configs/tudatasets/mutag.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Add edge_init_type
    config['edge_init_type'] = edge_init_type
    config['wandb'] = {'use_wandb': False, 'project': 'TUDatasets', 'name': f"MUTAG_{edge_init_type}_init"}
    config['max_epoch'] = max_epoch
    config['num_runs'] = num_runs
    
    # Save the modified config
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return output_path

def run_experiment(config_path, experiment_name, output_dir):
    """Run the experiment using the specified config file."""
    logger.info(f"Running experiment with config: {config_path}")
    
    # Create output directory
    log_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Prepare log file path
    log_file = os.path.join(log_dir, "run.log")
    
    # Run the experiment using run.py
    with open(log_file, 'w') as f:
        result = subprocess.run(
            [sys.executable, 'run.py', '--cfg', config_path], 
            stdout=f, 
            stderr=subprocess.STDOUT,
            check=False
        )
    
    if result.returncode != 0:
        logger.error(f"Experiment failed with code {result.returncode}. Check {log_file} for details.")
        return False
    
    logger.info(f"Experiment completed. Logs saved to {log_file}")
    return True

def extract_metrics_from_log(log_file):
    """Extract metrics from the experiment log file."""
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_metric_acc': [],
        'val_metric_acc': []
    }
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'train_loss' in line:
                    # Try to parse the metrics
                    try:
                        parts = line.strip().split()
                        for part in parts:
                            if ':' in part:
                                key, value = part.split(':')
                                if key in metrics:
                                    metrics[key].append(float(value.strip(',').strip("'")))
                    except Exception as e:
                        logger.warning(f"Failed to parse line: {line}. Error: {e}")
    except Exception as e:
        logger.error(f"Failed to extract metrics from {log_file}. Error: {e}")
    
    return metrics

def plot_comparison(uniform_metrics, cayley_metrics, save_path):
    """Generate comparison plots for the two initialization methods."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Set up the plots
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    if uniform_metrics['train_loss'] and cayley_metrics['train_loss']:
        plt.plot(uniform_metrics['train_loss'], label='Uniform')
        plt.plot(cayley_metrics['train_loss'], label='Cayley')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    # Plot validation loss
    plt.subplot(2, 2, 2)
    if uniform_metrics['val_loss'] and cayley_metrics['val_loss']:
        plt.plot(uniform_metrics['val_loss'], label='Uniform')
        plt.plot(cayley_metrics['val_loss'], label='Cayley')
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    # Plot training metric
    plt.subplot(2, 2, 3)
    if uniform_metrics['train_metric_acc'] and cayley_metrics['train_metric_acc']:
        plt.plot(uniform_metrics['train_metric_acc'], label='Uniform')
        plt.plot(cayley_metrics['train_metric_acc'], label='Cayley')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    # Plot validation metric
    plt.subplot(2, 2, 4)
    if uniform_metrics['val_metric_acc'] and cayley_metrics['val_metric_acc']:
        plt.plot(uniform_metrics['val_metric_acc'], label='Uniform')
        plt.plot(cayley_metrics['val_metric_acc'], label='Cayley')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Comparison plot saved to {save_path}")

def main():
    # Setup
    output_dir = 'comparison_results'
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create configs
    uniform_config = create_modified_config(
        edge_init_type='uniform',
        output_path=f'{output_dir}/mutag_uniform.yaml',
        max_epoch=25,  # Reduced for faster comparison
        num_runs=1
    )
    
    cayley_config = create_modified_config(
        edge_init_type='cayley',
        output_path=f'{output_dir}/mutag_cayley.yaml',
        max_epoch=25,  # Reduced for faster comparison
        num_runs=1
    )
    
    # Run experiments
    uniform_success = run_experiment(
        config_path=uniform_config,
        experiment_name='uniform',
        output_dir=output_dir
    )
    
    cayley_success = run_experiment(
        config_path=cayley_config,
        experiment_name='cayley',
        output_dir=output_dir
    )
    
    # If both experiments succeeded, generate comparison plots
    if uniform_success and cayley_success:
        # Extract metrics
        uniform_metrics = extract_metrics_from_log(f'{output_dir}/uniform/run.log')
        cayley_metrics = extract_metrics_from_log(f'{output_dir}/cayley/run.log')
        
        # Generate plots
        plot_path = f'{output_dir}/comparison_{timestamp}.png'
        plot_comparison(uniform_metrics, cayley_metrics, plot_path)
        
        # Save metrics for future reference
        with open(f'{output_dir}/metrics.json', 'w') as f:
            json.dump({
                'uniform': uniform_metrics,
                'cayley': cayley_metrics
            }, f, indent=2)
        
        logger.info("Comparison completed successfully!")
    else:
        logger.error("One or both experiments failed. Cannot generate comparison.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")