"""
Script to run a batch of statistical experiments across multiple datasets and k values.
"""

import os
import argparse
import subprocess
from datetime import datetime
import time

def run_all_statistical_experiments(datasets=['MUTAG', 'ENZYMES', 'PROTEINS'], 
                                   k_values=[2, 3, 4],
                                   num_trials=5,
                                   max_epochs=30,
                                   patience=7,
                                   convergence_delta=0.01,
                                   hidden_dim=16,
                                   batch_size=4,
                                   start_seed=42):
    """
    Run statistical experiments across multiple datasets and k values.
    
    Args:
        datasets: List of datasets to run experiments on
        k_values: List of top-k values to use
        num_trials: Number of trials per experiment
        max_epochs: Maximum number of epochs per trial
        patience: Early stopping patience
        convergence_delta: Improvement threshold for early stopping
        hidden_dim: Hidden dimension size
        batch_size: Training batch size
        start_seed: Starting random seed
    """
    total_experiments = len(datasets) * len(k_values)
    experiment_count = 0
    
    # Create overall results directory
    batch_dir = f"statistical_results/batch_{datetime.now().strftime('%Y%m%d_%H%M')}"
    os.makedirs(batch_dir, exist_ok=True)
    
    # Log file for batch results
    log_file = f"{batch_dir}/batch_summary.md"
    
    with open(log_file, "w") as f:
        f.write("# Batch Statistical Experiment Results\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Experiments\n\n")
        f.write("| Dataset | Top-k | Mean Balanced | Mean Uniform | p-value | Significant |\n")
        f.write("|---------|-------|---------------|--------------|---------|-------------|\n")
    
    # Record overall start time
    overall_start = time.time()
    
    for dataset in datasets:
        for k in k_values:
            experiment_count += 1
            print(f"\n[{experiment_count}/{total_experiments}] Running experiment: {dataset}, k={k}")
            
            # Record experiment start time
            exp_start = time.time()
            
            # Python command to run the experiment
            cmd = [
                "python", "run_statistical_experiment.py",
                "--dataset", dataset,
                "--trials", str(num_trials),
                "--epochs", str(max_epochs),
                "--k", str(k),
                "--batch", str(batch_size),
                "--dim", str(hidden_dim),
                "--seed", str(start_seed)
            ]
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Calculate experiment time
            exp_elapsed = time.time() - exp_start
            hours, remainder = divmod(exp_elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
            
            # Extract results (rough parsing of output)
            output = result.stdout
            
            # Extract balanced and uniform means
            balanced_line = [line for line in output.split('\n') if "Balanced Cayley: Mean=" in line]
            uniform_line = [line for line in output.split('\n') if "Uniform: Mean=" in line]
            p_value_line = [line for line in output.split('\n') if "T-test: t=" in line]
            
            balanced_mean = "N/A"
            uniform_mean = "N/A"
            p_value = "N/A"
            significant = "N/A"
            
            if balanced_line:
                parts = balanced_line[0].split("Mean=")[1].split(",")
                balanced_mean = parts[0].strip()
            
            if uniform_line:
                parts = uniform_line[0].split("Mean=")[1].split(",")
                uniform_mean = parts[0].strip()
            
            if p_value_line:
                parts = p_value_line[0].split("p=")[1].split()
                p_value = parts[0].strip()
                # Determine significance
                try:
                    p_val_float = float(p_value)
                    significant = "Yes" if p_val_float < 0.05 else "No"
                except:
                    significant = "N/A"
            
            # Log results to summary file
            with open(log_file, "a") as f:
                f.write(f"| {dataset} | {k} | {balanced_mean} | {uniform_mean} | {p_value} | {significant} |\n")
            
            # Copy the dataset's summary file to the batch directory
            dataset_summary = f"statistical_results/{dataset.lower()}/summary.md"
            if os.path.exists(dataset_summary):
                with open(dataset_summary, "r") as src:
                    with open(f"{batch_dir}/{dataset}_k{k}_summary.md", "w") as dst:
                        dst.write(f"# Results for {dataset} with k={k}\n\n")
                        dst.write(f"Runtime: {time_str}\n\n")
                        dst.write(src.read())
            
            print(f"Completed {dataset} k={k} in {time_str}")
            
    # Calculate overall elapsed time
    overall_elapsed = time.time() - overall_start
    hours, remainder = divmod(overall_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Add completion time to log
    with open(log_file, "a") as f:
        f.write(f"\n## Summary\n\n")
        f.write(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")
        f.write(f"Completed {experiment_count} experiments\n")
    
    print(f"\nAll experiments completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Results saved to {batch_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run batch statistical experiments')
    parser.add_argument('--datasets', nargs='+', default=['MUTAG', 'ENZYMES', 'PROTEINS'],
                       help='Datasets to run experiments on')
    parser.add_argument('--k-values', nargs='+', type=int, default=[2, 3, 4],
                       help='Top-k values to test')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials per experiment')
    parser.add_argument('--epochs', type=int, default=30, help='Maximum epochs per trial')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--delta', type=float, default=0.01, help='Improvement threshold for early stopping')
    parser.add_argument('--dim', type=int, default=16, help='Hidden dimension')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Starting random seed')
    
    args = parser.parse_args()
    
    run_all_statistical_experiments(
        datasets=args.datasets,
        k_values=args.k_values,
        num_trials=args.trials,
        max_epochs=args.epochs,
        patience=args.patience,
        convergence_delta=args.delta,
        hidden_dim=args.dim,
        batch_size=args.batch,
        start_seed=args.seed
    )
