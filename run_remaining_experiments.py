"""
Run experiments on MUTAG and ENZYMES datasets with the IPR-MPNN model.
"""

import os
import argparse
import torch
import numpy as np
from custom_aligned_comparison import run_comparison

if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set up output directories
    for dataset in ['MUTAG', 'ENZYMES']:
        output_dir = os.path.join('custom_comparison_results', dataset.lower())
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Running experiment on {dataset} dataset")
        print(f"{'='*50}\n")
        
        # Run comparison
        run_comparison(num_epochs=50, output_dir=output_dir, dataset_name=dataset)
