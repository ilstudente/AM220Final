# Statistical Experiment Results

This directory contains statistical experiment results comparing the balanced Cayley initialization with uniform initialization in IPR-MPNN models.

## Experiment Structure

Each experiment runs multiple trials on a dataset with the following parameters:
- Number of trials (default: 10)
- Maximum number of epochs per trial (default: 20)
- Early stopping with patience parameter 
- Top-k connectivity (default: k=3)
- Hidden dimensions (default: 16)

## Visualization and Metrics

For each trial, the experiment generates:
1. Training loss curves
2. Test accuracy curves
3. Convergence tracking

For the overall experiment, it generates:
1. Statistical summary with means, standard deviations, and confidence intervals
2. T-test for statistical significance between methods
3. Visualizations of accuracy distributions
4. Convergence analysis

## Directory Structure

Each dataset gets its own subdirectory:
```
statistical_results/
├── mutag/
│   ├── trials/
│   │   ├── trial_1_convergence.png
│   │   ├── trial_2_convergence.png
│   │   └── ...
│   ├── statistical_comparison.png
│   └── summary.md
├── enzymes/
│   └── ...
└── proteins/
    └── ...
```

## Running Statistical Experiments

To run a statistical experiment on a dataset, use:

```bash
python run_statistical_experiment.py --dataset DATASET_NAME --trials NUM_TRIALS --epochs MAX_EPOCHS --k TOP_K
```

Example:
```bash
python run_statistical_experiment.py --dataset MUTAG --trials 10 --epochs 20 --k 3
```

Additional parameters:
- `--batch`: Batch size (default: 4)
- `--dim`: Hidden dimension (default: 16)
- `--seed`: Starting random seed (default: 42)
