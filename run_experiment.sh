#!/bin/bash
# Script to run a specific statistical experiment

# Default parameters
dataset="MUTAG"
trials=5
epochs=20
k=3
batch=4
dim=16
patience=5
delta=0.01
seed=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      dataset="$2"
      shift
      shift
      ;;
    --trials)
      trials="$2"
      shift
      shift
      ;;
    --epochs)
      epochs="$2"
      shift
      shift
      ;;
    --k)
      k="$2"
      shift
      shift
      ;;
    --batch)
      batch="$2"
      shift
      shift
      ;;
    --dim)
      dim="$2"
      shift
      shift
      ;;
    --patience)
      patience="$2"
      shift
      shift
      ;;
    --delta)
      delta="$2"
      shift
      shift
      ;;
    --seed)
      seed="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Running statistical experiment with the following parameters:"
echo "Dataset: $dataset"
echo "Trials: $trials"
echo "Epochs: $epochs"
echo "Top-k: $k"
echo "Batch size: $batch"
echo "Hidden dimensions: $dim"
echo "Patience: $patience"
echo "Delta: $delta"
echo "Seed: $seed"
echo "----------------------------------------"

# Execute the experiment
python run_statistical_experiment.py \
  --dataset "$dataset" \
  --trials "$trials" \
  --epochs "$epochs" \
  --k "$k" \
  --batch "$batch" \
  --dim "$dim" \
  --patience "$patience" \
  --delta "$delta" \
  --seed "$seed"

echo "Experiment completed!"
