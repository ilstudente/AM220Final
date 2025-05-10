# Results for ENZYMES with k=4

Runtime: 0h 2m 3.68s

# Statistical Experiment: ENZYMES

Date: 2025-05-09 20:39:58

## Parameters
- Number of trials: 3
- Epochs per trial: 30
- Top-k connections: 4
- Batch size: 4
- Hidden dimensions: 16

## Results

### Balanced Cayley Model
- Mean accuracy: 0.2361
- Standard deviation: 0.0393
- 95% Confidence interval: [0.1166, 0.3556]

### Uniform Model
- Mean accuracy: 0.2639
- Standard deviation: 0.0219
- 95% Confidence interval: [0.1973, 0.3304]

### Statistical Significance
- T-statistic: -1.2804
- P-value: 0.3288
- The difference is not statistically significant (p >= 0.05)

### Individual Trial Results

| Trial | Balanced Cayley | Uniform |
|-------|----------------|--------|
| 1 | 0.2083 | 0.2750 |
| 2 | 0.2917 | 0.2833 |
| 3 | 0.2083 | 0.2333 |
