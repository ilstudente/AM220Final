# Results for PROTEINS with k=2

Runtime: 0h 2m 39.01s

# Statistical Experiment: PROTEINS

Date: 2025-05-09 20:42:37

## Parameters
- Number of trials: 3
- Epochs per trial: 30
- Top-k connections: 2
- Batch size: 4
- Hidden dimensions: 16

## Results

### Balanced Cayley Model
- Mean accuracy: 0.6966
- Standard deviation: 0.0148
- 95% Confidence interval: [0.6515, 0.7416]

### Uniform Model
- Mean accuracy: 0.6607
- Standard deviation: 0.0106
- 95% Confidence interval: [0.6285, 0.6928]

### Statistical Significance
- T-statistic: 2.6186
- P-value: 0.1201
- The difference is not statistically significant (p >= 0.05)

### Individual Trial Results

| Trial | Balanced Cayley | Uniform |
|-------|----------------|--------|
| 1 | 0.6996 | 0.6457 |
| 2 | 0.7130 | 0.6682 |
| 3 | 0.6771 | 0.6682 |
