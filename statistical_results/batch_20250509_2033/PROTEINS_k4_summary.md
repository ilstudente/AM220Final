# Results for PROTEINS with k=4

Runtime: 0h 2m 23.24s

# Statistical Experiment: PROTEINS

Date: 2025-05-09 20:47:26

## Parameters
- Number of trials: 3
- Epochs per trial: 30
- Top-k connections: 4
- Batch size: 4
- Hidden dimensions: 16

## Results

### Balanced Cayley Model
- Mean accuracy: 0.6876
- Standard deviation: 0.0298
- 95% Confidence interval: [0.5969, 0.7783]

### Uniform Model
- Mean accuracy: 0.6816
- Standard deviation: 0.0190
- 95% Confidence interval: [0.6237, 0.7395]

### Statistical Significance
- T-statistic: 0.2973
- P-value: 0.7943
- The difference is not statistically significant (p >= 0.05)

### Individual Trial Results

| Trial | Balanced Cayley | Uniform |
|-------|----------------|--------|
| 1 | 0.7040 | 0.7085 |
| 2 | 0.7130 | 0.6682 |
| 3 | 0.6457 | 0.6682 |
