# Results for ENZYMES with k=3

Runtime: 0h 2m 4.57s

# Statistical Experiment: ENZYMES

Date: 2025-05-09 20:37:54

## Parameters
- Number of trials: 3
- Epochs per trial: 30
- Top-k connections: 3
- Batch size: 4
- Hidden dimensions: 16

## Results

### Balanced Cayley Model
- Mean accuracy: 0.2417
- Standard deviation: 0.0360
- 95% Confidence interval: [0.1321, 0.3512]

### Uniform Model
- Mean accuracy: 0.2639
- Standard deviation: 0.0219
- 95% Confidence interval: [0.1973, 0.3304]

### Statistical Significance
- T-statistic: -1.3152
- P-value: 0.3190
- The difference is not statistically significant (p >= 0.05)

### Individual Trial Results

| Trial | Balanced Cayley | Uniform |
|-------|----------------|--------|
| 1 | 0.2250 | 0.2750 |
| 2 | 0.2917 | 0.2833 |
| 3 | 0.2083 | 0.2333 |
