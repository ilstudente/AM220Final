# Results for ENZYMES with k=2

Runtime: 0h 2m 2.80s

# Statistical Experiment: ENZYMES

Date: 2025-05-09 20:35:49

## Parameters
- Number of trials: 3
- Epochs per trial: 30
- Top-k connections: 2
- Batch size: 4
- Hidden dimensions: 16

## Results

### Balanced Cayley Model
- Mean accuracy: 0.2444
- Standard deviation: 0.0399
- 95% Confidence interval: [0.1231, 0.3657]

### Uniform Model
- Mean accuracy: 0.2583
- Standard deviation: 0.0136
- 95% Confidence interval: [0.2169, 0.2997]

### Statistical Significance
- T-statistic: -0.3716
- P-value: 0.7458
- The difference is not statistically significant (p >= 0.05)

### Individual Trial Results

| Trial | Balanced Cayley | Uniform |
|-------|----------------|--------|
| 1 | 0.2083 | 0.2750 |
| 2 | 0.3000 | 0.2417 |
| 3 | 0.2250 | 0.2583 |
