# Statistical Experiment: MUTAG

Date: 2025-05-09 20:26:46

## Parameters
- Number of trials: 5
- Epochs per trial: 20
- Top-k connections: 3
- Batch size: 4
- Hidden dimensions: 16

## Results

### Balanced Cayley Model
- Mean accuracy: 0.7211
- Standard deviation: 0.0516
- 95% Confidence interval: [0.6495, 0.7926]

### Uniform Model
- Mean accuracy: 0.7053
- Standard deviation: 0.0653
- 95% Confidence interval: [0.6146, 0.7959]

### Statistical Significance
- T-statistic: 0.3679
- P-value: 0.7316
- The difference is not statistically significant (p >= 0.05)

### Individual Trial Results

| Trial | Balanced Cayley | Uniform |
|-------|----------------|--------|
| 1 | 0.6316 | 0.6579 |
| 2 | 0.7105 | 0.7368 |
| 3 | 0.7895 | 0.6053 |
| 4 | 0.7368 | 0.7368 |
| 5 | 0.7368 | 0.7895 |
