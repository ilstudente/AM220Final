# Oversquashing Metrics Analysis for MUTAG

## Summary of Metrics

| Metric | Balanced Cayley | Uniform | Difference |
|--------|----------------|---------|------------|
| Mean Effective Resistance | 0.7208 | 0.1822 | 0.5387 (Uniform better) |
| Average Path Length | 1.2183 | 1.0000 | 0.2183 (Uniform better) |
| Spectral Gap | 0.3744 | 0.8243 | -0.4500 (Uniform better) |
| Cheeger Constant | 0.2895 | 0.4531 | -0.1636 (Uniform better) |

## Interpretation

- **Lower effective resistance** indicates less oversquashing
- **Shorter average path length** indicates more efficient message passing
- **Larger spectral gap** indicates faster information mixing in the graph
- **Larger Cheeger constant** indicates better connectivity with fewer bottlenecks
