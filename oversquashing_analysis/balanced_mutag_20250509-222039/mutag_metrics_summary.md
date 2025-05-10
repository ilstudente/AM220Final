# Oversquashing Metrics Summary for MUTAG

## Mean Metrics

| Metric | Balanced Cayley | Uniform | Difference | Better Approach |
|--------|----------------|---------|------------|----------------|
| Effective Resistance | 2.7251 | 2.7251 | 0.0000 | Uniform |
| Cheeger Constant | 0.1315 | 0.1315 | 0.0000 | Uniform |
| Spectral Gap | 0.0710 | 0.0710 | 0.0000 | Uniform |
| Dirichlet Energy | 0.0000 | 0.0000 | 0.0000 | Uniform |

## Interpretation

- **Lower effective resistance** indicates less oversquashing
- **Higher Cheeger constant** indicates better graph connectivity (fewer bottlenecks)
- **Higher spectral gap** indicates faster information mixing in the graph
- **Higher Dirichlet energy** indicates more pronounced feature differences across graph edges

## Overall Assessment

Based on the metrics above:

- **Uniform** shows less oversquashing (lower effective resistance)
- **Uniform** has better connectivity (higher Cheeger constant)
- **Uniform** has faster information mixing (higher spectral gap)
- **Uniform** has higher feature distinction across edges (higher Dirichlet energy)

**Summary:** 0 metrics favor Balanced Cayley, 4 metrics favor Uniform initialization.

**Overall, Uniform initialization appears to be more effective at reducing oversquashing for this dataset.**
