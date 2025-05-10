# Oversquashing Analysis: MUTAG

Date: 2025-05-09 22:34:08

## Summary

| Metric | Balanced Cayley | Uniform | Difference | Better |
|--------|----------------|---------|------------|--------|
| Algebraic Connectivity | 0.3430 | 1.0764 | -0.7334 (-68.1%) | Uniform |
| Avg Path Length | 0.1263 | 0.4444 | -0.3182 (-71.6%) | Balanced |
| Cheeger Constant | 0.4376 | 0.5764 | -0.1388 (-24.1%) | Uniform |
| Density | 0.7141 | 1.0000 | -0.2859 (-28.6%) | Uniform |
| Diameter | 0.3544 | 0.4444 | -0.0900 (-20.3%) | Balanced |
| Max Edge Weight | 0.5694 | 0.4444 | 0.1250 (28.1%) | Balanced |
| Mean Edge Weight | 0.2015 | 0.4444 | -0.2430 (-54.7%) | Uniform |
| Min Edge Weight | 0.0058 | 0.4444 | -0.4386 (-98.7%) | Uniform |
| Num Edges | 71.3333 | 97.3333 | -26.0000 (-26.7%) | Uniform |
| Num Nodes | 14.3333 | 14.3333 | 0.0000 (0.0%) | Uniform |
| Num Virtual Nodes | 6.6667 | 6.6667 | 0.0000 (0.0%) | Uniform |
| Spectral Gap | 0.3430 | 1.0764 | -0.7334 (-68.1%) | Uniform |
| Std Edge Weight | 0.1418 | 0.0000 | 0.1418 (0.0%) | Balanced |

## Overall Assessment

- **Connectivity**: Uniform approach is better
- **Path Efficiency**: Balanced approach is better
- **Oversquashing Reduction**: Uniform approach is better

## Interpretation

- **Cheeger constant**: Higher values indicate better connectivity with fewer bottlenecks
- **Spectral gap**: Higher values indicate faster information mixing
- **Average path length**: Lower values indicate more efficient message passing
- **Diameter**: Lower values indicate shorter maximum distances between nodes
- **Density**: Higher values indicate more connections between nodes

## Individual Graph Results

### Graph 0

| Metric | Balanced Cayley | Uniform | Difference |
|--------|----------------|---------|------------|
| Algebraic Connectivity | 0.3091 | 1.0625 | -0.7534 |
| Avg Path Length | 0.0320 | 0.3333 | -0.3014 |
| Cheeger Constant | 0.4667 | 0.5625 | -0.0958 |
| Density | 0.8088 | 1.0000 | -0.1912 |
| Diameter | 0.0784 | 0.3333 | -0.2550 |
| Max Edge Weight | 0.7083 | 0.3333 | 0.3750 |
| Mean Edge Weight | 0.1538 | 0.3333 | -0.1795 |
| Min Edge Weight | 0.0023 | 0.3333 | -0.3311 |
| Num Edges | 110.0000 | 136.0000 | -26.0000 |
| Num Nodes | 17.0000 | 17.0000 | 0.0000 |
| Num Virtual Nodes | 8.0000 | 8.0000 | 0.0000 |
| Spectral Gap | 0.3091 | 1.0625 | -0.7534 |
| Std Edge Weight | 0.1521 | 0.0000 | 0.1521 |

### Graph 1

| Metric | Balanced Cayley | Uniform | Difference |
|--------|----------------|---------|------------|
| Algebraic Connectivity | 0.3599 | 1.0833 | -0.7234 |
| Avg Path Length | 0.1734 | 0.5000 | -0.3266 |
| Cheeger Constant | 0.4231 | 0.5833 | -0.1603 |
| Density | 0.6667 | 1.0000 | -0.3333 |
| Diameter | 0.4924 | 0.5000 | -0.0076 |
| Max Edge Weight | 0.5000 | 0.5000 | 0.0000 |
| Mean Edge Weight | 0.2253 | 0.5000 | -0.2747 |
| Min Edge Weight | 0.0076 | 0.5000 | -0.4924 |
| Num Edges | 52.0000 | 78.0000 | -26.0000 |
| Num Nodes | 13.0000 | 13.0000 | 0.0000 |
| Num Virtual Nodes | 6.0000 | 6.0000 | 0.0000 |
| Spectral Gap | 0.3599 | 1.0833 | -0.7234 |
| Std Edge Weight | 0.1366 | 0.0000 | 0.1366 |

### Graph 2

| Metric | Balanced Cayley | Uniform | Difference |
|--------|----------------|---------|------------|
| Algebraic Connectivity | 0.3599 | 1.0833 | -0.7234 |
| Avg Path Length | 0.1734 | 0.5000 | -0.3266 |
| Cheeger Constant | 0.4231 | 0.5833 | -0.1603 |
| Density | 0.6667 | 1.0000 | -0.3333 |
| Diameter | 0.4924 | 0.5000 | -0.0076 |
| Max Edge Weight | 0.5000 | 0.5000 | 0.0000 |
| Mean Edge Weight | 0.2253 | 0.5000 | -0.2747 |
| Min Edge Weight | 0.0076 | 0.5000 | -0.4924 |
| Num Edges | 52.0000 | 78.0000 | -26.0000 |
| Num Nodes | 13.0000 | 13.0000 | 0.0000 |
| Num Virtual Nodes | 6.0000 | 6.0000 | 0.0000 |
| Spectral Gap | 0.3599 | 1.0833 | -0.7234 |
| Std Edge Weight | 0.1366 | 0.0000 | 0.1366 |

