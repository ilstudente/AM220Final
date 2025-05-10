# Oversquashing Analysis: PROTEINS

Date: 2025-05-09 22:32:11

## Summary

| Metric | Balanced Cayley | Uniform | Difference | Better |
|--------|----------------|---------|------------|--------|
| Algebraic Connectivity | 0.5079 | 1.0748 | -0.5669 (-52.7%) | Uniform |
| Avg Path Length | 0.0823 | 0.3333 | -0.2510 (-75.3%) | Balanced |
| Cheeger Constant | 0.4263 | 0.5587 | -0.1324 (-23.7%) | Uniform |
| Density | 0.8769 | 1.0000 | -0.1231 (-12.3%) | Uniform |
| Diameter | 0.1984 | 0.3333 | -0.1349 (-40.5%) | Balanced |
| Max Edge Weight | 0.6458 | 0.3333 | 0.3125 (93.7%) | Balanced |
| Mean Edge Weight | 0.1607 | 0.3333 | -0.1726 (-51.8%) | Uniform |
| Min Edge Weight | 0.0099 | 0.3333 | -0.3234 (-97.0%) | Uniform |
| Num Edges | 127.0000 | 158.8333 | -31.8333 (-20.0%) | Uniform |
| Num Nodes | 17.0000 | 17.0000 | 0.0000 (0.0%) | Uniform |
| Num Virtual Nodes | 7.3333 | 7.3333 | 0.0000 (0.0%) | Uniform |
| Spectral Gap | 0.5079 | 1.0748 | -0.5669 (-52.7%) | Uniform |
| Std Edge Weight | 0.1174 | 0.0000 | 0.1174 (158667694741487552.0%) | Balanced |

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
| Algebraic Connectivity | 0.3855 | 1.0385 | -0.6529 |
| Avg Path Length | 0.0414 | 0.3333 | -0.2920 |
| Cheeger Constant | 0.1928 | 0.5192 | -0.3265 |
| Density | 0.7778 | 1.0000 | -0.2222 |
| Diameter | 0.1587 | 0.3333 | -0.1746 |
| Max Edge Weight | 0.7083 | 0.3333 | 0.3750 |
| Mean Edge Weight | 0.1259 | 0.3333 | -0.2074 |
| Min Edge Weight | 0.0040 | 0.3333 | -0.3294 |
| Num Edges | 273.0000 | 351.0000 | -78.0000 |
| Num Nodes | 27.0000 | 27.0000 | 0.0000 |
| Num Virtual Nodes | 10.0000 | 10.0000 | 0.0000 |
| Spectral Gap | 0.3855 | 1.0385 | -0.6529 |
| Std Edge Weight | 0.1277 | 0.0000 | 0.1277 |

### Graph 1

| Metric | Balanced Cayley | Uniform | Difference |
|--------|----------------|---------|------------|
| Algebraic Connectivity | 0.6011 | 1.1111 | -0.5100 |
| Avg Path Length | 0.1075 | 0.3333 | -0.2259 |
| Cheeger Constant | 0.5556 | 0.5556 | 0.0000 |
| Density | 1.0000 | 1.0000 | 0.0000 |
| Diameter | 0.2222 | 0.3333 | -0.1111 |
| Max Edge Weight | 0.7083 | 0.3333 | 0.3750 |
| Mean Edge Weight | 0.1887 | 0.3333 | -0.1446 |
| Min Edge Weight | 0.0159 | 0.3333 | -0.3175 |
| Num Edges | 45.0000 | 45.0000 | 0.0000 |
| Num Nodes | 10.0000 | 10.0000 | 0.0000 |
| Num Virtual Nodes | 5.0000 | 5.0000 | 0.0000 |
| Spectral Gap | 0.6011 | 1.1111 | -0.5100 |
| Std Edge Weight | 0.1267 | 0.0000 | 0.1267 |

### Graph 2

| Metric | Balanced Cayley | Uniform | Difference |
|--------|----------------|---------|------------|
| Algebraic Connectivity | 0.3759 | 1.0435 | -0.6676 |
| Avg Path Length | 0.0423 | 0.3333 | -0.2910 |
| Cheeger Constant | 0.1880 | 0.5217 | -0.3338 |
| Density | 0.7645 | 1.0000 | -0.2355 |
| Diameter | 0.1429 | 0.3333 | -0.1905 |
| Max Edge Weight | 0.7083 | 0.3333 | 0.3750 |
| Mean Edge Weight | 0.1277 | 0.3333 | -0.2056 |
| Min Edge Weight | 0.0040 | 0.3333 | -0.3294 |
| Num Edges | 211.0000 | 276.0000 | -65.0000 |
| Num Nodes | 24.0000 | 24.0000 | 0.0000 |
| Num Virtual Nodes | 10.0000 | 10.0000 | 0.0000 |
| Spectral Gap | 0.3759 | 1.0435 | -0.6676 |
| Std Edge Weight | 0.1220 | 0.0000 | 0.1220 |

### Graph 3

| Metric | Balanced Cayley | Uniform | Difference |
|--------|----------------|---------|------------|
| Algebraic Connectivity | 0.5876 | 1.1000 | -0.5124 |
| Avg Path Length | 0.1051 | 0.3333 | -0.2283 |
| Cheeger Constant | 0.6000 | 0.6000 | 0.0000 |
| Density | 1.0000 | 1.0000 | 0.0000 |
| Diameter | 0.2222 | 0.3333 | -0.1111 |
| Max Edge Weight | 0.7083 | 0.3333 | 0.3750 |
| Mean Edge Weight | 0.1903 | 0.3333 | -0.1430 |
| Min Edge Weight | 0.0159 | 0.3333 | -0.3175 |
| Num Edges | 55.0000 | 55.0000 | 0.0000 |
| Num Nodes | 11.0000 | 11.0000 | 0.0000 |
| Num Virtual Nodes | 5.0000 | 5.0000 | 0.0000 |
| Spectral Gap | 0.5876 | 1.1000 | -0.5124 |
| Std Edge Weight | 0.1220 | 0.0000 | 0.1220 |

### Graph 4

| Metric | Balanced Cayley | Uniform | Difference |
|--------|----------------|---------|------------|
| Algebraic Connectivity | 0.5096 | 1.0556 | -0.5459 |
| Avg Path Length | 0.0926 | 0.3333 | -0.2407 |
| Cheeger Constant | 0.4215 | 0.5556 | -0.1341 |
| Density | 0.7193 | 1.0000 | -0.2807 |
| Diameter | 0.2222 | 0.3333 | -0.1111 |
| Max Edge Weight | 0.3333 | 0.3333 | 0.0000 |
| Mean Edge Weight | 0.1412 | 0.3333 | -0.1921 |
| Min Edge Weight | 0.0040 | 0.3333 | -0.3294 |
| Num Edges | 123.0000 | 171.0000 | -48.0000 |
| Num Nodes | 19.0000 | 19.0000 | 0.0000 |
| Num Virtual Nodes | 9.0000 | 9.0000 | 0.0000 |
| Spectral Gap | 0.5096 | 1.0556 | -0.5459 |
| Std Edge Weight | 0.0842 | 0.0000 | 0.0842 |

### Graph 5

| Metric | Balanced Cayley | Uniform | Difference |
|--------|----------------|---------|------------|
| Algebraic Connectivity | 0.5876 | 1.1000 | -0.5124 |
| Avg Path Length | 0.1051 | 0.3333 | -0.2283 |
| Cheeger Constant | 0.6000 | 0.6000 | 0.0000 |
| Density | 1.0000 | 1.0000 | 0.0000 |
| Diameter | 0.2222 | 0.3333 | -0.1111 |
| Max Edge Weight | 0.7083 | 0.3333 | 0.3750 |
| Mean Edge Weight | 0.1903 | 0.3333 | -0.1430 |
| Min Edge Weight | 0.0159 | 0.3333 | -0.3175 |
| Num Edges | 55.0000 | 55.0000 | 0.0000 |
| Num Nodes | 11.0000 | 11.0000 | 0.0000 |
| Num Virtual Nodes | 5.0000 | 5.0000 | 0.0000 |
| Spectral Gap | 0.5876 | 1.1000 | -0.5124 |
| Std Edge Weight | 0.1220 | 0.0000 | 0.1220 |

