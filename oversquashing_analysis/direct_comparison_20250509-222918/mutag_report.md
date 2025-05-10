# Oversquashing Analysis: MUTAG

Date: 2025-05-09 22:31:34

## Summary

| Metric | Balanced Cayley | Uniform | Difference | Better |
|--------|----------------|---------|------------|--------|
| Algebraic Connectivity | 0.4223 | 1.0694 | -0.6472 (-60.5%) | Uniform |
| Avg Path Length | 0.0897 | 0.3833 | -0.2937 (-76.6%) | Balanced |
| Cheeger Constant | 0.4370 | 0.5571 | -0.1201 (-21.6%) | Uniform |
| Density | 0.7703 | 1.0000 | -0.2297 (-23.0%) | Uniform |
| Diameter | 0.2577 | 0.3833 | -0.1256 (-32.8%) | Balanced |
| Max Edge Weight | 0.5440 | 0.3833 | 0.1607 (41.9%) | Balanced |
| Mean Edge Weight | 0.1736 | 0.3833 | -0.2097 (-54.7%) | Uniform |
| Min Edge Weight | 0.0040 | 0.3833 | -0.3793 (-98.9%) | Uniform |
| Num Edges | 108.2000 | 140.8000 | -32.6000 (-23.2%) | Uniform |
| Num Nodes | 16.6000 | 16.6000 | 0.0000 (0.0%) | Uniform |
| Num Virtual Nodes | 7.6000 | 7.6000 | 0.0000 (0.0%) | Uniform |
| Spectral Gap | 0.4223 | 1.0694 | -0.6472 (-60.5%) | Uniform |
| Std Edge Weight | 0.1256 | 0.0000 | 0.1256 (452413783555059520.0%) | Balanced |

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

### Graph 3

| Metric | Balanced Cayley | Uniform | Difference |
|--------|----------------|---------|------------|
| Algebraic Connectivity | 0.5609 | 1.0556 | -0.4947 |
| Avg Path Length | 0.0666 | 0.3333 | -0.2667 |
| Cheeger Constant | 0.4545 | 0.5556 | -0.1010 |
| Density | 0.7836 | 1.0000 | -0.2164 |
| Diameter | 0.2222 | 0.3333 | -0.1111 |
| Max Edge Weight | 0.3333 | 0.3333 | -0.0000 |
| Mean Edge Weight | 0.1330 | 0.3333 | -0.2004 |
| Min Edge Weight | 0.0040 | 0.3333 | -0.3294 |
| Num Edges | 134.0000 | 171.0000 | -37.0000 |
| Num Nodes | 19.0000 | 19.0000 | 0.0000 |
| Num Virtual Nodes | 9.0000 | 9.0000 | 0.0000 |
| Spectral Gap | 0.5609 | 1.0556 | -0.4947 |
| Std Edge Weight | 0.0832 | 0.0000 | 0.0832 |

### Graph 4

| Metric | Balanced Cayley | Uniform | Difference |
|--------|----------------|---------|------------|
| Algebraic Connectivity | 0.6775 | 1.1000 | -0.4225 |
| Avg Path Length | 0.0783 | 0.3333 | -0.2550 |
| Cheeger Constant | 0.6000 | 0.6000 | 0.0000 |
| Density | 1.0000 | 1.0000 | 0.0000 |
| Diameter | 0.1944 | 0.3333 | -0.1389 |
| Max Edge Weight | 0.4405 | 0.3333 | 0.1071 |
| Mean Edge Weight | 0.1852 | 0.3333 | -0.1481 |
| Min Edge Weight | 0.0023 | 0.3333 | -0.3311 |
| Num Edges | 55.0000 | 55.0000 | 0.0000 |
| Num Nodes | 11.0000 | 11.0000 | 0.0000 |
| Num Virtual Nodes | 5.0000 | 5.0000 | 0.0000 |
| Spectral Gap | 0.6775 | 1.1000 | -0.4225 |
| Std Edge Weight | 0.1064 | 0.0000 | 0.1064 |

### Graph 5

| Metric | Balanced Cayley | Uniform | Difference |
|--------|----------------|---------|------------|
| Algebraic Connectivity | 0.4359 | 1.0370 | -0.6011 |
| Avg Path Length | 0.0481 | 0.3333 | -0.2852 |
| Cheeger Constant | 0.2180 | 0.5185 | -0.3006 |
| Density | 0.7672 | 1.0000 | -0.2328 |
| Diameter | 0.2222 | 0.3333 | -0.1111 |
| Max Edge Weight | 0.7083 | 0.3333 | 0.3750 |
| Mean Edge Weight | 0.1320 | 0.3333 | -0.2013 |
| Min Edge Weight | 0.0023 | 0.3333 | -0.3311 |
| Num Edges | 290.0000 | 378.0000 | -88.0000 |
| Num Nodes | 28.0000 | 28.0000 | 0.0000 |
| Num Virtual Nodes | 10.0000 | 10.0000 | 0.0000 |
| Spectral Gap | 0.4359 | 1.0370 | -0.6011 |
| Std Edge Weight | 0.1274 | 0.0000 | 0.1274 |

### Graph 6

| Metric | Balanced Cayley | Uniform | Difference |
|--------|----------------|---------|------------|
| Algebraic Connectivity | 0.2942 | 1.0667 | -0.7724 |
| Avg Path Length | 0.0334 | 0.3333 | -0.2999 |
| Cheeger Constant | 0.4583 | 0.5333 | -0.0750 |
| Density | 0.8083 | 1.0000 | -0.1917 |
| Diameter | 0.0784 | 0.3333 | -0.2550 |
| Max Edge Weight | 0.7083 | 0.3333 | 0.3750 |
| Mean Edge Weight | 0.1576 | 0.3333 | -0.1757 |
| Min Edge Weight | 0.0023 | 0.3333 | -0.3311 |
| Num Edges | 97.0000 | 120.0000 | -23.0000 |
| Num Nodes | 16.0000 | 16.0000 | 0.0000 |
| Num Virtual Nodes | 8.0000 | 8.0000 | 0.0000 |
| Spectral Gap | 0.2942 | 1.0667 | -0.7724 |
| Std Edge Weight | 0.1453 | 0.0000 | 0.1453 |

### Graph 7

| Metric | Balanced Cayley | Uniform | Difference |
|--------|----------------|---------|------------|
| Algebraic Connectivity | 0.5877 | 1.0526 | -0.4650 |
| Avg Path Length | 0.0792 | 0.3333 | -0.2541 |
| Cheeger Constant | 0.4307 | 0.5263 | -0.0957 |
| Density | 0.7263 | 1.0000 | -0.2737 |
| Diameter | 0.2222 | 0.3333 | -0.1111 |
| Max Edge Weight | 0.3333 | 0.3333 | 0.0000 |
| Mean Edge Weight | 0.1297 | 0.3333 | -0.2036 |
| Min Edge Weight | 0.0023 | 0.3333 | -0.3311 |
| Num Edges | 138.0000 | 190.0000 | -52.0000 |
| Num Nodes | 20.0000 | 20.0000 | 0.0000 |
| Num Virtual Nodes | 10.0000 | 10.0000 | 0.0000 |
| Spectral Gap | 0.5877 | 1.0526 | -0.4650 |
| Std Edge Weight | 0.0770 | 0.0000 | 0.0770 |

### Graph 8

| Metric | Balanced Cayley | Uniform | Difference |
|--------|----------------|---------|------------|
| Algebraic Connectivity | 0.3283 | 1.0909 | -0.7626 |
| Avg Path Length | 0.1803 | 0.5000 | -0.3197 |
| Cheeger Constant | 0.4286 | 0.5455 | -0.1169 |
| Density | 0.6667 | 1.0000 | -0.3333 |
| Diameter | 0.4962 | 0.5000 | -0.0038 |
| Max Edge Weight | 0.5000 | 0.5000 | 0.0000 |
| Mean Edge Weight | 0.2405 | 0.5000 | -0.2595 |
| Min Edge Weight | 0.0076 | 0.5000 | -0.4924 |
| Num Edges | 44.0000 | 66.0000 | -22.0000 |
| Num Nodes | 12.0000 | 12.0000 | 0.0000 |
| Num Virtual Nodes | 6.0000 | 6.0000 | 0.0000 |
| Spectral Gap | 0.3283 | 1.0909 | -0.7626 |
| Std Edge Weight | 0.1391 | 0.0000 | 0.1391 |

### Graph 9

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

