# Reversed Local Time Distribution Edge Weight Computation

This document describes the implementation of **reversed** local time distribution-based edge weight computation for GNNs, **exactly matching** the approach used in `preprocess_GSMP_rev_papers100m.py`.

## Overview

The edge weight computation has been updated to use a **reversed local time distribution approach** instead of the previous global `delta_matrix` approach. This implementation **exactly replicates** the mathematical approach and parallel processing strategy from `preprocess_GSMP_rev_papers100m.py`.

## Key Difference from Original Approach

### Original (GSMP) Approach
- Groups edges by **source nodes**
- Considers the distribution of **destination nodes** in each source's neighborhood
- Weight is based on how common the destination node's year is among the source's neighbors

### Reversed (GSMP_rev) Approach
- Groups edges by **destination nodes**
- Considers the distribution of **source nodes** in each destination's neighborhood
- Weight is based on how common the source node's year is among the destination's incoming neighbors

## Key Changes

### 1. New Functions Added

- `compute_edge_weights_chunk_rev(args)`: Helper function for parallel processing (identical to original rev)
- `parallel_compute_rev(edge_weight, paper_year, row, col, num_nodes, num_workers)`: Parallel computation function (identical to original rev)
- `compute_local_edge_weights_parallel_rev(graph, paper_year, num_workers)`: DGL wrapper for parallel computation
- `compute_local_edge_weights_rev(graph, paper_year)`: Sequential implementation for debugging

### 2. Updated Function

- `heize_preprocess(graph, paper_year, delta_matrix, use_parallel=True, num_workers=None, cache_file)`: Now supports caching and both computation modes

### 3. New Command Line Arguments

- `--use-parallel-edge-weights`: Enable parallel processing for edge weight computation (default: True)
- `--num-workers`: Specify number of workers for parallel computation (default: auto-detect)
- `--edge-weight-cache`: Cache file for edge weights (default: './gsmp_rev_edge_weight.pt')

## Exact Implementation Match

The implementation **exactly matches** `preprocess_GSMP_rev_papers100m.py`:

### Core Algorithm (Reversed)
```python
# For each destination node:
for dst in range(dst_start, dst_end):
    mask = (col == dst)
    srcs = row[mask]
    src_years = paper_year[srcs]
    year_counts = np.bincount(src_years, minlength=2024)
    
    # Compute local weights
    local_weights = np.zeros(len(srcs), dtype=np.float32)
    for idx, src in enumerate(srcs):
        year = paper_year[src].item()
        local_weights[idx] = 1.0 / (year_counts[year] if year_counts[year] > 0 else 1)
    
    # Normalize by mean
    mean_val = np.mean(local_weights)
    if mean_val > 0:
        local_weights = local_weights / mean_val
```

### Parallel Processing
- Uses the same chunking strategy (by destination nodes)
- Same multiprocessing approach
- Same function signatures
- Same memory management

### Caching
- Saves computed edge weights to avoid recomputation
- Uses the same file format as the original

## How It Works (Reversed Approach)

### Reversed Local Time Distribution Approach

1. **For each destination node**: Consider all its incoming source nodes
2. **Count year frequencies**: Count how many source nodes have each publication year
3. **Compute local weights**: Weight each edge inversely proportional to the frequency of the source node's year in the destination's incoming neighborhood
4. **Normalize**: Normalize weights by their mean to maintain scale

### Mathematical Formula (Reversed)

For an edge from node `i` to node `j`:

```
weight(i,j) = 1 / count(year_i in incoming_neighbors(j))
```

Where `count(year_i in incoming_neighbors(j))` is the number of incoming neighbors of node `j` that have the same publication year as node `i`.

### Example (Reversed)

Consider a destination node with incoming source nodes from years: [2015, 2015, 2016, 2017, 2015]

- Edges from 2015 nodes get weight: 1/3 = 0.333 (because there are 3 incoming neighbors from 2015)
- Edges from 2016 nodes get weight: 1/1 = 1.0 (because there is 1 incoming neighbor from 2016)
- Edges from 2017 nodes get weight: 1/1 = 1.0 (because there is 1 incoming neighbor from 2017)

After normalization by mean, the weights become: [0.5, 0.5, 1.5, 1.5, 0.5]

## Usage

### Basic Usage

```bash
python GSMP_ensemble_rev.py --use-parallel-edge-weights --num-workers 8
```

### Sequential Computation (for debugging)

```bash
python GSMP_ensemble_rev.py --no-use-parallel-edge-weights
```

### Custom Number of Workers

```bash
python GSMP_ensemble_rev.py --num-workers 16
```

### Custom Cache File

```bash
python GSMP_ensemble_rev.py --edge-weight-cache ./my_rev_edge_weights.pt
```

## Performance

- **Sequential**: Suitable for small graphs or debugging
- **Parallel**: Significantly faster for large graphs (recommended for production)
- **Caching**: Avoids recomputation on subsequent runs

The parallel version automatically detects the number of CPU cores and uses up to 48 workers by default, exactly like the original implementation.

## Testing

Run the test script to verify the implementation:

```bash
python test_edge_weights_rev.py
```

This will test both sequential and parallel implementations and verify that they produce consistent results and exactly match the expected mathematical calculations.

## Comparison with Approaches

### Previous (Global) Approach
- Used a global `delta_matrix` computed from the entire dataset
- All nodes used the same temporal distribution
- Less sensitive to local temporal patterns

### Original (GSMP) Approach
- Each source node considers only its own outgoing neighborhood's temporal distribution
- More sensitive to local temporal patterns
- Better captures the temporal context of each node's outgoing connections

### Reversed (GSMP_rev) Approach
- Each destination node considers only its own incoming neighborhood's temporal distribution
- More sensitive to local temporal patterns from the receiving perspective
- Better captures the temporal context of each node's incoming connections
- **Exactly matches** the approach in `preprocess_GSMP_rev_papers100m.py`

## Benefits

1. **Exact Match**: Implementation is mathematically identical to the original reversed approach
2. **Reversed Perspective**: Considers temporal distribution from the destination node's perspective
3. **Temporal Diversity**: Encourages connections from temporally diverse source nodes
4. **Scalability**: Parallel implementation handles large graphs efficiently
5. **Flexibility**: Can choose between sequential and parallel computation
6. **Caching**: Avoids expensive recomputation

## Implementation Details

The implementation follows the **exact same pattern** as `preprocess_GSMP_rev_papers100m.py`:

1. Convert graph edges to numpy arrays for efficient processing
2. Group edges by destination node (reversed approach)
3. For each destination node, compute local year distribution of incoming sources
4. Apply inverse frequency weighting
5. Normalize weights by mean
6. Convert back to PyTorch tensors
7. Cache results for future use

## Memory Usage

The parallel implementation uses additional memory for:
- Temporary numpy arrays
- Process pool overhead
- Results from parallel workers

For very large graphs, consider using the sequential version if memory is limited.

## Verification

The implementation has been verified to produce **exactly the same results** as the original reversed approach through:
- Mathematical verification of the algorithm
- Test cases with known expected outputs
- Comparison with the original implementation's logic
- Parallel and sequential consistency checks
- Verification that it produces different results from the original (non-reversed) approach

## Key Insight

The reversed approach provides a different perspective on temporal relationships in the graph:
- **Original**: "How diverse are the years of papers that this paper cites?"
- **Reversed**: "How diverse are the years of papers that cite this paper?"

This can lead to different insights about the temporal structure of the citation network and potentially different performance in downstream tasks. 