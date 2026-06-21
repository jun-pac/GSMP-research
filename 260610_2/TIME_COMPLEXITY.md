# Time Complexity Notes

This note summarizes the computational cost of the completed TAPE experiments in
this folder. It covers only GNN training/evaluation using the official TAPE
feature files. It does not include the cost of generating the TAPE language-model
features, because those `.emb` files were precomputed and placed under
`upstream/TAPE`.

## Symbols

- `n`: number of nodes.
- `m`: number of edges after the graph is converted to the training graph.
  In these logs, the graph has `m = 2,484,941` edges after bidirection and
  self-loops.
- `L`: number of GNN layers.
- `d_l`: hidden width at layer `l`; `d_0` is the input feature dimension.
- `T`: training epochs per run. Here `T = 200` for full runs.
- `S`: number of seeds. Here `S = 3`.
- `F`: number of TAPE feature views. Here `F = 3` for `TA`, `P`, and `E`.

## TAPE+LinearRevGAT

LinearRevGAT removes attention-score computation and keeps a RevGAT-like model
shell with linear message aggregation.

For layer `l`, the dominant terms are:

```text
node projection:  O(n d_{l-1} d_l)
edge aggregation: O(m d_l)
```

Thus one epoch costs:

```text
O( sum_l [ n d_{l-1} d_l + m d_l ] )
```

The full 3-seed, 3-feature-view experiment costs:

```text
O( S F T sum_l [ n d_{l-1} d_l + m d_l ] )
```

Memory is dominated by node activations, graph structure, feature tensors, and
model parameters:

```text
O(n d_max + m + sum_l d_{l-1} d_l)
```

where `d_max` is the largest hidden width used during training.

## TAPE+LinearRevGAT+GSMP

GSMP adds temporal edge weights before training. The preprocessing step scans
the graph edges, compares node years, creates edge weights, and normalizes them.
With tensor operations this is linear in the edge count:

```text
GSMP preprocessing per feature view: O(m)
```

The cached GSMP weights are reused across seeds. With caching, the full
experiment costs:

```text
O( F m + S F T sum_l [ n d_{l-1} d_l + m d_l ] )
```

Without caching, the preprocessing term would be repeated for each seed:

```text
O( S F m + S F T sum_l [ n d_{l-1} d_l + m d_l ] )
```

The training-time asymptotic complexity is the same as TAPE+LinearRevGAT,
because GSMP only changes the scalar edge weights used by aggregation. The edge
aggregation term remains:

```text
O(m d_l)
```

GSMP adds edge-weight storage:

```text
additional memory: O(m)
```

For this graph, the cached GSMP files are small relative to the model/feature
tensors; `cache/gsmp` was about 48 MB after these runs.

## Measured Full-Run Results

These are the completed full runs using official TAPE `TA_P_E` features,
`S = 3` seeds, and `T = 200` epochs.

| Method | Val Acc | Test Acc | Slurm time | GPU-hours | Billing-hours | Max CPU RAM | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TAPE+LinearRevGAT | `0.7764 +/- 0.0020` | `0.7668 +/- 0.0035` | `8m54s` | `0.148` | `1.19` | `~5.1 GB` | `~6.7 GB` |
| TAPE+LinearRevGAT+GSMP | `0.7767 +/- 0.0004` | `0.7678 +/- 0.0026` | `9m00s` | `0.150` | `1.20` | `~5.0 GB` | `~6.7 GB` |

The observed GSMP overhead was therefore very small:

```text
extra wall time: 6 seconds
extra GPU-hours: about 0.002
extra billing-hours: about 0.01
```

## Interpretation

Asymptotically, GSMP does not change the per-epoch training complexity of
LinearRevGAT. It adds an `O(m)` preprocessing/cache term and `O(m)` extra edge
weight storage. Since the experiment uses 200 epochs and 3 feature views, the
training term dominates the one-time GSMP preprocessing term.

Empirically, the final ensemble improvement was small:

```text
Val improvement:  +0.0003
Test improvement: +0.0010
```

So the current result is computationally cheap, but the accuracy gain is modest.
