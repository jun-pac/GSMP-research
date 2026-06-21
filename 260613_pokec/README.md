# SNAP Pokec Temporal Split Analyzer

This folder contains a CPU-only script for analyzing SNAP Pokec registration
years and constructing an OGB-style chronological node split.

## Cheapest Local Run

The SNAP files already exist in this workspace under `../data/pokec`. This
command avoids GPU use and skips exact duplicate-edge counting for the lightest
CPU/disk/memory run:

```bash
python analyze_pokec_temporal.py \
  --profile-path ../data/pokec/soc-pokec-profiles.txt.gz \
  --edge-path ../data/pokec/soc-pokec-relationships.txt.gz \
  --out-dir pokec_temporal_outputs \
  --target-year-row 2010 \
  --duplicate-check none
```

## Exact Duplicate Count Run

This is still memory-safe, but uses temporary disk partitions while the edge
file is streamed:

```bash
python analyze_pokec_temporal.py \
  --profile-path ../data/pokec/soc-pokec-profiles.txt.gz \
  --edge-path ../data/pokec/soc-pokec-relationships.txt.gz \
  --out-dir pokec_temporal_outputs \
  --target-year-row 2010 \
  --duplicate-check partition
```

## Colab / Fresh Linux Run

```bash
python analyze_pokec_temporal.py \
  --download \
  --profile-path soc-pokec-profiles.txt.gz \
  --edge-path soc-pokec-relationships.txt.gz \
  --out-dir pokec_temporal_outputs \
  --target-year-row 2010
```

After it finishes, inspect:

- `pokec_temporal_outputs/pokec_dataset_report.md`
- `pokec_temporal_outputs/pokec_edge_counts_by_year_undirected.csv`
- `pokec_temporal_outputs/pokec_split_table.tex`
- `pokec_temporal_outputs/pokec_dataset_description.tex`

The script uses pandas chunked edge reading, a dense NumPy user-id lookup when
the IDs are dense enough, and no GPU libraries.

## Plot Connectivity

After `analyze_pokec_temporal.py` has created the CSV files, generate directed
and undirected connectivity plots with:

```bash
python plot_pokec_connectivity.py \
  --input-dir pokec_temporal_outputs \
  --out-dir pokec_connectivity_plots
```

This only reads the small CSV matrices, so it is suitable for Colab and does
not require the raw graph files or a GPU.

The plotter writes two overlaid count plots:

- `pokec_all_years_connectivity_directed.png`
- `pokec_all_years_connectivity_undirected.png`

## Direction-Agnostic Connectivity

For direction-agnostic analysis matching the paper-style GCN preprocessing,
use the `undirected` files:

- `pokec_edge_counts_by_year_undirected.csv`
- `pokec_edge_probs_by_year_undirected.csv`

These are computed from the actual coalesced GCN training graph:
`to_undirected`, remove self-loops, then add self-loops. Self-loops are excluded
from the year-connectivity matrix. This avoids overcounting reciprocal raw
relationships such as both `a -> b` and `b -> a`.

For raw directed-edge accounting, use the `symmetrized` files. Those retain the
simple matrix sum `C_sym[a,b] = C_directed[a,b] + C_directed[b,a]` and therefore
count reciprocal raw relationships twice per direction relative to the coalesced
GCN graph.
