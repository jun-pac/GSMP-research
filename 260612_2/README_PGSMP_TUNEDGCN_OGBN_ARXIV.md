# P-GSMP on tunedGNN GCN for ogbn-arxiv

This folder contains a clean local tunedGNN checkout plus a scoped P-GSMP patch for the large-graph ogbn-arxiv GCN experiment.

Upstream:

- Repository: https://github.com/LUOyk1999/tunedGNN
- Local commit: `23f9604e8b13a9a6d3faa2f691cd844006979153`
- Official large-graph command inspected: `upstream/tunedGNN/large_graph/arxiv.sh`

## What Was Inspected

The tunedGNN ogbn-arxiv GCN path is:

- Training script: `upstream/tunedGNN/large_graph/main-arxiv.py`
- CLI/config defaults: `upstream/tunedGNN/large_graph/lg_parse.py`
- Model: `upstream/tunedGNN/large_graph/lg_model.py`
- Dataset loader: `upstream/tunedGNN/large_graph/dataset.py`
- Evaluation: `upstream/tunedGNN/large_graph/eval.py` and `data_utils.py`

Official GCN command from `arxiv.sh`:

```bash
python main-arxiv.py --dataset ogbn-arxiv --hidden_channels 512 --epochs 2000 --lr 0.0005 --runs 2 --local_layers 5 --bn --device 0 --res
```

The code loads ogbn-arxiv with OGB `NodePropPredDataset`, uses the official OGB split, converts citation edges with `to_undirected`, removes existing self-loops, then adds one self-loop per node for GCN training. The GCN uses PyG `GCNConv(cached=False, normalize=True)`, residual linear paths, batch norm when `--bn` is set, Adam, parser default `weight_decay=5e-4`, parser default `in_dropout=0.15`, and parser default `dropout=0.5`.

## P-GSMP vs Layer-Wise GSMP

P-GSMP is not layer-wise GSMP. It computes a timestamp-balanced neighbor feature matrix once:

```text
X_pg = P_GSMP(X_original)
logits = ordinary_GCN(X_pg, edge_index)
```

The GCN architecture and message passing are unchanged. This experiment only swaps the input feature matrix when `--use-pgsmp` is enabled.

For each edge `src -> dst`, source features contribute to the target. Counts are keyed by:

```text
count[dst, node_year[src]]
```

Default P-GSMP settings:

```text
alpha=0.5
depth=1
norm=strict_observed
self_mode=neighbor_only
```

The training graph is the official tunedGNN undirected graph with self-loops. P-GSMP preprocessing uses the same undirected edge direction but removes self-loops for `neighbor_only`; `include_self` adds exactly one self-edge per node for preprocessing only.

## Environment

The upstream README reports testing with Python 3.7, PyTorch 1.12.1, PyG 2.3.1, and DGL 1.0.2. The SLURM scripts have an editable section:

```bash
source ~/.bashrc
conda activate "${CONDA_ENV:-tunedgnn}"
```

Edit that section for your cluster.

## Static Checks

Run these before spending GPU time:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260612_2
python -m py_compile upstream/tunedGNN/large_graph/pgsmp.py
python -m py_compile upstream/tunedGNN/large_graph/lg_parse.py
python -m py_compile upstream/tunedGNN/large_graph/dataset.py
python -m py_compile upstream/tunedGNN/large_graph/main-arxiv.py
python tests/test_pgsmp.py
```

## One-Command Smoke Test

This runs 3 epochs of baseline and 3 epochs of P-GSMP for one seed:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260612_2
bash scripts/run_tunedgcn_pgsmp_experiments.sh smoke
```

Useful overrides:

```bash
CPU=1 bash scripts/run_tunedgcn_pgsmp_experiments.sh smoke
EPOCHS=5 DEVICE=0 bash scripts/run_tunedgcn_pgsmp_experiments.sh smoke
```

## Baseline First

The previous `260612` baseline check already produced a healthy seed-0 run:

```text
/users/PAS1289/jyp531/GSMP-research/260612/results/tunedgcn_gsmp/20260611_213823/final_summary.json
val_mean=0.746938
test_at_best_val_mean=0.731004
best_raw_test_mean=0.736415
```

To reproduce tunedGNN GCN again in this folder:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260612_2
SEEDS="0" bash scripts/run_tunedgcn_pgsmp_experiments.sh baseline
```

Or with SLURM:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260612_2
SEED=0 sbatch slurm/reproduce_tunedgcn_baseline.sbatch
```

Default full scripts use `EVAL_EVERY=10` to save time. For stricter reproduction of the official selection protocol, use:

```bash
EVAL_EVERY=1 LOG_EVERY=50 SEED=0 sbatch slurm/reproduce_tunedgcn_baseline.sbatch
```

Expected leaderboard target:

- tunedGNN GCN validation accuracy: `0.7447 +/- 0.0014`
- tunedGNN GCN test accuracy: `0.7360 +/- 0.0018`

If the baseline is far below target, stop before running full P-GSMP:

```bash
python tools/check_baseline_gate.py results/tunedgcn_pgsmp/<run>/final_summary.json
```

## Run GCN+P-GSMP

After the baseline is healthy:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260612_2
SEEDS="0" bash scripts/run_tunedgcn_pgsmp_experiments.sh pgsmp
```

SLURM array, one seed per task:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260612_2
sbatch slurm/run_tunedgcn_pgsmp.sbatch
```

The P-GSMP SLURM script defaults to one array task to avoid accidental extra GPU allocations. For five seeds, submit explicitly:

```bash
SEEDS="0 1 2 3 4" sbatch --array=0-4%1 slurm/run_tunedgcn_pgsmp.sbatch
```

Baseline plus P-GSMP array:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260612_2
sbatch slurm/run_tunedgcn_pgsmp_all_array.sbatch
```

For five seeds across both methods, submit an explicit ten-task array:

```bash
SEEDS="0 1 2 3 4" sbatch --array=0-9%1 slurm/run_tunedgcn_pgsmp_all_array.sbatch
```

Optional ablations only after the default works:

```bash
PGSMP_ALPHA=0.25 bash scripts/run_tunedgcn_pgsmp_experiments.sh pgsmp
PGSMP_ALPHA=0.75 bash scripts/run_tunedgcn_pgsmp_experiments.sh pgsmp
PGSMP_DEPTH=2 bash scripts/run_tunedgcn_pgsmp_experiments.sh pgsmp
PGSMP_NORM=scale_preserve bash scripts/run_tunedgcn_pgsmp_experiments.sh pgsmp
PGSMP_SELF_MODE=include_self bash scripts/run_tunedgcn_pgsmp_experiments.sh pgsmp
```

## Monitor Logs

```bash
tail -f logs/<jobname>_<jobid>.out
grep "\[PGSMP\]" logs/<jobname>_<jobid>.out
grep "\[RESULT\]" logs/<jobname>_<jobid>.out
grep "\[SEED_SUMMARY\]" logs/<jobname>_<jobid>.out
grep "\[FINAL\]" logs/<jobname>_<jobid>.out
watch -n 5 squeue -u $USER
```

Array logs use `%A_%a` in the filename:

```bash
tail -f logs/tunedgcn-pgsmp_<array_jobid>_<taskid>.out
```

## Output Files

Each run writes:

```text
results/tunedgcn_pgsmp/YYYYMMDD_HHMMSS/config.json
results/tunedgcn_pgsmp/YYYYMMDD_HHMMSS/epoch_logs.csv
results/tunedgcn_pgsmp/YYYYMMDD_HHMMSS/seed_summary.csv
results/tunedgcn_pgsmp/YYYYMMDD_HHMMSS/final_summary.json
```

The main scientific number is `test_at_best_val`, not `best_raw_test`.

Build a compact final table:

```bash
python tools/compare_results.py \
  results/tunedgcn_pgsmp/<baseline_run>/final_summary.json \
  results/tunedgcn_pgsmp/<pgsmp_run>/final_summary.json
```

## Cache

P-GSMP feature caches live under:

```text
cache/pgsmp/
```

Cache names encode dataset, edge direction, undirected setting, preprocessing self-loop mode, norm, depth, alpha, self mode, feature shape, edge count, and a graph fingerprint. Weight caches live under:

```text
cache/pgsmp/weights/
```

Do not use `PGSMP_FORCE_RECOMPUTE=1` unless you intentionally want to overwrite caches.

## Interpretation

This experiment tests whether timestamp-balanced neighbor feature preprocessing improves a tuned GCN without changing the GCN architecture.

- If `GCN+P-GSMP > GCN`, preprocessing node features with timestamp-balanced neighbor information helps chronological generalization.
- If `GCN+P-GSMP < GCN`, P-GSMP may oversmooth input features or remove useful citation-time signal before the GCN can learn.
- If `GCN+P-GSMP ~= GCN`, the tuned GCN may already extract similar information through ordinary message passing, or the one-step preprocessing may be redundant.

Because P-GSMP is cached and applied once, it is cheaper than layer-wise GSMP and useful as a low-cost ablation.

## Modified Files

Patched upstream files:

- `upstream/tunedGNN/large_graph/main-arxiv.py`
- `upstream/tunedGNN/large_graph/lg_parse.py`
- `upstream/tunedGNN/large_graph/dataset.py`

Added files:

- `upstream/tunedGNN/large_graph/pgsmp.py`
- `tests/test_pgsmp.py`
- `tools/check_baseline_gate.py`
- `tools/compare_results.py`
- `scripts/run_tunedgcn_pgsmp_experiments.sh`
- `slurm/reproduce_tunedgcn_baseline.sbatch`
- `slurm/run_tunedgcn_pgsmp.sbatch`
- `slurm/run_tunedgcn_pgsmp_all_array.sbatch`
- `README_PGSMP_TUNEDGCN_OGBN_ARXIV.md`
