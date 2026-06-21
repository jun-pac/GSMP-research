# GSMP on tunedGNN GCN for ogbn-arxiv

This folder contains a shallow official tunedGNN checkout plus a scoped GSMP patch for the large-graph ogbn-arxiv GCN experiment.

Upstream:

- Repository: https://github.com/LUOyk1999/tunedGNN
- Local commit: `23f9604e8b13a9a6d3faa2f691cd844006979153`
- Official large-graph command inspected: `large_graph/arxiv.sh`

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

The code loads ogbn-arxiv with OGB `NodePropPredDataset`, uses the official OGB split, converts citation edges with `to_undirected`, removes existing self-loops, then adds one self-loop per node. The GCN uses PyG `GCNConv(cached=False, normalize=True)`, residual linear paths, batch norm when `--bn` is set, Adam, parser default `weight_decay=5e-4`, parser default `in_dropout=0.15`, and parser default `dropout=0.5`.

## Added GSMP Behavior

GSMP is disabled unless `--use-gsmp` is passed. The baseline call path still passes no `edge_weight` to `GCNConv`.

For GSMP, weights are computed on the final PyG edge representation:

```text
src = edge_index[0]
dst = edge_index[1]
count[dst, node_year[src]]
```

Default mode:

```bash
--gsmp-norm scale_preserve
--gcn-gsmp-mode weighted_gcn_norm
--gsmp-apply all_layers
```

This passes GSMP weights into PyG `GCNConv` and lets `GCNConv` apply standard weighted GCN normalization. Optional modes are:

- `--gsmp-norm strict`
- `--gcn-gsmp-mode post_norm_message_scale`
- `--gcn-gsmp-mode strict_gsmp_mean`
- `--gsmp-apply first_layer`

`--gsmp-apply all_layers` sends the GSMP edge weights into every GCN layer.
`--gsmp-apply first_layer` sends GSMP edge weights only into layer 0; layers
1-4 are vanilla PyG GCN layers. This is the intended "GSMP preprocessing, then
GCN" ablation.

Weights are cached in `260612/cache/gsmp` by default, with filenames encoding dataset, edge direction, undirected/self-loop state, GSMP norm, edge count, GCN-GSMP mode, and a graph fingerprint.

## Environment

The upstream README reports testing with Python 3.7, PyTorch 1.12.1, PyG 2.3.1, and DGL 1.0.2. The SLURM scripts have an editable section:

```bash
source ~/.bashrc
conda activate tunedgnn
```

Edit `CONDA_ENV`, or edit that section directly, for your cluster.

## Static Checks

Run these before spending GPU time:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260612
python -m py_compile upstream/tunedGNN/large_graph/gsmp.py
python -m py_compile upstream/tunedGNN/large_graph/lg_model.py
python -m py_compile upstream/tunedGNN/large_graph/lg_parse.py
python -m py_compile upstream/tunedGNN/large_graph/eval.py
python -m py_compile upstream/tunedGNN/large_graph/main-arxiv.py
python tests/test_gsmp_weights.py
```

## One-Command Smoke Test

This runs 3 epochs of baseline and 3 epochs of GSMP for one seed:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260612
bash scripts/run_tunedgcn_gsmp_experiments.sh smoke
```

Useful overrides:

```bash
CPU=1 bash scripts/run_tunedgcn_gsmp_experiments.sh smoke
EPOCHS=5 DEVICE=0 bash scripts/run_tunedgcn_gsmp_experiments.sh smoke
```

## Baseline First

Reproduce tunedGNN GCN before running full GSMP:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260612
SEEDS="0 1 2" bash scripts/run_tunedgcn_gsmp_experiments.sh baseline
```

Or with SLURM:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260612
SEED=0 sbatch slurm/reproduce_tunedgcn_baseline.sbatch
```

Default full scripts use `EVAL_EVERY=10` to save time. For stricter reproduction of the official selection protocol, use `EVAL_EVERY=1 LOG_EVERY=10`.

Best faithful resource-conscious baseline run:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260612
env -u CONDA_ENV SEED=0 EPOCHS=2000 EVAL_EVERY=1 LOG_EVERY=50 \
  sbatch -A PAS1289 --time=01:30:00 slurm/reproduce_tunedgcn_baseline.sbatch
```

This preserves the official 2000 epochs and every-epoch validation selection.
The patched logger still writes every evaluated epoch to `epoch_logs.csv`, but
only prints `[RESULT]`/console progress every `LOG_EVERY` epochs plus epoch 1
and the final epoch.

Expected leaderboard target:

- tunedGNN GCN validation accuracy: `0.7447 +/- 0.0014`
- tunedGNN GCN test accuracy: `0.7360 +/- 0.0018`

If the baseline is far below target, stop and debug before running GSMP:

```bash
python tools/check_baseline_gate.py results/tunedgcn_gsmp/<run>/final_summary.json
```

## Run GCN+GSMP

After the baseline is healthy:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260612
SEEDS="0 1 2" bash scripts/run_tunedgcn_gsmp_experiments.sh gsmp
```

SLURM array, one seed per task:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260612
sbatch slurm/run_tunedgcn_gsmp.sbatch
```

The GSMP SLURM script defaults to a single array task to avoid accidental extra
GPU allocations. For five seeds, submit with an explicit array:

```bash
SEEDS="0 1 2 3 4" sbatch --array=0-4%1 slurm/run_tunedgcn_gsmp.sbatch
```

First-layer/preprocessing GSMP, then vanilla GCN layers:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260612
env -u CONDA_ENV SEEDS="0" EPOCHS=2000 EVAL_EVERY=1 LOG_EVERY=50 \
  sbatch -A PAS1289 --time=01:30:00 slurm/run_tunedgcn_gsmp_first_layer.sbatch
```

Equivalent local/script override:

```bash
GSMP_APPLY=first_layer SEEDS="0" EPOCHS=2000 EVAL_EVERY=1 LOG_EVERY=50 \
  bash scripts/run_tunedgcn_gsmp_experiments.sh gsmp
```

Baseline plus GSMP array:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260612
sbatch slurm/run_tunedgcn_all_array.sbatch
```

The all-array script defaults to two tasks, baseline and GSMP for seed 0. For
five seeds across both methods, submit with:

```bash
SEEDS="0 1 2 3 4" sbatch --array=0-9%1 slurm/run_tunedgcn_all_array.sbatch
```

Optional ablations:

```bash
GSMP_NORM=strict bash scripts/run_tunedgcn_gsmp_experiments.sh gsmp
GCN_GSMP_MODE=post_norm_message_scale bash scripts/run_tunedgcn_gsmp_experiments.sh gsmp
GCN_GSMP_MODE=strict_gsmp_mean bash scripts/run_tunedgcn_gsmp_experiments.sh gsmp
GSMP_APPLY=first_layer bash scripts/run_tunedgcn_gsmp_experiments.sh gsmp
```

## Monitor Logs

```bash
tail -f logs/<jobname>_<jobid>.out
grep "\[RESULT\]" logs/<jobname>_<jobid>.out
grep "\[SEED_SUMMARY\]" logs/<jobname>_<jobid>.out
grep "\[FINAL\]" logs/<jobname>_<jobid>.out
watch -n 5 squeue -u $USER
```

Array logs use `%A_%a` in the filename:

```bash
tail -f logs/tunedgcn-all_<array_jobid>_<taskid>.out
```

## Output Files

Each run writes:

```text
results/tunedgcn_gsmp/YYYYMMDD_HHMMSS/config.json
results/tunedgcn_gsmp/YYYYMMDD_HHMMSS/epoch_logs.csv
results/tunedgcn_gsmp/YYYYMMDD_HHMMSS/seed_summary.csv
results/tunedgcn_gsmp/YYYYMMDD_HHMMSS/final_summary.json
```

The main scientific number is `test_at_best_val`, not `best_raw_test`.
For GSMP runs, check `gsmp_apply` in `config.json`/`final_summary.json`.

Build a compact final table:

```bash
python tools/compare_results.py \
  results/tunedgcn_gsmp/<baseline_run>/final_summary.json \
  results/tunedgcn_gsmp/<gsmp_run>/final_summary.json
```

## Interpretation

This is the cleanest ogbn-arxiv GSMP ablation here because GCN already uses linear aggregation. There is no attention-removal confound.

- If `GCN+GSMP > GCN`, GSMP likely improves chronological generalization by reducing year-wise neighbor imbalance.
- If `GCN+GSMP < GCN`, GSMP may be removing useful citation-time signal or interacting poorly with symmetric GCN normalization.
- If `GCN+GSMP ~= GCN`, the tuned GCN may already be robust to timestamp imbalance.

## Modified Files

Patched upstream files:

- `upstream/tunedGNN/large_graph/main-arxiv.py`
- `upstream/tunedGNN/large_graph/lg_model.py`
- `upstream/tunedGNN/large_graph/lg_parse.py`
- `upstream/tunedGNN/large_graph/eval.py`

Added files:

- `upstream/tunedGNN/large_graph/gsmp.py`
- `tests/test_gsmp_weights.py`
- `scripts/run_tunedgcn_gsmp_experiments.sh`
- `slurm/reproduce_tunedgcn_baseline.sbatch`
- `slurm/run_tunedgcn_gsmp.sbatch`
- `slurm/run_tunedgcn_gsmp_first_layer.sbatch`
- `slurm/run_tunedgcn_all_array.sbatch`
- `tools/check_baseline_gate.py`
- `tools/compare_results.py`
