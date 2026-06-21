# ogbn-arxiv SimTeG/TAPE GraphSAGE Temporal Modes

This directory runs a cost-conscious comparison of:

- `baseline`: SimTeG/TAPE or OGB features + GraphSAGE
- `smp`: baseline + SMP edge weights
- `ump`: baseline + UMP temporal edge filtering
- `gsmp`: baseline + GSMP neighbor-year balancing

The code never fine-tunes a language model. It reuses local cached SimTeG/TAPE embeddings when found, accepts an explicit `--features_path`, and otherwise prints:

```text
WARNING: using default ogbn-arxiv features, not SimTeG/TAPE embeddings.
```

## Quick Start

```bash
cd /users/PAS1289/jyp531/GSMP-research/ogbn_arxiv_temporal
mkdir -p logs results checkpoints

# Optional: point to cached SimTeG/TAPE features.
export FEATURES_PATH=/path/to/simteg_tape_ogbn_arxiv.pt

# Cheap correctness check first.
sbatch slurm/smoke_test_arxiv_sage_smp.sh

# Monitor a running task.
tail -f logs/arxiv_sage_smoke_<job-id>_<array-id>.out
```

Run the full single-seed comparison only after the smoke test succeeds:

```bash
sbatch slurm/run_arxiv_sage_modes.sh
```

The optional multi-seed run uses seeds `1 2 3` and should be submitted only after the single-seed results look healthy:

```bash
sbatch slurm/run_arxiv_sage_multiseed_optional.sh
```

## SMP/GSMP Sweep

The sweep script keeps the fixed E5 mini-batch baseline recipe and varies only SMP/GSMP hyperparameters. The default grid launches 36 jobs:

```text
mode = smp, gsmp
label_smoothing = 0.2, 0.3, 0.4
dropout = 0.3, 0.4
weight_decay = 1e-6, 2e-6, 4e-6
lr = 0.01
batch_size = 4096
num_neighbors = 15,10
seed = 1
```

Submit:

```bash
cd /users/PAS1289/jyp531/GSMP-research/ogbn_arxiv_temporal
mkdir -p logs
export FEATURES_PATH=/users/PAS1289/jyp531/GSMP-research/SimTeG/out/ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt
sbatch slurm/sweep_arxiv_sage_smp_gsmp.sh
```

Monitor:

```bash
squeue -u jyp531
tail -f logs/arxiv_sage_sweep_<job-id>_<array-id>.out
```

Aggregate completed sweep runs:

```bash
../.venv/bin/python aggregate_sweep_results.py --sweep_dir outputs/e5_smp_gsmp_sweep
```

For a wider second-stage sweep, override grids before submission and pass a matching Slurm array range:

```bash
export LR_GRID="0.005 0.01"
export SEED_GRID="1 2 3"
sbatch --array=0-215%8 slurm/sweep_arxiv_sage_smp_gsmp.sh
```

## Dry Run

Use `--dry_run` to load data, validate features, build graph weights, print preprocessing statistics, and exit before training:

```bash
python -u train_arxiv_sage_temporal.py \
  --mode gsmp \
  --features_path /path/to/ogbn_arxiv_embeddings.npy \
  --dry_run \
  --save_dir .
```

## Direct Python Run

```bash
python -u train_arxiv_sage_temporal.py \
  --mode baseline \
  --features_path /path/to/simteg_tape_ogbn_arxiv.pt \
  --seed 1 \
  --runs 1 \
  --epochs 300 \
  --patience 50 \
  --eval_every 10 \
  --log_every 10 \
  --hidden_channels 256 \
  --num_layers 3 \
  --device cuda:0 \
  --save_dir .
```

If `--features_path` is omitted, the runner first checks common local SimTeG output paths under `../SimTeG/out`, `../lambda_out`, `../out`, and `../data`.

## Outputs

Each seed writes:

```text
results/ogbn_arxiv_simteg_tape_sage_<mode>_seed<seed>.csv
results/run_summary_<mode>_seed<seed>.json
```

The runner also updates:

```text
results/summary.json
```

Checkpoints are written only when `--save_checkpoint` is passed:

```text
checkpoints/ogbn_arxiv_simteg_tape_sage_<mode>_seed<seed>_best.pt
```

Aggregate completed results:

```bash
python aggregate_results.py --results_dir results
```

## Notes

- Default training is full-batch GraphSAGE, which is stable for ogbn-arxiv with precomputed node features.
- Full-batch aggregation is chunked by default; set `AGGREGATION_CHUNK_SIZE` in Slurm or pass `--aggregation_chunk_size` to tune memory/speed.
- `--training_mode mini` enables PyG `NeighborLoader` with `--batch_size` and `--num_neighbors`, but full-batch should be tried first.
- The default Slurm scripts request one GPU per array task and do not launch multiple seeds unless you submit the optional script.
