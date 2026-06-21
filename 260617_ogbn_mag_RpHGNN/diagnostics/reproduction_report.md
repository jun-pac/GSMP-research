# RpHGNN ogbn-mag Reproduction Notes

Created: 2026-06-16

## Upstream

- Repository: `https://github.com/CrawlScript/RpHGNN`
- Local clone: `RpHGNN`
- Commit inspected: `1a1779a747a28ac8d936280a6b96951636183965`
- Official leaderboard script: `RpHGNN/scripts/run_leaderboard_OGBN-MAG.sh`

## Official Leaderboard Settings Preserved

The new launch wrapper preserves the official leaderboard arguments:

- `--dataset mag`
- `--method rphgnn`
- `--use_nrl True`
- `--use_label True`
- `--even_odd all`
- `--train_strategy cl`
- `--use_input True`
- `--input_drop_rate 0.1`
- `--drop_rate 0.4`
- `--hidden_size 512`
- `--squash_k 3`
- `--num_epochs 500`
- `--max_patience 50`
- `--embedding_size 512`
- `--use_all_feat True`

The code path also preserves the repository's leaderboard-specific LP+CR behavior:

- `running_leaderboard_mag = dataset == "mag" and use_label`
- scheduler gamma `0.99`
- `num_views = 3`
- consistency regularization rate `cl_rate = 0.6`
- official OGB evaluator `Evaluator("ogbn-mag")`

## Seed Discrepancy In Upstream Script

The README says `scripts/run_leaderboard_OGBN-MAG.sh` runs seeds `0` through `9`, but the cloned script currently contains:

```bash
for SEED in $(seq 0 9)
do
SEED=11
```

The new SLURM array uses seeds `0-9` to match the README, OGB-style reporting, and the experiment request. This is a documented wrapper correction, not a hyperparameter change.

## Variants

- `baseline`: original RpHGNN propagation, LP, CR, and LINE/NRL embeddings.
- `smp_layerwise`: corrected target-normalized SMP on every direct paper-paper citation propagation step.
- `gsmp_first_layer`: target-side GSMP on only the first direct paper-paper citation propagation step.
- `gsmp_paper_added`: corrected target-side GSMP for RpHGNN paper additions. Direct
  `P-P` steps use cached edge weights, and effective `P-A-P`/`P-F-P` paper outputs are
  recomputed by source-paper-year buckets when the destination paper representation is formed.

Default temporal relation scope is `paper_paper_only`.

## Resource Policy

- Smoke runs should be launched first.
- Full jobs use one seed per GPU.
- Array concurrency defaults to `%1`.
- `cache/mag.p` is shared across seeds and variants.
- Temporal weights are cached once at `RpHGNN/cache/mag_temporal_weights.pt`.
- LINE/NRL pretraining is not launched by default by `precompute_rphgnn_cache.sh`; set `RUN_LINE_PRECOMPUTE=1` only when intentionally spending that time.

## Commands

```bash
cd /users/PAS1289/jyp531/GSMP-research/260617_ogbn_mag_RpHGNN/RpHGNN

# Precompute temporal weights and verify LINE cache.
bash scripts/precompute_rphgnn_cache.sh

# Smoke tests.
sbatch scripts/run_ogbn_mag_rphgnn_smp_gsmp.slurm baseline_smoke
sbatch scripts/run_ogbn_mag_rphgnn_smp_gsmp.slurm smp_layerwise_smoke
sbatch scripts/run_ogbn_mag_rphgnn_smp_gsmp.slurm gsmp_first_layer_smoke
sbatch scripts/run_ogbn_mag_rphgnn_smp_gsmp.slurm gsmp_paper_added_smoke

# Full 10-seed runs after smoke passes.
sbatch scripts/run_ogbn_mag_rphgnn_smp_gsmp.slurm baseline
sbatch scripts/run_ogbn_mag_rphgnn_smp_gsmp.slurm smp_layerwise
sbatch scripts/run_ogbn_mag_rphgnn_smp_gsmp.slurm gsmp_first_layer
sbatch scripts/run_ogbn_mag_rphgnn_smp_gsmp.slurm gsmp_paper_added

# Optional env-var resource override wrapper.
CONCURRENCY=1 MEM=48G TIME_LIMIT=12:00:00 GRES=gpu:1 \
  bash scripts/submit_ogbn_mag_rphgnn_smp_gsmp.sh baseline

# Monitor and aggregate.
bash scripts/monitor.sh logs/ogbn_mag_rphgnn/<run_dir>
python scripts/aggregate_results.py --log-dir logs/ogbn_mag_rphgnn/<run_dir>
```

## Status

No full reproduction has been launched from this report. Run baseline smoke first, then the 10-seed baseline. Only add full SMP/GSMP runs after the baseline reproduction is acceptable.
