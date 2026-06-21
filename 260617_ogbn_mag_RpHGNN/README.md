# 260617 ogbn-mag RpHGNN Pipeline

Experiment root for:

- `baseline`: official `RpHGNN+LP+CR (LINE embs)` settings.
- `smp_layerwise`: baseline plus corrected layerwise SMP on paper-paper citation propagation.
- `gsmp_first_layer`: baseline plus first-layer target-side GSMP on paper-paper citation propagation.
- `gsmp_paper_added`: corrected GSMP mode for RpHGNN paper additions. It keeps direct `P-P`
  GSMP and additionally applies exact source-year-bucketed GSMP to effective `P-A-P`
  and `P-F-P` propagation whenever the destination `P` representation is formed.

Main code lives in `RpHGNN/`, cloned from `https://github.com/CrawlScript/RpHGNN`.

## First Commands

```bash
cd /users/PAS1289/jyp531/GSMP-research/260617_ogbn_mag_RpHGNN/RpHGNN

bash scripts/precompute_rphgnn_cache.sh

sbatch scripts/run_ogbn_mag_rphgnn_smp_gsmp.slurm baseline_smoke
sbatch scripts/run_ogbn_mag_rphgnn_smp_gsmp.slurm smp_layerwise_smoke
sbatch scripts/run_ogbn_mag_rphgnn_smp_gsmp.slurm gsmp_first_layer_smoke
sbatch scripts/run_ogbn_mag_rphgnn_smp_gsmp.slurm gsmp_paper_added_smoke
```

For environment-variable resource overrides, use the submit wrapper:

```bash
CONCURRENCY=1 MEM=48G TIME_LIMIT=12:00:00 GRES=gpu:1 \
  bash scripts/submit_ogbn_mag_rphgnn_smp_gsmp.sh baseline_smoke
```

Launch full 10-seed jobs only after smoke logs and cache behavior look correct:

```bash
sbatch scripts/run_ogbn_mag_rphgnn_smp_gsmp.slurm baseline
sbatch scripts/run_ogbn_mag_rphgnn_smp_gsmp.slurm smp_layerwise
sbatch scripts/run_ogbn_mag_rphgnn_smp_gsmp.slurm gsmp_first_layer
sbatch scripts/run_ogbn_mag_rphgnn_smp_gsmp.slurm gsmp_paper_added
```

Monitor and aggregate:

```bash
bash scripts/monitor.sh logs/ogbn_mag_rphgnn/<run_dir>
python scripts/aggregate_results.py --log-dir logs/ogbn_mag_rphgnn/<run_dir>
```

See `diagnostics/reproduction_report.md` for the upstream commit, official settings, and the upstream seed-script discrepancy.
