# ogbn-mag HGAMLP-HOPE Env-Matched GSMP Stage 0-3 No Stage3 Earlystop

This workspace is the `no_stage3_earlystop` follow-up. It disables early stopping
only for GSMP stage 3, keeps the original early-stopping behavior for baselines,
and keeps per-stage best checkpoints plus raw predictions for reuse.

Seeds are run with one seed per Slurm array task. The production arrays use
`%4`, not `%1`, so seeds 0, 1, 2, and 3 are not intentionally serialized.

This workspace is for the priority-2 env-matched follow-up after the completed stage-0 run in:

`../260615_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0`

Baseline seeds 1, 2, and 3 are already available from that completed run, so this workspace is configured to run only GSMP seeds 1, 2, and 3.

## Target Runs

```bash
cd /users/PAS1289/jyp531/GSMP-research/260621_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3_no_stage3_earlystop
sbatch slurm/run_priority1_stage0_gsmp_all_hops_no_stage3_earlystop.sbatch
sbatch slurm/run_priority2_allstages_gsmp_all_hops_no_stage3_earlystop.sbatch
sbatch slurm/run_priority3_stage0_gsmp_first_hop_no_stage3_earlystop.sbatch
```

Configuration:

- Run names include `no_stage3_earlystop`
- Method/seeds: `gsmp:0 gsmp:1 gsmp:2 gsmp:3`
- GSMP no early stopping: stage `3` only
- Per-stage checkpoints: `checkpoints/<run_name>_gsmp_seed<seed>/best_<stage>.pkl`
- Impact scope: feature and label propagation
- Conda env: `hope_official`
- Cache: `impact_cache_official_env`, symlinked to the completed env-matched stage-0 workspace

## Grid Search Fallback

If the first no-stage3-earlystop run is not good enough, submit:

```bash
sbatch slurm/run_gsmp_grid_no_stage3_earlystop.sbatch
```

The grid covers the three priority settings, seeds 0-3, learning rate, dropout,
pseudo-label threshold, and prototype-separation lambda. It uses one grid point
per array task with `%12` parallelism.

## Monitoring

```bash
tail -f results/envmatched_p2_gsmp_stage0to3_all_hops_live_progress.tsv
```

```bash
watch -n 10 'tail -n 20 results/envmatched_p2_gsmp_stage0to3_all_hops_live_progress.tsv'
```

Slurm output files go to `logs/priority2_<job_id>_<task_id>.out` and per-run logs go under `logs/<job_id>/`.

## Notes

The old stage-0 summary is in:

`../260615_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0/STAGE0_RESULTS_SUMMARY.md`
