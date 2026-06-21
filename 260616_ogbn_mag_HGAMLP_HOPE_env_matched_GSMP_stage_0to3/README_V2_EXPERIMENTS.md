# ogbn-mag HGAMLP-HOPE Env-Matched GSMP Stage 0-3

This workspace is for the priority-2 env-matched follow-up after the completed stage-0 run in:

`../260615_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0`

Baseline seeds 1, 2, and 3 are already available from that completed run, so this workspace is configured to run only GSMP seeds 1, 2, and 3.

## Target Run

```bash
cd /users/PAS1289/jyp531/GSMP-research/260616_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3
sbatch slurm/run_priority2_allstages_gsmp_all_hops.sbatch
```

Configuration:

- Run name: `envmatched_p2_gsmp_stage0to3_all_hops`
- Method/seeds: `gsmp:1 gsmp:2 gsmp:3`
- GSMP stages: `0-3`
- GSMP direct `P-P` hops: all eligible hops
- Impact scope: feature and label propagation
- Conda env: `hope_official`
- Cache: `impact_cache_official_env`, symlinked to the completed env-matched stage-0 workspace

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
