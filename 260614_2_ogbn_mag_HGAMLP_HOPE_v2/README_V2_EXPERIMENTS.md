# ogbn-mag HGAMLP-HOPE GSMP v2 experiments

This folder is a clean v2 copy of `260614_ogbn_mag_HGAMLP_HOPE`.
It adds stage-aware impact controls without touching the currently running SLURM job.

## New controls

- `--impact-stages`: stages where SMP/GSMP is active. Examples: `0`, `0-3`, `all`, `none`.
- `--impact-apply-to`: choose whether SMP/GSMP changes `feature`, `label`, or `both` propagation.
- `GSMP_LAYER_MODE=first`: GSMP only for the first direct `P-P` propagation hop, mainly `P->PP`.
- `GSMP_LAYER_MODE=all`: GSMP for all eligible direct `P-P` propagation hops, including `P->PP`, `PP->PPP`, `PA->PPA`, and `PF->PPF`.

When `--impact-stages 0 --impact-apply-to both` is used, stage 0 uses impact-propagated feature tensors and label propagation; stages 1-3 use baseline feature tensors and baseline label propagation. When `--impact-apply-to feature` is used, only the selected stages' feature tensors use impact propagation and all label propagation stays baseline.

## Priority scripts

Priority 0:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260614_2_ogbn_mag_HGAMLP_HOPE_v2
sbatch slurm/run_priority0_stage0_feature_gsmp_all_hops.sbatch
```

- GSMP stages: `0`
- GSMP apply to: feature propagation only
- GSMP direct `P-P` hops: all eligible hops
- Seeds: `1 2 3`

Priority 1:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260614_2_ogbn_mag_HGAMLP_HOPE_v2
sbatch slurm/run_priority1_stage0_gsmp_all_hops.sbatch
```

- GSMP stages: `0`
- GSMP apply to: feature and label propagation
- GSMP direct `P-P` hops: all eligible hops
- Seeds: `1 2 3`

Priority 2:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260614_2_ogbn_mag_HGAMLP_HOPE_v2
sbatch slurm/run_priority2_allstages_gsmp_all_hops.sbatch
```

- GSMP stages: `0-3`
- GSMP direct `P-P` hops: all eligible hops
- Seeds: `1 2 3`

Priority 3:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260614_2_ogbn_mag_HGAMLP_HOPE_v2
sbatch slurm/run_priority3_stage0_gsmp_first_hop.sbatch
```

- GSMP stages: `0`
- GSMP direct `P-P` hops: first hop only
- Seeds: `1 2 3`

## Monitoring

Replace `priority1_stage0_gsmp_all_hops` with the run name you launched:

```bash
tail -f results/priority1_stage0_gsmp_all_hops_live_progress.tsv
```

```bash
watch -n 10 'tail -n 20 results/priority1_stage0_gsmp_all_hops_live_progress.tsv'
```

Logs are under `logs/<job_id>/`.
