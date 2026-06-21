# 260617_2_ogbn_mag_PSHGCN

Official source clone:

```bash
/users/PAS1289/jyp531/GSMP-research/260617_2_ogbn_mag_PSHGCN/PSHGCN
```

Work from the ogbn-mag directory:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260617_2_ogbn_mag_PSHGCN/PSHGCN/ogbn-mag
```

The leaderboard baseline is the repository's ComplEx + multi-stage command:

```bash
python main.py --extra_emb --stage 4 --layers_x 2
```

The Slurm pipeline preserves that setting for full runs and adds:

```bash
baseline
smp_layerwise
gsmp_first_linear
gsmp_first_linear_stage0
baseline_smoke
smp_layerwise_smoke
gsmp_first_linear_smoke
gsmp_first_linear_stage0_smoke
```

`gsmp_first_linear` uses GSMP-weighted first-linear feature inputs for every PSHGCN self-training stage.
`gsmp_first_linear_stage0` uses GSMP-weighted first-linear feature inputs only for stage 0, then switches stages 1-3 back to baseline propagated features. Label propagation remains baseline for both GSMP variants.

Before launching GPU jobs, place the official ComplEx files in:

```bash
PSHGCN/ogbn-mag/complEx/author.pt
PSHGCN/ogbn-mag/complEx/field_of_study.pt
PSHGCN/ogbn-mag/complEx/institution.pt
```

The clone currently includes only `institution.pt`; the runner exits before GPU training if the other two files are missing.

Cheap checks:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260617_2_ogbn_mag_PSHGCN/PSHGCN/ogbn-mag
bash scripts/precompute_pshgcn_cache.sh
```

Smoke after ComplEx files are present:

```bash
SMOKE_STAGE=1 EPOCHS=1 MEM=64G SMOKE_TIME_LIMIT=01:00:00 \
bash scripts/submit_ogbn_mag_pshgcn_smp_gsmp.sh baseline_smoke

bash scripts/monitor_latest.sh
```

Full single-seed pilot after smoke passes:

```bash
ARRAY_SPEC=0-0 EPOCHS=50 STAGE=4 LAYERS_X=2 MEM=64G TIME_LIMIT=02:00:00 \
bash scripts/submit_ogbn_mag_pshgcn_smp_gsmp.sh baseline
```

Use the 10-seed array only after the smoke and single-seed pilot are clean.
