# HGAMLP-HOPE Impact Experiments on ogbn-mag

This folder patches the official HOPE reproduction code from:

```text
https://github.com/Dpens/HOPE
```

The official code is under `HOPE/`. The HOPE prediction head is preserved; SMP/GSMP are only added to the meta-path propagation preprocessing path.

## What Was Added

- `HOPE/impact.py`: SMP/GSMP weighting, target-normalized weighted DGL propagation, live progress logging, toy checks.
- `HOPE/training.py`: new CLI flags, propagation cache, `RESULT ...` logs, append-only `results/live_progress.tsv`.
- `HOPE/utils.py`: minimal patch to call weighted propagation only when `--impact-method smp|gsmp`.
- `slurm/`: smoke, pilot, full, and generic array scripts.
- `scripts/summarize_results.py`: creates `results/summary.tsv`, `results/per_seed.tsv`, and `results/summary.md`.

## Data And Embeddings

The project is configured to reuse existing local files instead of copying large data:

- OGB data: `/users/PAS1289/jyp531/GSMP-research/data/ogbn_mag`
- pretrained Line embedding `mag.p`: `/users/PAS1289/jyp531/GSMP-research/HGAMLP_MAG/mag.p`
- sparse tools: `/users/PAS1289/jyp531/GSMP-research/sparse_tools`

Refresh symlinks with:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260614_ogbn_mag_HGAMLP_HOPE
./scripts/prepare_data_links.sh
```

## Environment

The official target environment is:

```text
Python 3.8.20
torch==1.13.0+cu117
dgl-cu117==0.9.1.post1
ogb==1.3.6
torch_sparse==0.6.18
```

The Slurm scripts try `CONDA_ENV=hope` first. Override it if your environment has a different name:

```bash
CONDA_ENV=your_env sbatch slurm/run_smoke.sbatch
```

## Smoke First

Smoke uses methods `none`, `smp`, `gsmp`, seed `1`, and stages `2 2 2 2`.

```bash
cd /users/PAS1289/jyp531/GSMP-research/260614_ogbn_mag_HGAMLP_HOPE
sbatch slurm/run_smoke.sbatch
```

Monitor:

```bash
tail -f logs/latest/*.log
tail -f results/live_progress.tsv
```

Smoke is only for code, tensor shape, logging, and Slurm sanity checks. Do not report smoke accuracy.

## Pilot Second

Pilot uses methods `none`, `smp`, `gsmp`, seeds `1 2 3`, stages `60 60 60 60`, and eval every `5` epochs by default.

```bash
sbatch slurm/run_pilot.sbatch
```

This is meant to check whether SMP/GSMP is promising before spending full GPU hours.

## Budget Full Run

This is the recommended cost-controlled run for the current plan: verify the baseline with seed `1`, then run `smp` and `gsmp` for seeds `1 2 3`.

```bash
sbatch slurm/run_budget_full.sbatch
```

Defaults:

- jobs: `none:1 smp:1 smp:2 smp:3 gsmp:1 gsmp:2 gsmp:3`
- stages: `300 300 300 300`
- eval every `5` epochs
- patience `100`
- feature propagation caches reused from `impact_cache/`
- lower-overhead epoch cleanup: `--gc-every 10 --no-empty-cache-every-epoch`

For a more paper-faithful but slightly more expensive version, keep the same 7-job list and evaluate every epoch:

```bash
EVAL_EVERY=1 sbatch slurm/run_budget_full.sbatch
```

## Full Last

Full uses official leaderboard-style settings:

```bash
sbatch slurm/run_full.sbatch
```

Equivalent baseline command:

```bash
cd HOPE
python -u training.py \
  --aggregation HGAMLP-HOPE \
  --impact-method none \
  --label-residual \
  --alpha 0.5 \
  --similarity-threshold 0.6 \
  --lower-bound 0.5 \
  --upper-bound 3 \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --use-sparse-tools
```

SMP:

```bash
python -u training.py \
  --aggregation HGAMLP-HOPE \
  --impact-method smp \
  --impact-apply-to both \
  --label-residual \
  --alpha 0.5 \
  --similarity-threshold 0.6 \
  --lower-bound 0.5 \
  --upper-bound 3 \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --use-sparse-tools
```

GSMP first-layer-only:

```bash
python -u training.py \
  --aggregation HGAMLP-HOPE \
  --impact-method gsmp \
  --impact-apply-to both \
  --impact-gsmp-first-layer-only \
  --label-residual \
  --alpha 0.5 \
  --similarity-threshold 0.6 \
  --lower-bound 0.5 \
  --upper-bound 3 \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --use-sparse-tools
```

## Summarize

```bash
python scripts/summarize_results.py --log-dir logs --out results
```

Outputs:

```text
results/summary.tsv
results/per_seed.tsv
results/summary.md
```

The main metric is `test_at_best_valid`, not best test accuracy.

## Implementation Notes

- `--impact-method none` follows the original `fn.copy_u` + `fn.mean` path in `utils.py::hg_propagate`.
- SMP/GSMP use target-side denominator normalization.
- Direct `P-P` propagation is weighted inside `utils.py::hg_propagate`.
- Effective `P-A-P` and `P-F-P` paper-to-paper meta-paths are recomputed by sparse source-year buckets in `HOPE/impact.py`, so weights use source paper year and target paper year without materializing a dense paper-paper matrix.
- SMP uses the corrected rule:

```python
delta = abs(time[u] - time[v])
radius = min(time[v] - t_min, t_max - time[v])
raw_weight = 2.0 if (delta == 0 or delta > radius) else 1.0
```

- GSMP uses target-based inverse-frequency timestamp weighting:

```python
raw_weight[u, v] = 1.0 / C_v[time[u]]
```

- GSMP is first-layer-only by default. Later eligible paper-paper sparse multiplications fall back to official HGAMLP propagation.
- No labels/classes are used to construct SMP/GSMP weights.
- No dense `N x N` adjacency is materialized.
- Feature propagation is cached under `impact_cache/` with method, scope, hops, GSMP mode, and implementation version in the filename.

Run the toy checks:

```bash
cd HOPE
python impact.py
```
