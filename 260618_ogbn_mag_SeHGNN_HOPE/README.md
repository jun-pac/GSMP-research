# SeHGNN-HOPE + GSMP on ogbn-mag

This workspace compares:

- `baseline`: SeHGNN + HOPE on ogbn-mag
- `gsmp`: GSMP paper-stack feature propagation + SeHGNN + HOPE on ogbn-mag

The baseline target is the HGAMLP-HOPE Table 2 SeHGNN + HOPE result:
`57.95 +/- 0.14` test accuracy over seeds `1 2 3 4 5 6 7 8 9 10`.

The code starts from the tracked clean `Dpens/HOPE` files already present in the local HOPE clone and keeps the baseline path as the original `copy_u + mean` propagation unless `--gsmp-first-layer` is passed.

## GSMP Scope

`GSMP paper-stack` matches the prior `priority1` meaning from `260614_2_ogbn_mag_HGAMLP_HOPE_v2`: every eligible direct `P-P` feature-propagation step in the compact `P/A/I/F` graph. It is not just `P -> PP`, and it is not limited to hop 1.

The guarded insertion point is `HOPE/utils.py:hg_propagate`. GSMP is applied only when all are true:

- `--gsmp-first-layer` is set
- this is feature propagation, not label propagation
- source and destination node types are both target paper `P`

With the default `--gsmp-scope paper-stack`, this catches all direct paper-paper stack updates. For `--num-hops 2`, direct GSMP is applied to `P->PP`, `PA->PPA`, `PF->PPF`, and `PP->PPP`. It is not applied to `A->PA`, `F->PF`, `AP->PAP`, `FP->PFP`, or `AI->PAI`. Label propagation, HOPE experts, SeHGNN semantic fusion, and MLP projections are unchanged by default.

The compatibility scope `--gsmp-scope first-hop` is available only as an ablation for the narrower direct-hop setting.

## Data

Expected defaults:

- OGB data: `../data/ogbn_mag`
- LINE embedding: `../HGAMLP_MAG/mag.p`
- sparse_tools: `../sparse_tools`

Prepare symlinks:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260618_ogbn_mag_SeHGNN_HOPE
bash scripts/prepare_data_links.sh
```

Validate before spending GPU time:

```bash
python scripts/check_data.py
```

The checker verifies `mag.p`, compact graph edge types, expected ogbn-mag node counts, splits, and derived GSMP time bins.

## Environment

Create the pinned HOPE environment:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260618_ogbn_mag_SeHGNN_HOPE
bash env/create_env.sh
```

It targets Python `3.8.20`, torch `1.13.0+cu117`, DGL `0.9.1.post1`, OGB `1.3.6`, torch-sparse `0.6.18`, and `sparse_tools`. Version checks are written to `logs/env_check.log`.

## Exact Commands

Baseline:

```bash
cd HOPE
python -u training.py \
  --dataset ogbn-mag \
  --aggregation SeHGNN-HOPE \
  --label-residual \
  --similarity-threshold 0.6 \
  --lower-bound 0.5 \
  --upper-bound 3 \
  --lamb 0.5 \
  --hidden 512 \
  --n-layers-1 2 \
  --n-layers-2 2 \
  --lr 0.001 \
  --weight-decay 0 \
  --stages 300 300 300 300 \
  --num-hops 2 \
  --label-feats \
  --num-label-hops 2 \
  --extra-embedding Line \
  --amp \
  --use-sparse-tools \
  --eval-every 1 \
  --patience 100 \
  --seeds 1 2 3 4 5 6 7 8 9 10
```

Proposed:

```bash
cd HOPE
python -u training.py \
  --dataset ogbn-mag \
  --aggregation SeHGNN-HOPE \
  --label-residual \
  --similarity-threshold 0.6 \
  --lower-bound 0.5 \
  --upper-bound 3 \
  --lamb 0.5 \
  --hidden 512 \
  --n-layers-1 2 \
  --n-layers-2 2 \
  --lr 0.001 \
  --weight-decay 0 \
  --stages 300 300 300 300 \
  --num-hops 2 \
  --label-feats \
  --num-label-hops 2 \
  --extra-embedding Line \
  --amp \
  --use-sparse-tools \
  --eval-every 1 \
  --patience 100 \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --gsmp-first-layer \
  --gsmp-scope paper-stack \
  --gsmp-normalizer nonempty \
  --gsmp-time-source all \
  --gsmp-derived-time mode
```

The Slurm wrapper runs the same command one seed per array task and passes `--method-name baseline` or `--method-name gsmp` for result aggregation.

## Slurm

Run a cheap smoke first:

```bash
sbatch slurm/check_data.sbatch
sbatch slurm/run_smoke.sbatch
```

Run a short pilot:

```bash
sbatch slurm/run_pilot.sbatch
```

Run the full 20-task comparison:

```bash
sbatch slurm/run_full.sbatch
```

The full array defaults to `0-19%1` intentionally. This warms and reuses feature-propagation caches instead of making many seeds duplicate CPU preprocessing while holding GPUs.

Monitor:

```bash
tail -f results/sehgnn_hope_full_live_progress.tsv
```

Summarize:

```bash
python scripts/summarize_results.py \
  --progress-file results/sehgnn_hope_full_live_progress.tsv \
  --out results/sehgnn_hope_full_summary
```

## Tests

With the HOPE environment active:

```bash
python tests/test_gsmp_weights.py
```

This checks that destination-centered nonempty GSMP weights sum to one and that the optional global normalizer uses the full source time universe.
