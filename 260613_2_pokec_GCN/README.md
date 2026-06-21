# Pokec Temporal GCN vs First-Layer GSMP-GCN

This folder contains a SLURM-ready experiment pipeline for chronological Pokec
node classification.

The main comparison is:

- `gcn`: tunedGNN-style GCN baseline
- `gcn_gsmp_first`: same GCN, but the first GCN layer uses target-wise GSMP

The main training graph is undirected by default, following the tunedGNN Pokec
GCN convention: convert to undirected, remove self-loops, then add self-loops.
`--use-directed` is available only as an ablation.

## Paper-Aligned Pokec GCN* Setting

For the GCN* Pokec setting from "Classic GNNs are Strong Baselines:
Reassessing GNNs for Node Classification", the final SLURM scripts default to:

- residual connections: enabled
- normalization: batch normalization
- dropout: `0.2`
- local GCN layers: `7`
- hidden dimension: `256`
- learning rate: `0.0005`
- weight decay: `0.0`
- epochs: `2000`
- batch size: `550000`
- evaluation: every `9` epochs after epoch `1000`
- runs/seeds: `5`, seeds `0 1 2 3 4`

The model implementation follows the paper code's large-graph architecture:
input features are passed through `7` hidden GCN layers and then a separate
linear prediction head. The GSMP variant changes only the first GCN layer's
adjacency weights.

## Cost-Saving Defaults

- Preprocessing is CPU-only.
- Training keeps full tensors and full `edge_index` on CPU.
- Only one induced mini-batch subgraph is moved to GPU at a time.
- Full-graph evaluation defaults to CPU with `--eval-device cpu`.
- Smoke runs use one seed, few epochs, and smaller batch size.
- Exact GSMP weights are recomputed on each batch-induced subgraph by default.
- The script catches CUDA OOM and suggests smaller batch sizes.

## Important Feature Note

For final paper-quality runs, use the processed Pokec benchmark file, usually
`pokec.mat`, with 65 node features and 2 classes:

```bash
PROCESSED_PATH=/path/to/pokec.mat sbatch slurm/preprocess_pokec_temporal.slurm
```

The current visible repo does not expose `pokec.mat`. For cheap pipeline smoke
testing only, you can create raw profile-derived features:

```bash
ALLOW_RAW_PROFILE_FEATURES=1 sbatch slurm/preprocess_pokec_temporal.slurm
```

That fallback is marked in `data/pokec_temporal/metadata.json` as
`raw_profile_fallback` and should not be described as the tunedGNN 65-feature
benchmark.

## SLURM Commands

Preprocess:

```bash
sbatch slurm/preprocess_pokec_temporal.slurm
```

Smoke test:

```bash
sbatch slurm/smoke_pokec_gcn_gsmp.slurm
```

Final GCN:

```bash
sbatch slurm/run_pokec_gcn_temporal.slurm
```

Final GCN+GSMP:

```bash
sbatch slurm/run_pokec_gsmp_temporal.slurm
```

Run all:

```bash
sbatch slurm/run_pokec_all_temporal.slurm
```

Tail logs:

```bash
tail -f logs/pokec_gcn_temporal_<jobid>.log
tail -f logs/pokec_gsmp_temporal_<jobid>.log
```

## Direct Python Commands

Preprocess with a processed benchmark file:

```bash
python scripts/preprocess_pokec_temporal.py \
  --snap-dir ../data/pokec \
  --processed-path /path/to/pokec.mat \
  --out-dir data/pokec_temporal
```

Smoke fallback preprocess:

```bash
python scripts/preprocess_pokec_temporal.py \
  --snap-dir ../data/pokec \
  --out-dir data/pokec_temporal \
  --allow-raw-profile-features
```

GCN:

```bash
python scripts/train_pokec_temporal.py \
  --data-dir data/pokec_temporal \
  --method gcn \
  --hidden-channels 256 \
  --num-layers 7 \
  --dropout 0.2 \
  --in-dropout 0.0 \
  --lr 0.0005 \
  --weight-decay 0.0 \
  --batch-size 550000 \
  --epochs 2000 \
  --bn \
  --res \
  --eval-step 9 \
  --eval-start-epoch 1000 \
  --runs 5 \
  --seeds 0 1 2 3 4
```

GCN + first-layer GSMP:

```bash
python scripts/train_pokec_temporal.py \
  --data-dir data/pokec_temporal \
  --method gcn_gsmp_first \
  --hidden-channels 256 \
  --num-layers 7 \
  --dropout 0.2 \
  --in-dropout 0.0 \
  --lr 0.0005 \
  --weight-decay 0.0 \
  --batch-size 550000 \
  --epochs 2000 \
  --bn \
  --res \
  --eval-step 9 \
  --eval-start-epoch 1000 \
  --runs 5 \
  --seeds 0 1 2 3 4
```

Summarize:

```bash
python scripts/summarize_results.py \
  --results-csv results/pokec_temporal_results.csv \
  --out-md results/pokec_temporal_summary.md
```

## Outputs

Preprocessing writes:

- `data/pokec_temporal/train_idx.pt`
- `data/pokec_temporal/valid_idx.pt`
- `data/pokec_temporal/test_idx.pt`
- `data/pokec_temporal/year_raw.pt`
- `data/pokec_temporal/year_idx.pt`
- `data/pokec_temporal/split_group.pt`
- `data/pokec_temporal/edge_index_directed.pt`
- `data/pokec_temporal/edge_index_undirected_self_loop.pt`
- `data/pokec_temporal/x.pt`
- `data/pokec_temporal/y.pt`
- `data/pokec_temporal/metadata.json`
- temporal connectivity CSVs and paper-description files

Training appends rows to:

- `results/pokec_temporal_results.csv`

Summary writes:

- `results/pokec_temporal_summary.md`

## Resuming Longer Runs

To extend a run later without replaying previous epochs, enable last-checkpoint
saving before the original run starts:

```bash
SAVE_CHECKPOINTS=1 sbatch ...
```

This writes per-method/per-seed checkpoints under `results/checkpoints/`, for
example:

- `gcn_seed1_last.pt`
- `gcn_gsmp_first_seed1_last.pt`

Then resume to a larger epoch count with:

```bash
RESUME_FROM_CHECKPOINT=1 SAVE_CHECKPOINTS=1 EPOCHS=200 sbatch ...
```

Runs that were launched without `SAVE_CHECKPOINTS=1` only save CSV metrics, not
model/optimizer state, and cannot be continued exactly from their final epoch.

## GSMP Definition Implemented

For each message edge `src -> dst`, GSMP counts source timestamps target-wise:

```text
C_dst(tau) = |{u : (u, dst) in E and time(u) = tau}|
```

For non-self edges:

```text
base(src -> dst) = 1 / max(1, C_dst(time(src)))
weight(src -> dst) = base(src -> dst) / mean_base_into_dst
```

Self-loops are excluded from timestamp counts and assigned weight `1.0` by
default. This GSMP weighting is applied only to the first GCN layer. Layers 2
through 7 are ordinary GCN layers.
