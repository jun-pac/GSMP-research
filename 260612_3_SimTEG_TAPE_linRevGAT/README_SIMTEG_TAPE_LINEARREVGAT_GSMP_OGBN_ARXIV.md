# SimTeG/TAPE LinearRevGAT GSMP on ogbn-arxiv

This folder contains the 2026-06-12 experiment harness for:

```text
SimTeG+TAPE+linearRevGAT
SimTeG+TAPE+linearRevGAT+GSMP-first-layer-only
SimTeG+TAPE+linearRevGAT+P-GSMP
```

The official SimTeG+TAPE+RevGAT leaderboard number is the reproduction anchor:

```text
test accuracy:       0.7803 +/- 0.0007
validation accuracy: 0.7846 +/- 0.0004
```

It is not the direct GSMP comparator, because RevGAT uses learned attention and this experiment does not apply GSMP to attention coefficients. The fair comparison is only among the `linearRevGAT` variants.

## What Was Inspected

Local official SimTeG files used as reference:

- `../SimTeG/scripts/ogbn-arxiv/e5-large/main.sh`
- `../SimTeG/scripts/ogbn-arxiv-tape/e5-large/main.sh`
- `../SimTeG/scripts/ogbn-arxiv/roberta-large/main.sh`
- `../SimTeG/scripts/ogbn-arxiv-tape/roberta-large/main.sh`
- `../SimTeG/scripts/ogbn-arxiv-tape/revgat/main.sh`
- `../SimTeG/src/misc/revgat/main.py`
- `../SimTeG/src/misc/revgat/model_rev.py`
- `../SimTeG/compute_ensemble.py`

The official RevGAT path uses DGL `ogbn-arxiv`, makes the graph bidirected, removes self-loops, then adds one self-loop per node. This harness computes GSMP/P-GSMP on that exact DGL edge direction: `src -> dst`, where `src` contributes to `dst`.

## Cached Features

This harness reuses cached embeddings by default and does not fine-tune language models or regenerate TAPE explanations.

Expected local cached embeddings:

```bash
ls ../SimTeG/out/ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt
ls ../SimTeG/out/ogbn-arxiv/all-roberta-large-v1/main/cached_embs/x_embs.pt
ls ../SimTeG/out/ogbn-arxiv-tape/e5-large/main/cached_embs/x_embs.pt
ls ../SimTeG/out/ogbn-arxiv-tape/all-roberta-large-v1/main/cached_embs/x_embs.pt
```

The GPT-prediction component is supported for `baseline` and `gsmp_first_layer`, but P-GSMP is disabled for GPT-pred label features because P-GSMP assumes real-valued `N x F` node features.

## Variant Definitions

Full layer-wise GSMP applies temporal balancing at every GNN layer. That is not the main variant here.

GSMP-first-layer-only uses GSMP weights only at layer `0`:

```text
Layer 0: GSMP-weighted linear aggregation
Layer 1..K: ordinary linearRevGAT aggregation
```

P-GSMP is preprocessing only:

```text
X_pg = P_GSMP(X)
logits = linearRevGAT(X_pg, edge_index)
```

It does not alter later GNN aggregation.

## Environment

The SLURM scripts include an editable cluster section:

```bash
source ~/.bashrc || true
conda activate "${CONDA_ENV:-simteg}" || true
```

The launcher passes cluster options to `sbatch`. It infers `ACCOUNT=PAS1289` from this workspace path on Pitzer, and you can override it:

```bash
ACCOUNT=PAS1289 PARTITION=<partition> QOS=<qos> bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh smoke
```

On this workspace, the local virtualenv also works:

```bash
../.venv/bin/python -m py_compile gsmp_utils.py linear_revgat_gsmp_experiment.py
```

## Static Checks

Run:

```bash
../.venv/bin/python -m py_compile gsmp_utils.py linear_revgat_gsmp_experiment.py verify_revgat_anchor.py ensemble_logits.py summarize_final.py
bash -n scripts/run_one_linearrevgat_component.sh scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh slurm/*.sbatch
../.venv/bin/python tests/test_gsmp_utils.py
```

## One-Command Smoke

Submit a conservative smoke test: one seed, one component, three modes, three epochs.

```bash
bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh smoke
```

For a local CPU integration check without SLURM:

```bash
CPU=1 LOCAL=1 USE_LABELS=0 N_LABEL_ITERS=0 EPOCHS=1 COMPONENTS="arxiv_e5" \
  EXPERIMENT_MODES="baseline" SEEDS="42" \
  bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh smoke
```

## Anchor Verification

This is budget-safe by default: it looks for cached official RevGAT logits and computes the ensemble if they exist. It does not launch expensive official training.

```bash
bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh anchor
```

If cached logits are missing, it prints:

```text
[WARNING] SimTeG+TAPE+RevGAT leaderboard anchor was not reproduced in this run.
[WARNING] Interpret linearRevGAT GSMP comparisons as internal ablations only.
```

## Run Main Modes

Single-seed text-component baseline:

```bash
bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh baseline
```

First-layer-only GSMP:

```bash
bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh gsmp1
```

P-GSMP:

```bash
bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh pgsmp
```

All three modes:

```bash
bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh all
```

Three-seed run, still explicit:

```bash
SEEDS="42 43 44" EPOCHS=200 bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh all
```

Include GPT-pred labels only for baseline/GSMP1:

```bash
INCLUDE_GPT_PREDS=1 bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh baseline
INCLUDE_GPT_PREDS=1 bash scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh gsmp1
```

## Monitor

```bash
tail -f logs/<jobname>_<jobid>.out
grep "\[GSMP\]" logs/<jobname>_<jobid>.out
grep "\[PGSMP\]" logs/<jobname>_<jobid>.out
grep "\[RESULT\]" logs/<jobname>_<jobid>.out
grep "\[SEED_SUMMARY\]" logs/<jobname>_<jobid>.out
grep "\[FINAL\]" logs/<jobname>_<jobid>.out
watch -n 5 squeue -u $USER
```

## Output Files

Each component/seed job writes:

```text
results/simteg_tape_linearrevgat_gsmp/<run_id>/config.json
results/simteg_tape_linearrevgat_gsmp/<run_id>/epoch_logs.csv
results/simteg_tape_linearrevgat_gsmp/<run_id>/seed_summary.csv
results/simteg_tape_linearrevgat_gsmp/<run_id>/final_summary.json
```

If `SAVE_PRED=1`, best-validation logits are saved in:

```text
results/simteg_tape_linearrevgat_gsmp/<run_id>/cached_embs/logits_seed<seed>.pt
```

## Final Comparison Table

After runs finish:

```bash
../.venv/bin/python summarize_final.py \
  results/simteg_tape_linearrevgat_gsmp/<baseline_run>/final_summary.json \
  results/simteg_tape_linearrevgat_gsmp/<gsmp1_run>/final_summary.json \
  results/simteg_tape_linearrevgat_gsmp/<pgsmp_run>/final_summary.json
```

The table reports:

```text
method
val_acc_mean+/-std
test_at_best_val_mean+/-std
best_raw_test_mean+/-std
best_epoch_mean+/-std
runtime_mean
peak_gpu_memory_mean
preprocessing_time
cache_reused
```

The main scientific number is `test_at_best_val`, not best raw test.

## Ensembling

After component logits exist, ensemble them:

```bash
../.venv/bin/python ensemble_logits.py \
  --name linearrevgat_text4_baseline \
  --seeds "42" \
  --run-dirs "results/simteg_tape_linearrevgat_gsmp/<arxiv_e5_run> results/simteg_tape_linearrevgat_gsmp/<arxiv_roberta_run> results/simteg_tape_linearrevgat_gsmp/<tape_e5_run> results/simteg_tape_linearrevgat_gsmp/<tape_roberta_run>" \
  --weights "2 2 1 1"
```

For the optional five-component baseline/GSMP1 ensemble, add the GPT-pred run directory and use:

```bash
--weights "2 2 1 1 1"
```

## Cache Rules

GSMP edge weights are cached under:

```text
cache/gsmp/
```

P-GSMP features are cached under:

```text
cache/pgsmp/
```

Cache names encode dataset, direction, bidirected setting, self-loop setting, normalization, edge count, node count, and for P-GSMP the alpha/depth/self-mode/feature shape/source fingerprint.

To force recomputation, pass:

```bash
GSMP_FORCE_RECOMPUTE=1
PGSMP_FORCE_RECOMPUTE=1
```

or call `linear_revgat_gsmp_experiment.py` directly with `--gsmp-force-recompute` / `--pgsmp-force-recompute`.

## Interpretation Rules

1. The leaderboard anchor is SimTeG+TAPE+RevGAT, but the fair GSMP comparison is among linearRevGAT variants because GSMP is not applied to learned attention coefficients.
2. First-layer-only GSMP is a weaker intervention than full layer-wise GSMP. It tests whether early temporal balancing helps without over-regularizing all layers.
3. P-GSMP is a preprocessing method. It tests whether timestamp-balanced feature smoothing helps before the GNN while keeping the GNN architecture unchanged.
4. If first-layer-only GSMP improves over linearRevGAT, early-layer temporal balancing is useful even with strong SimTeG/TAPE features.
5. If first-layer-only GSMP hurts, even light temporal balancing may remove useful citation-time signal or conflict with strong LM-derived features.
6. If P-GSMP improves, the best role of GSMP in this strong-feature setting may be preprocessing rather than repeated layer-wise aggregation.
7. If P-GSMP hurts, preprocessing may oversmooth or distort strong SimTeG/TAPE features before the GNN can use them.

## Files Added

- `gsmp_utils.py`
- `linear_revgat_gsmp_experiment.py`
- `verify_revgat_anchor.py`
- `ensemble_logits.py`
- `summarize_final.py`
- `tests/test_gsmp_utils.py`
- `scripts/run_one_linearrevgat_component.sh`
- `scripts/run_simteg_tape_linearrevgat_gsmp_experiments.sh`
- `slurm/reproduce_simteg_tape_revgat_anchor.sbatch`
- `slurm/run_simteg_tape_linearrevgat_baseline.sbatch`
- `slurm/run_simteg_tape_linearrevgat_gsmp1.sbatch`
- `slurm/run_simteg_tape_linearrevgat_pgsmp.sbatch`
- `slurm/run_simteg_tape_linearrevgat_all_array.sbatch`
