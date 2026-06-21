# 260610: OGBN-Arxiv SimTeG/TAPE RevGAT GSMP

This folder is a self-contained experiment harness for testing whether GSMP helps the SimTeG/TAPE + RevGAT OGBN-Arxiv pipeline.

The code is deliberately conservative by default:

- It uses cached SimTeG/TAPE embeddings and GPT-pred labels.
- It does not fine-tune language models.
- Smoke runs use one seed, one component, and two epochs.
- The 10-seed SLURM script refuses to run unless `CONFIRM_FINAL=1`.
- The 10-seed SLURM script caps each run at 200 epochs and enables validation early stopping after epoch 80 with patience 40.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in runners to reduce CUDA fragmentation failures.

## Files

- `linear_revgat_gsmp.py`: Linearized RevGAT trainer. It keeps the RevGAT shell, reversible block machinery, normalization, dropout, label injection, label smoothing, RMSprop, and warmup schedule, but replaces GAT attention with fixed weighted linear aggregation.
- `ensemble_logits.py`: Ensembles saved best-validation logits.
- `scripts/run_linear_component.sh`: Runs one component and one seed at a time.
- `scripts/run_linear_components.sh`: Runs the component matrix for `linear` and/or `gsmp`.
- `scripts/run_official_revgat_component.sh`: Wrapper around the local official SimTeG RevGAT code.
- `scripts/run_official_revgat_components.sh`: Smoke/sanity runner for official RevGAT components.
- `scripts/ensemble_official_revgat.sh`: Ensembles official RevGAT component logits saved by the wrapper.
- `slurm/smoke_ogbn_arxiv_revgat_gsmp.sbatch`: Safe smoke run.
- `slurm/smoke_cpu_ogbn_arxiv_revgat_gsmp.sbatch`: No-GPU smoke run for environments with CPU-only DGL.
- `slurm/three_seed_ogbn_arxiv_revgat_gsmp.sbatch`: Conservative 3-seed sanity run.
- `slurm/ten_seed_ogbn_arxiv_revgat_gsmp.sbatch`: Guarded final run.

## GSMP Convention

DGL edges are treated as `source u -> destination v`. Message passing aggregates incoming source messages into destination receiver nodes.

For each edge `u -> v`, GSMP uses `node_year[u]` as the source/neighbor time. For a receiver `v`, incoming neighbors are grouped by source year:

```text
C_v(tau) = number of incoming neighbors u with node_year[u] = tau
base_weight[u -> v] = 1 / C_v(node_year[u])
```

Then:

- `--gsmp_norm active_years`: divide by the number of nonempty source-year groups for `v`.
- `--gsmp_norm all_years`: divide by the whole year universe.

The default is `active_years`.

## Setup

From this folder:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260610
```

Check cached embeddings:

```bash
ls ../SimTeG/out/ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt
ls ../SimTeG/out/ogbn-arxiv/all-roberta-large-v1/main/cached_embs/x_embs.pt
ls ../SimTeG/out/ogbn-arxiv-tape/e5-large/main/cached_embs/x_embs.pt
ls ../SimTeG/out/ogbn-arxiv-tape/all-roberta-large-v1/main/cached_embs/x_embs.pt
```

Download missing embeddings only if needed:

```bash
ALL_COMPONENTS=1 bash scripts/download_embeddings.sh
```

Download GPT-pred labels if needed:

```bash
bash scripts/download_gpt_preds.sh
```

## Safe Run Order

Smoke test:

```bash
sbatch slurm/smoke_ogbn_arxiv_revgat_gsmp.sbatch
```

If the GPU smoke fails with `Device API cuda is not enabled`, the environment has CPU-only DGL. Use the no-GPU smoke to validate code paths without burning GPU allocation:

```bash
sbatch slurm/smoke_cpu_ogbn_arxiv_revgat_gsmp.sbatch
```

For real GPU RevGAT runs, install or load a CUDA-enabled DGL build matching the cluster CUDA/PyTorch stack.

Monitor:

```bash
bash scripts/monitor_latest.sh
```

Optional official RevGAT smoke, still short:

```bash
RUN_OFFICIAL_BASELINE=1 sbatch slurm/smoke_ogbn_arxiv_revgat_gsmp.sbatch
```

Three-seed sanity run:

```bash
sbatch slurm/three_seed_ogbn_arxiv_revgat_gsmp.sbatch
```

Final 10-seed run, guarded:

```bash
CONFIRM_FINAL=1 sbatch slurm/ten_seed_ogbn_arxiv_revgat_gsmp.sbatch
```

## Direct Local Commands

Run one cheap component directly:

```bash
COMPONENT_LIMIT=1 VARIANTS="linear gsmp" SEEDS="42" EPOCHS=2 bash scripts/run_linear_components.sh
```

Run all five components for three seeds with conservative epochs:

```bash
COMPONENT_LIMIT=5 VARIANTS="linear gsmp" SEEDS="42 43 44" EPOCHS=30 bash scripts/run_linear_components.sh
```

Ensemble after all five component logits exist:

```bash
PREFIX=smoke VARIANT=linear SEEDS="42" bash scripts/ensemble_linear.sh
PREFIX=smoke VARIANT=gsmp SEEDS="42" bash scripts/ensemble_linear.sh
```

Ensemble official RevGAT wrapper outputs after all five official components exist:

```bash
PREFIX=official_smoke SEEDS="1" bash scripts/ensemble_official_revgat.sh
```

## Result Layout

Logs and CSVs are written under:

```text
runs/
  ogbn_arxiv_simteg_tape_revgat_gsmp/
    logs/
      exp_<experiment_name>_seed<seed>.log
      smoke.log or slurm_*.out
    csv/
      per_epoch.csv
      per_seed_summary.csv
      aggregate_summary.csv
    components/
      <run_name>/
        cached_embs/logits_seed<seed>.pt
        per_epoch.csv
        per_seed_summary.csv
        aggregate_summary.json
```

The main metric is `test_acc_at_best_valid`. Pure best test accuracy is logged as diagnostic only.
