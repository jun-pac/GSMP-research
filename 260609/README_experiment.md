# 260609 ogbn-arxiv SimTeG/TAPE GraphSAGE SMP/UMP/GSMP

This folder is the clean 2026-06-09 experiment pipeline. New work should stay under:

```text
/users/PAS1289/jyp531/GSMP-research/260609
```

It compares:

- `baseline`: SimTeG/TAPE cached embeddings + GraphSAGE
- `smp`: baseline + SMP weighted temporal mean aggregation
- `ump`: baseline + future-to-past edges removed
- `gsmp`: baseline + uniform averaging across neighbor-year groups

The primary selection metric is always validation accuracy. Test accuracy is reported as `test_at_best_val`. The diagnostic `oracle_best_test_acc_not_for_model_selection` is logged but should not be used for model selection.

## Cached Embeddings

Default path:

```bash
/users/PAS1289/jyp531/GSMP-research/SimTeG/out/ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt
```

This pipeline does not launch LM fine-tuning. If embeddings are missing, download them explicitly:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260609
bash scripts/download_embeddings.sh
```

For the leaderboard-style SimTeG+TAPE GraphSAGE recipe, download all four cached embedding components:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260609
ALL_COMPONENTS=1 bash scripts/download_embeddings.sh
```

The official GraphSAGE ensemble also trains one component from GPT prediction labels. Download that small CSV separately:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260609
bash scripts/download_gpt_preds.sh
```

## Smoke Test

Runs `baseline smp ump gsmp`, seed `1`, for `5` epochs.

```bash
cd /users/PAS1289/jyp531/GSMP-research/260609
mkdir -p logs/smoke
sbatch slurm/smoke_ogbn_arxiv_smpumpgsmp.sbatch
```

Monitor:

```bash
bash scripts/monitor_latest.sh smoke
```

or:

```bash
tail -f logs/smoke/smoke_arxiv_mp_<jobid>.out
```

## Main Experiment

Default main run: variants `baseline smp ump gsmp`, seeds `1 2 3`, `100` epochs.

```bash
cd /users/PAS1289/jyp531/GSMP-research/260609
mkdir -p logs/main
sbatch slurm/main_ogbn_arxiv_smpumpgsmp.sbatch
```

Monitor:

```bash
bash scripts/monitor_latest.sh main
```

Summarize after or during the run:

```bash
../.venv/bin/python summarize_results.py --run_dir results/main_<jobid>
```

## Optional 10-Seed Final Reproduction

Run this only after the 3-seed main experiment looks reasonable:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260609
SEEDS="1 2 3 4 5 6 7 8 9 10" sbatch slurm/main_ogbn_arxiv_smpumpgsmp.sbatch
```

## Outputs

For run name `<run_name>`:

```text
results/<run_name>/epoch_metrics.csv
results/<run_name>/epoch_metrics.jsonl
results/<run_name>/summary.csv
results/<run_name>/checkpoints/<variant>/seed_<seed>/best_by_val.pt
```

Every epoch prints one line:

```text
RESULT variant=gsmp seed=1 epoch=37 val_acc=0.7742 test_acc=0.7701 best_val_acc=0.7742 test_at_best_val=0.7701 best_epoch=37 loss=...
```

Each seed ends with:

```text
SUMMARY variant=gsmp seed=1 best_epoch=... best_val_acc=... test_at_best_val=... oracle_best_test_acc_not_for_model_selection=...
```

## Official-Style GraphSAGE Defaults

The default settings mirror the SimTeG GraphSAGE cached-embedding setup:

```text
dataset: ogbn-arxiv
LM embedding: e5-large cached x_embs.pt
model: GraphSAGE
gnn_batch_size: 10000
gnn_eval_batch_size: 10000
gnn_epochs: 100
gnn_dropout: 0.4
gnn_label_smoothing: 0.4
gnn_lr: 0.01
gnn_num_layers: 2
gnn_weight_decay: 4e-6
gnn_eval_interval: 1
```

Override only when needed:

```bash
GNN_BATCH_SIZE=4096 EVAL_MODE=full sbatch slurm/smoke_ogbn_arxiv_smpumpgsmp.sbatch
```

## Baseline Calibration

The local SimTeG code applies `ToUndirected()` to `ogbn-arxiv` before GraphSAGE training. This 260609 runner does the same. To check the baseline only after code changes:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260609
mkdir -p logs/main
VARIANTS="baseline" SEEDS="42 43 44" RUN_NAME=baseline_calibration sbatch slurm/main_ogbn_arxiv_smpumpgsmp.sbatch
```

Monitor:

```bash
bash scripts/monitor_latest.sh main
```

Summarize:

```bash
../.venv/bin/python summarize_results.py --run_dir results/baseline_calibration
```

The OGB leaderboard-scale `77.89 / 77.48` result is a stronger SimTeG/TAPE GraphSAGE ensemble, not a single `ogbn-arxiv/e5-large/main/x_embs.pt` run. The official GraphSAGE ensemble uses cached logits from `ogbn-arxiv`, `ogbn-arxiv-tape`, `e5-large`, `all-roberta-large-v1`, and GPT-prediction labels. This folder first calibrates the single-source controlled baseline, then SMP/UMP/GSMP should be compared against that same controlled source. For leaderboard chasing, GSMP needs to be applied consistently to every ensemble component before ensembling.

## Leaderboard-Style Components

The rank-3 `SimTeG+TAPE+GraphSAGE` script trains these cached-embedding components:

```text
ogbn-arxiv/e5-large
ogbn-arxiv/all-roberta-large-v1
ogbn-arxiv-tape/e5-large
ogbn-arxiv-tape/all-roberta-large-v1
```

It also trains a fifth GPT-prediction component from `src/misc/gpt_preds/ogbn-arxiv.csv`. The commands below include that component by default.

Quick 3-seed baseline and GSMP component run:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260609
mkdir -p logs/main
ALL_COMPONENTS=1 bash scripts/download_embeddings.sh
bash scripts/download_gpt_preds.sh
VARIANTS="baseline gsmp" SEEDS="42 43 44" PREFIX=leaderboard3 \
  sbatch slurm/leaderboard_components_ogbn_arxiv.sbatch
```

Official-style 10-seed run:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260609
VARIANTS="baseline gsmp" SEEDS="42 43 44 45 46 47 48 49 50 51" PREFIX=leaderboard10 \
  sbatch slurm/leaderboard_components_ogbn_arxiv.sbatch
```

After the component job finishes, ensemble the baseline logits in the official component order:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260609
../.venv/bin/python ensemble_logits.py \
  --variant baseline \
  --seeds "42 43 44" \
  --run_dirs "results/leaderboard3_arxiv_e5_baseline results/leaderboard3_arxiv_roberta_baseline results/leaderboard3_arxiv_tape_e5_baseline results/leaderboard3_arxiv_tape_roberta_baseline results/leaderboard3_arxiv_gpt_preds_baseline" \
  --weights "2 2 1 1 1"
```

Then ensemble GSMP the same way:

```bash
../.venv/bin/python ensemble_logits.py \
  --variant gsmp \
  --seeds "42 43 44" \
  --run_dirs "results/leaderboard3_arxiv_e5_gsmp results/leaderboard3_arxiv_roberta_gsmp results/leaderboard3_arxiv_tape_e5_gsmp results/leaderboard3_arxiv_tape_roberta_gsmp results/leaderboard3_arxiv_gpt_preds_gsmp" \
  --weights "2 2 1 1 1"
```

For a `leaderboard10` run, replace `leaderboard3` with `leaderboard10` and use:

```bash
--seeds "42 43 44 45 46 47 48 49 50 51"
```
