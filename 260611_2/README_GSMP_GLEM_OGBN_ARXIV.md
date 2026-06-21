# GLEM RevGAT GSMP on ogbn-arxiv

This workspace compares:

1. `GLEM+RevGAT`: official GLEM leaderboard baseline.
2. `GLEM+LinearRevGAT`: RevGAT shell with learned attention aggregation replaced by fixed linear aggregation.
3. `GLEM+LinearRevGAT+GSMP`: the same linear aggregation with publication-year GSMP weights.

The official GLEM repo is cloned under:

```text
260611_2/upstream/GLEM
```

No full training job has been launched automatically. Defaults and scripts are smoke-first to protect GPU hours, credits, and memory.

## Target Anchor

Reproduce the official GLEM+RevGAT anchor before spending full GPU time on GSMP:

```text
validation accuracy: 0.7746 +/- 0.0018
test accuracy:       0.7694 +/- 0.0025
```

The main scientific number is `test_at_best_val`, not `best_raw_test`.

## Modified Files

- `upstream/GLEM/src/models/GNNs/LinearRevGAT/gsmp.py`: CPU GSMP and linear edge-weight preprocessing with disk cache.
- `upstream/GLEM/src/models/GNNs/LinearRevGAT/model.py`: `LinearConv`, `LinearRevGATBlock`, and `LinearRevGAT`.
- `upstream/GLEM/src/models/GNNs/LinearRevGAT/config.py`: LinearRevGAT config flags.
- `upstream/GLEM/src/models/GNNs/LinearRevGAT/__init__.py`: model registration.
- `upstream/GLEM/src/models/GNNs/gnn_utils.py`: trainer dispatch and result/logging flags.
- `upstream/GLEM/src/models/GNNs/gnn_trainer.py`: LinearRevGAT construction, CPU weight attachment, `[RESULT]` logs, CSV/JSON summaries.
- `upstream/GLEM/src/models/GNNs/RevGAT/config.py`: shared logging/result flags.
- `upstream/GLEM/src/models/GLEM/config.py`: experiment-mode metadata flags.
- `upstream/GLEM/src/utils/data/preprocess.py`: LinearRevGAT uses the same bidirected self-loop graph path as RevGAT.

## Added Files

- `scripts/run_glem_gsmp_experiments.sh`: one-command launcher.
- `scripts/run_one_glem_gnn.sh`: single method wrapper around official `trainGLEM.py`.
- `scripts/check_glem_assets.sh`: checks likely cached data/LM/GNN artifacts.
- `scripts/monitor_latest.sh`: tails the newest Slurm log.
- `slurm/*.sbatch`: conservative one-GPU Slurm scripts.
- `tests/test_gsmp_weights.py`: deterministic GSMP unit test.
- `tools/check_baseline_gate.py`: blocks full GSMP until RevGAT baseline looks reproduced.
- `tools/compare_results.py`: aggregates result JSON files.

## Environment

Use the official GLEM environment when possible:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260611_2/upstream/GLEM
conda env create -f environment.yml
conda activate glem
```

The Slurm scripts contain this editable section:

```bash
source ~/.bashrc
conda activate "${GLEM_CONDA_ENV:-glem}"
```

Account, partition, and V100 constraints are present as commented `##SBATCH` lines.

## Assets

Check existing OGB data, LM outputs, logits, and GNN predictions:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260611_2
bash scripts/check_glem_assets.sh
```

If pretrained LM embeddings/predictions already exist, GLEM should reuse them. If they are missing, even a smoke run may train or infer with the LM, which costs GPU time.

## Static Checks

Run this before any GPU job:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260611_2
bash scripts/run_glem_gsmp_experiments.sh static
```

This runs the tiny GSMP unit test and bytecode checks.

## Smoke Test

Recommended on HPC:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260611_2
USE_SLURM=1 PROFILE=smoke bash scripts/run_glem_gsmp_experiments.sh smoke
```

This uses seed 0 and 3 GNN epochs by default for baseline, linear, and GSMP. For a direct local smoke run, omit `USE_SLURM=1`.

Inspect the exact command without launching training:

```bash
DRY_RUN=1 PROFILE=smoke bash scripts/run_one_glem_gnn.sh gsmp
```

## Budget Guard

Smoke/full launchers default to:

```bash
BUDGET_GUARD_NO_LM_WORK=T
```

This stops before GLEM launches LM pretraining or LM inference when cached pretrained LM artifacts are missing. Slurm jobs also start `scripts/budget_guard_watch.sh`, which cancels the job if the log contains LM-pretraining, LM-inference, raw-text/tokenization, or `[STOPPED_BUDGET_GUARD]` sentinels. The required OGB graph download is allowed.

For `PROFILE=smoke`, the launcher also submits with reduced Slurm resources by default:

```bash
SBATCH_TIME=00:15:00
SBATCH_MEM=24G
BUDGET_GUARD_MAX_SECONDS=600
BUDGET_GUARD_MAX_GPU_MEM_MB=14500
BUDGET_GUARD_MAX_RSS_MB=22000
```

The dedicated linear and linear+GSMP Slurm scripts set the same watchdog caps for smoke jobs, even if you submit them directly. Override these only deliberately, for example:

```bash
BUDGET_GUARD_MAX_SECONDS=900 USE_SLURM=1 SBATCH_ACCOUNT=PAS1289 PROFILE=smoke bash scripts/run_glem_gsmp_experiments.sh gsmp
```

GNN pretrain caches are keyed by profile, seed, epoch count, early-stop setting, and GSMP norm. This prevents a full run from accidentally reusing a 3-epoch smoke prediction cache.

Override only deliberately:

```bash
BUDGET_GUARD_NO_LM_WORK=F SBATCH_ACCOUNT=PAS1289 PROFILE=smoke sbatch slurm/reproduce_glem_revgat_baseline.sbatch
```

## Full Run Order

First reproduce the official baseline:

```bash
USE_SLURM=1 bash scripts/run_glem_gsmp_experiments.sh baseline
```

Then run the linear ablation:

```bash
USE_SLURM=1 bash scripts/run_glem_gsmp_experiments.sh linear
```

Then run GSMP:

```bash
USE_SLURM=1 bash scripts/run_glem_gsmp_experiments.sh gsmp
```

Full GSMP runs call `tools/check_baseline_gate.py` first. Override only deliberately:

```bash
SKIP_BASELINE_GATE=1 USE_SLURM=1 bash scripts/run_glem_gsmp_experiments.sh gsmp
```

Run all methods/seeds as a serial Slurm array:

```bash
USE_SLURM=1 bash scripts/run_glem_gsmp_experiments.sh all
```

The all-array script uses `#SBATCH --array=0-14%1`: 3 methods times 5 seeds, one concurrent task.

## Optional Full EM Retraining

The launcher exposes:

```bash
FULL_EM_RETRAIN=T USE_SLURM=1 bash scripts/run_glem_gsmp_experiments.sh linear
FULL_EM_RETRAIN=T USE_SLURM=1 bash scripts/run_glem_gsmp_experiments.sh gsmp
```

Do this only after the cheaper frozen-output GNN ablation suggests GSMP is worth the cost.

## Monitoring

```bash
bash scripts/monitor_latest.sh
FOLLOW_LOG=0 bash scripts/monitor_latest.sh
WATCH_SECONDS=30 FOLLOW_LOG=0 bash scripts/monitor_latest.sh
tail -f logs/<jobname>_<jobid>.out
grep "\[RESULT\]" logs/<jobname>_<jobid>.out
grep "\[SEED_SUMMARY\]" logs/<jobname>_<jobid>.out
grep "\[FINAL\]" logs/<jobname>_<jobid>.out
watch -n 5 squeue -u $USER
```

`scripts/monitor_latest.sh` prints the Slurm queue, completed accuracy summary, latest epoch progress, and recent result/guard lines from the newest log. By default it then tails the latest log. Use `FOLLOW_LOG=0` for a one-shot table, or `WATCH_SECONDS=30 FOLLOW_LOG=0` for an auto-refreshing accuracy dashboard.

## Results

Each GNN run writes:

```text
results/glem_revgat_gsmp/<run_id>/config.json
results/glem_revgat_gsmp/<run_id>/epoch_logs.csv
results/glem_revgat_gsmp/<run_id>/seed_summary.csv
results/glem_revgat_gsmp/<run_id>/final_summary.json
```

Aggregate:

```bash
python tools/compare_results.py --results-root results/glem_revgat_gsmp
```

Expected table columns:

```text
method,val_acc_mean+/-std,test_at_best_val_mean+/-std,best_raw_test_mean+/-std,best_epoch_mean+/-std,runtime_mean,gpu_memory_peak_mean
```

## GSMP Direction

GLEM processes ogbn-arxiv with:

```text
dgl.to_bidirected(graph)
remove_self_loop()
add_self_loop()
```

LinearRevGAT computes weights on that exact graph. For every DGL message edge `src -> dst`, GSMP counts:

```text
count[dst, year[src]]
```

GSMP is never multiplied into learned attention coefficients. `use_gsmp=T` is only supported for `LinearRevGAT`.

## Interpretation

The fair GSMP comparison is:

```text
GLEM+LinearRevGAT vs GLEM+LinearRevGAT+GSMP
```

`GLEM+RevGAT` is the reproduction anchor and upper baseline, not the direct GSMP ablation baseline.

If GSMP beats LinearRevGAT, timestamp-balanced aggregation likely helps the GNN M-step. If it is worse, GSMP may be over-regularizing or disrupting useful citation-time signal. If it is neutral, GLEM's LM/GNN interaction may already absorb much of the correction.
