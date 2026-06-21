# TAPE RevGAT GSMP OGBN-Arxiv

This folder is an isolated TAPE-only workspace for comparing:

1. `TAPE+RevGAT`
2. `TAPE+LinearRevGAT`
3. `TAPE+LinearRevGAT+GSMP`

The official TAPE repository is cloned under:

```text
260610_2/upstream/TAPE
```

No long runs are launched automatically. Defaults are smoke-first to protect GPU hours, credits, and memory.

## What Was Added

- `upstream/TAPE/core/gsmp.py`: CPU GSMP preprocessing with cache support.
- `upstream/TAPE/core/GNNs/RevGAT/linear_model.py`: RevGAT shell with attention replaced by linear aggregation.
- `upstream/TAPE/core/GNNs/dgl_gnn_trainer.py`: RevGAT/LinearRevGAT trainer with `[RESULT]`, `[SEED_SUMMARY]`, `[FINAL]` logs and CSV/JSON output.
- `tests/test_gsmp.py`: tiny deterministic GSMP correctness test.
- `slurm/*.sbatch`: smoke-first Slurm jobs.
- `scripts/run_tape_gsmp_experiments.sh`: one-command launcher.
- `tools/check_baseline_gate.py`: blocks full GSMP unless baseline reproduction is close enough.
- `tools/compare_results.py`: final comparison table.

## Cost-Safe Run Order

From this folder:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260610_2
```

Static checks:

```bash
python tests/test_gsmp.py
python -m py_compile upstream/TAPE/core/gsmp.py
python -m py_compile upstream/TAPE/core/GNNs/RevGAT/linear_model.py
python -m py_compile upstream/TAPE/core/GNNs/dgl_gnn_trainer.py
```

Smoke run for all three methods, 1 seed and 5 epochs by default:

```bash
bash scripts/run_tape_gsmp_experiments.sh smoke
```

Official TAPE+RevGAT baseline smoke:

```bash
bash scripts/run_tape_gsmp_experiments.sh baseline
```

Linear aggregation ablation smoke:

```bash
bash scripts/run_tape_gsmp_experiments.sh linear
```

GSMP smoke:

```bash
bash scripts/run_tape_gsmp_experiments.sh gsmp
```

Full baseline reproduction, using `TA_P_E` and 3 seeds by default:

```bash
FULL=1 bash scripts/run_tape_gsmp_experiments.sh baseline --full
```

Only after the RevGAT baseline is close to the expected validation accuracy, run:

```bash
FULL=1 CONFIRM_FULL=1 bash scripts/run_tape_gsmp_experiments.sh all --full
```

The GSMP full path runs `tools/check_baseline_gate.py` unless `SKIP_BASELINE_GATE=1` is set.

## Direct TAPE Commands

Baseline:

```bash
cd upstream/TAPE
python -m core.trainEnsemble \
  dataset ogbn-arxiv \
  gnn.model.name RevGAT \
  gnn.train.feature_type TA_P_E \
  gnn.train.lr 0.002 \
  gnn.train.dropout 0.75
```

Linear ablation:

```bash
python -m core.trainEnsemble \
  dataset ogbn-arxiv \
  gnn.model.name LinearRevGAT \
  gnn.model.use_gsmp False \
  gnn.train.feature_type TA_P_E \
  gnn.train.lr 0.002 \
  gnn.train.dropout 0.75
```

GSMP:

```bash
python -m core.trainEnsemble \
  dataset ogbn-arxiv \
  gnn.model.name LinearRevGAT \
  gnn.model.use_gsmp True \
  gnn.model.gsmp_norm scale_preserve \
  gnn.train.feature_type TA_P_E \
  gnn.train.lr 0.002 \
  gnn.train.dropout 0.75
```

## Monitoring

Slurm logs are written to:

```text
260610_2/logs/%x_%j.out
260610_2/logs/%x_%j.err
```

Useful commands:

```bash
tail -f logs/<jobname>_<jobid>.out
grep "\[RESULT\]" logs/<jobname>_<jobid>.out
grep "\[SEED_SUMMARY\]" logs/<jobname>_<jobid>.out
grep "\[FINAL\]" logs/<jobname>_<jobid>.out
watch -n 5 squeue -u $USER
bash scripts/monitor_latest.sh
```

The separate Slurm resource dashboard added earlier is still useful for GPU-hours, memory allocation, and cost-unit monitoring:

```bash
cd /users/PAS1289/jyp531/GSMP-research
./slurm_monitor/run_monitor.sh status
./slurm_monitor/run_dashboard.sh status
```

## Results

Each run writes:

```text
results/tape_revgat_gsmp/<run_id>/config.json
results/tape_revgat_gsmp/<run_id>/epoch_logs.csv
results/tape_revgat_gsmp/<run_id>/seed_summary.csv
results/tape_revgat_gsmp/<run_id>/final_summary.json
```

Create the comparison table:

```bash
python tools/compare_results.py --results-root results/tape_revgat_gsmp
python tools/compare_results.py --results-root results/tape_revgat_gsmp --feature-type all
```

The main scientific number is:

```text
test accuracy at the best validation epoch
```

Best raw test accuracy is logged only as a diagnostic.

## GSMP Direction Convention

DGL edges are treated as source `src` to target `dst`.

For each edge `src -> dst`, GSMP uses `node_year[src]` and counts how many incoming neighbors of `dst` have the same source publication year.

Supported modes:

- `scale_preserve`: inverse source-year frequency, normalized per target, then multiplied into ordinary mean aggregation.
- `strict`: empirical GSMP group average, where each observed neighbor-year group contributes equally.

Weights are computed on CPU and cached under:

```text
cache/gsmp/
```

## Environment

The official TAPE README recommends:

```text
Python 3.8
PyTorch 1.12.1
CUDA 11.3
pytorch-sparse, pytorch-scatter, pytorch-cluster, pyg
ogb
dgl cu113
yacs
transformers
accelerate
```

The scripts do not regenerate LLM explanations or fine-tune language models.
