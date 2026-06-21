# LD RevGAT GSMP on ogbn-arxiv

This folder is an isolated LD workspace for comparing:

1. `LD+RevGAT`: official LD leaderboard baseline.
2. `LD+LinearRevGAT`: RevGAT shell with learned attention aggregation replaced by fixed linear aggregation.
3. `LD+LinearRevGAT+GSMP`: the same linear aggregation using publication-year GSMP weights.

The official LD repository is cloned under:

```text
260611/upstream/LD
```

No full run is launched automatically. Defaults are smoke-first to protect GPU hours, credits, and memory.

## Official Target

The LD paper reports for `LD+RevGAT` on `ogbn-arxiv`:

```text
validation accuracy: 0.7762 +/- 0.0008
test accuracy:       0.7726 +/- 0.0017
```

Reproduce this anchor before spending full GPU time on GSMP.

## Modified Files

- `upstream/LD/transformer/gnn/revgat/gsmp.py`: CPU GSMP and linear edge-weight preprocessing with disk cache.
- `upstream/LD/transformer/gnn/revgat/loader.py`: preserves all node data through bidirection/self-loop preprocessing and attaches linear weights.
- `upstream/LD/transformer/gnn/revgat/model.py`: adds `LinearWeightedConv`, `LinearRevGATBlock`, and `LinearRevGAT`.
- `upstream/LD/transformer/conf/model/revgat.yaml`: adds `linear`, `use_gsmp`, `gsmp_norm`, and cache flags, all defaulting to official RevGAT behavior.
- `upstream/LD/transformer/main_bertgnn.py`: adds `[RESULT]`, `[SEED_SUMMARY]`, `[FINAL]` logging plus CSV/JSON result output.

## Added Files

- `scripts/run_ld_gsmp_experiments.sh`: one-command launcher.
- `scripts/run_one_ld_gnn.sh`: single-run wrapper around LD's `main_bertgnn.py`.
- `scripts/check_ld_assets.sh`: checks token files and cached LD hidden states.
- `scripts/monitor_latest.sh`: tails the newest Slurm log.
- `slurm/*.sbatch`: conservative one-GPU Slurm scripts.
- `tests/test_gsmp_weights.py`: deterministic GSMP unit test.
- `tools/check_baseline_gate.py`: blocks full GSMP if the baseline has not reproduced.
- `tools/compare_results.py`: aggregates result JSON files.

## Environment

Use the official LD environment when possible:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260611/upstream/LD
conda env create -f environment.yml
conda activate ld
```

The Slurm scripts contain an editable section:

```bash
source ~/.bashrc
conda activate "${LD_CONDA_ENV:-ld}"
```

Cluster account and partition lines are commented as `##SBATCH`; uncomment/edit them if your cluster requires them.

## LD Assets

Check whether token files and hidden states are present:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260611
bash scripts/check_ld_assets.sh
```

Expected defaults:

```text
LD_LM_PATH=$HOME/model/deberta-base
LD_TOKEN_FOLDER=/OGB/ogbn_arxiv/token/microsoft/deberta-base
```

In this workspace, the scripts first look for the downloaded local token folder:

```text
260611/downloads/ld_tokens/extracted/ogbn_arxiv/token/microsoft/deberta-base
```

The official `deberta-base/arxiv_revgat/hidden_state.pt` feature is linked into:

```text
$HOME/model/deberta-base/arxiv_seed0/hidden_state.pt
...
$HOME/model/deberta-base/arxiv_seed4/hidden_state.pt
```

If `hidden_state.pt` is missing, LD may regenerate it through the language model, which costs extra GPU time. To avoid that, place official/generated hidden states at:

```text
$LD_LM_PATH/arxiv_seed0/hidden_state.pt
$LD_LM_PATH/arxiv_seed1/hidden_state.pt
...
```

## Static Checks

Run these before any GPU job:

```bash
python tests/test_gsmp_weights.py
python -m py_compile upstream/LD/transformer/gnn/revgat/gsmp.py
python -m py_compile upstream/LD/transformer/gnn/revgat/loader.py
python -m py_compile upstream/LD/transformer/gnn/revgat/model.py
python -m py_compile upstream/LD/transformer/main_bertgnn.py
```

## Smoke Test

Smoke uses seed 0 and 3 epochs by default for:

```text
LD+RevGAT
LD+LinearRevGAT
LD+LinearRevGAT+GSMP(scale_preserve)
```

Run:

```bash
bash scripts/run_ld_gsmp_experiments.sh smoke
```

Monitor:

```bash
bash scripts/monitor_latest.sh
grep "\[RESULT\]" logs/<jobname>_<jobid>.out
grep "\[SEED_SUMMARY\]" logs/<jobname>_<jobid>.out
grep "\[FINAL\]" logs/<jobname>_<jobid>.out
watch -n 5 squeue -u $USER
```

## Full Run Order

First reproduce the official baseline:

```bash
FULL=1 bash scripts/run_ld_gsmp_experiments.sh baseline --full
```

If the baseline is close to the target, run the linear ablation:

```bash
FULL=1 bash scripts/run_ld_gsmp_experiments.sh linear --full
```

Then run GSMP:

```bash
FULL=1 bash scripts/run_ld_gsmp_experiments.sh gsmp --full
```

To run all three methods as one serial array:

```bash
FULL=1 CONFIRM_FULL=1 bash scripts/run_ld_gsmp_experiments.sh all --full
```

The full GSMP path runs `tools/check_baseline_gate.py` unless `SKIP_BASELINE_GATE=1` is set.

## Results

Each seed writes:

```text
results/ld_revgat_gsmp/<run_id>/config.json
results/ld_revgat_gsmp/<run_id>/epoch_logs.csv
results/ld_revgat_gsmp/<run_id>/seed_summary.csv
results/ld_revgat_gsmp/<run_id>/final_summary.json
```

Aggregate:

```bash
python tools/compare_results.py --results-root results/ld_revgat_gsmp
```

The main scientific number is:

```text
test_at_best_val
```

`best_raw_test` is diagnostic only and must not be used for model selection.

## GSMP Direction

The model uses the DGL graph after:

```text
dgl.to_bidirected(graph)
remove_self_loop()
add_self_loop()
```

GSMP weights are computed on that exact graph. For every DGL message edge `src -> dst`, the count is:

```text
count[dst, year[src]]
```

`scale_preserve` multiplies the ordinary mean aggregation by a normalized inverse source-year frequency. `strict` directly implements the empirical average over observed neighbor-year groups.

GSMP is not applied to learned attention coefficients. If `use_gsmp=true`, the code requires `linear=true`.

## Interpretation

The fair GSMP ablation is:

```text
LD+LinearRevGAT vs LD+LinearRevGAT+GSMP
```

`LD+RevGAT` is the official reproduction anchor and upper baseline, not the direct GSMP ablation baseline.

If `LD+LinearRevGAT+GSMP > LD+LinearRevGAT`, GSMP likely complements LD by correcting residual temporal neighbor-distribution bias.

If `LD+LinearRevGAT+GSMP < LD+LinearRevGAT`, GSMP may conflict with LD because LD features are trained around the original downstream graph-convolution behavior.

If the two are close, LD may already absorb much of the correction GSMP would otherwise provide.
