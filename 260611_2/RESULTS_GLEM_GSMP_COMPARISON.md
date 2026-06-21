# GLEM RevGAT GSMP Results

Generated from `results/glem_revgat_gsmp/*/seed_summary.csv` in this
workspace.

## Scope And Caveat

These runs compare three GNN-side variants on `ogbn-arxiv`:

1. `GLEM+RevGAT`
2. `GLEM+LinearRevGAT`
3. `GLEM+LinearRevGAT+GSMP`

The main metric is `test_at_best_val`. `best_raw_test` is diagnostic only and
should not be used for model selection.

Important: these results are not a full reproduction of the official GLEM
leaderboard number:

```text
official GLEM+RevGAT validation accuracy: 0.7746 +/- 0.0018
official GLEM+RevGAT test accuracy:       0.7694 +/- 0.0025
```

The current runs are cheaper frozen-output GNN-pretrain ablations. They use:

- `stage = GNN-pretrain`
- `freeze_lm_outputs_for_gnn_ablation = T`
- `full_em_retrain = F`
- `budget_guard_no_lm_work = T`

The logs also show that cached/linked DeBERTa artifacts were reused and the
budget guard stopped before EM LM training or LM inference:

```text
[BUDGET_GUARD] cached LM artifacts and GNN pretraining are done; stopping before EM LM training/inference.
```

Therefore, the fair scientific comparison here is:

```text
GLEM+LinearRevGAT vs GLEM+LinearRevGAT+GSMP
```

`GLEM+RevGAT` is included as a local anchor, but this anchor is not the full
leaderboard pipeline.

## Full Runs, Seeds 0-3

All full runs used an epoch budget of `2000` with early stopping patience
`300`. In practice, all runs early-stopped around epoch 313-329.

| Seed | Method | Stage | Epoch Budget | Early Stop | Best Epoch | Val Acc | Test @ Best Val | Best Raw Test | Runtime (s) | Peak GPU GB |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `GLEM+RevGAT` | GNN-pretrain | 2000 | 300 | 23 | 0.769053 | 0.760591 | 0.765303 | 213.0 | 7.640 |
| 0 | `GLEM+LinearRevGAT` | GNN-pretrain | 2000 | 300 | 29 | 0.767811 | 0.760138 | 0.763410 | 154.7 | 6.021 |
| 0 | `GLEM+LinearRevGAT+GSMP` | GNN-pretrain | 2000 | 300 | 19 | 0.766167 | 0.756291 | 0.762237 | 149.3 | 6.021 |
| 1 | `GLEM+RevGAT` | GNN-pretrain | 2000 | 300 | 15 | 0.767408 | 0.756867 | 0.762422 | 212.3 | 7.640 |
| 1 | `GLEM+LinearRevGAT` | GNN-pretrain | 2000 | 300 | 23 | 0.767408 | 0.762587 | 0.763965 | 155.1 | 6.021 |
| 1 | `GLEM+LinearRevGAT+GSMP` | GNN-pretrain | 2000 | 300 | 25 | 0.767677 | 0.760159 | 0.762319 | 155.3 | 6.021 |
| 2 | `GLEM+RevGAT` | GNN-pretrain | 2000 | 300 | 21 | 0.768381 | 0.762360 | 0.763965 | 213.7 | 7.640 |
| 2 | `GLEM+LinearRevGAT` | GNN-pretrain | 2000 | 300 | 24 | 0.767845 | 0.760323 | 0.763513 | 155.1 | 6.021 |
| 2 | `GLEM+LinearRevGAT+GSMP` | GNN-pretrain | 2000 | 300 | 24 | 0.767677 | 0.760591 | 0.762628 | 153.6 | 6.021 |
| 3 | `GLEM+RevGAT` | GNN-pretrain | 2000 | 300 | 13 | 0.768113 | 0.755303 | 0.763080 | 207.8 | 7.640 |
| 3 | `GLEM+LinearRevGAT` | GNN-pretrain | 2000 | 300 | 18 | 0.768818 | 0.754974 | 0.764521 | 151.6 | 6.021 |
| 3 | `GLEM+LinearRevGAT+GSMP` | GNN-pretrain | 2000 | 300 | 20 | 0.766804 | 0.762546 | 0.762546 | 152.6 | 6.021 |

## Aggregate, Seeds 0-3

| Method | Seeds | Mean Val | Val Std | Mean Test @ Best Val | Test Std | Mean Best Raw Test | Mean Runtime (s) | Peak GPU GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `GLEM+RevGAT` | 0,1,2,3 | 0.768239 | 0.000589 | 0.758780 | 0.002822 | 0.763693 | 211.7 | 7.640 |
| `GLEM+LinearRevGAT` | 0,1,2,3 | 0.767970 | 0.000518 | 0.759506 | 0.002788 | 0.763852 | 154.1 | 6.021 |
| `GLEM+LinearRevGAT+GSMP` | 0,1,2,3 | 0.767081 | 0.000637 | 0.759897 | 0.002268 | 0.762432 | 152.7 | 6.021 |

For seeds 0-3, `GLEM+LinearRevGAT+GSMP` has the best mean
`test_at_best_val`, but the margin over `GLEM+LinearRevGAT` is very small:

```text
0.759897 - 0.759506 = +0.000391
```

## Aggregate, Seeds 1-3

Seed 0 was the worst GSMP case. The seed 1-3 subset is therefore useful as a
secondary view, but it should not replace the full seed 0-3 table.

| Method | Seeds | Mean Val | Val Std | Mean Test @ Best Val | Test Std | Mean Best Raw Test |
|---|---|---:|---:|---:|---:|---:|
| `GLEM+RevGAT` | 1,2,3 | 0.767968 | 0.000410 | 0.758177 | 0.003026 | 0.763156 |
| `GLEM+LinearRevGAT` | 1,2,3 | 0.768024 | 0.000589 | 0.759295 | 0.003192 | 0.763999 |
| `GLEM+LinearRevGAT+GSMP` | 1,2,3 | 0.767386 | 0.000411 | 0.761098 | 0.001038 | 0.762498 |

For seeds 1-3, GSMP is more favorable:

```text
0.761098 - 0.759295 = +0.001804
```

## Pairwise Differences

All differences below are on `test_at_best_val`, except the final column.
Positive means the left method is better.

| Seeds | Comparison | Mean Test Difference | Per-Seed Test Differences | Mean Val Difference |
|---|---|---:|---|---:|
| 0-3 | `GLEM+LinearRevGAT+GSMP` - `GLEM+LinearRevGAT` | +0.000391 | -0.003847, -0.002428, +0.000267, +0.007572 | -0.000889 |
| 0-3 | `GLEM+LinearRevGAT+GSMP` - `GLEM+RevGAT` | +0.001116 | -0.004300, +0.003292, -0.001769, +0.007242 | -0.001158 |
| 0-3 | `GLEM+LinearRevGAT` - `GLEM+RevGAT` | +0.000725 | -0.000453, +0.005720, -0.002037, -0.000329 | -0.000268 |
| 1-3 | `GLEM+LinearRevGAT+GSMP` - `GLEM+LinearRevGAT` | +0.001804 | -0.002428, +0.000267, +0.007572 | -0.000638 |
| 1-3 | `GLEM+LinearRevGAT+GSMP` - `GLEM+RevGAT` | +0.002922 | +0.003292, -0.001769, +0.007242 | -0.000582 |
| 1-3 | `GLEM+LinearRevGAT` - `GLEM+RevGAT` | +0.001118 | +0.005720, -0.002037, -0.000329 | +0.000056 |

## Interpretation

The current evidence is mixed but mildly positive for GSMP in this frozen-output
GNN-pretrain setting:

- Across seeds 0-3, GSMP improves mean `test_at_best_val` by only `+0.000391`
  over LinearRevGAT.
- Across seeds 1-3, GSMP improves mean `test_at_best_val` by `+0.001804`.
- GSMP is worse on seeds 0 and 1, slightly better on seed 2, and clearly better
  on seed 3.
- GSMP reduces variance in `test_at_best_val` compared with LinearRevGAT in the
  seed 0-3 aggregate.
- GSMP does not improve validation accuracy on average. Its advantage comes from
  `test_at_best_val`, not from higher mean validation accuracy.

The appropriate conclusion is therefore conservative:

```text
In the frozen-output GLEM GNN-pretrain ablation, GSMP is roughly neutral to
slightly positive on test-at-best-validation, with a small seed-dependent gain.
This does not establish that GSMP improves the full GLEM leaderboard pipeline.
```

## Smoke Runs

Earlier smoke runs used only 3 epochs and are not scientific evidence. They are
listed here only for completeness.

| Seed | Method | Epochs | Val Acc | Test @ Best Val | Best Raw Test |
|---:|---|---:|---:|---:|---:|
| 0 | `GLEM+RevGAT` | 3 | 0.495923 | 0.483941 | 0.483941 |
| 0 | `GLEM+LinearRevGAT` | 3 | 0.455686 | 0.419563 | 0.419563 |
| 0 | `GLEM+LinearRevGAT+GSMP` | 3 | 0.452733 | 0.417896 | 0.417896 |

