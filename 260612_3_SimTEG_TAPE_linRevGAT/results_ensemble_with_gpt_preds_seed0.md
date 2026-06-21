# Results: Ensemble With GPT Predictions, Seed 0

## Experiment Setting

This file summarizes the seed `0` `SimTeG+TAPE+linearRevGAT` GPT-prediction ensemble. It uses the same setup as the seeds `1-10` summary:

- `baseline`: `SimTeG+TAPE+linearRevGAT`
- `gsmp_first_layer`: `SimTeG+TAPE+linearRevGAT+GSMP1`
- Components: OGBN-Arxiv E5, OGBN-Arxiv RoBERTa, OGBN-Arxiv-TAPE E5, OGBN-Arxiv-TAPE RoBERTa, and GPT-prediction features
- Ensemble weights: `2:2:1:1:1`

Run prefix:

```text
20260620_1452_seed0_gpt_ensemble
```

All ten Slurm array tasks completed successfully with exit code `0:0`.

## Output Files

- Machine-readable result: `results/ensemble_with_gpt_preds_seed0.json`
- Per-seed CSV: `results/ensemble_with_gpt_preds_seed0.csv`

## Main Result

| Method | Val accuracy | Test accuracy |
|---|---:|---:|
| Baseline + GPT ensemble | `0.78539548` | `0.78044565` |
| GSMP first-layer + GPT ensemble | `0.78526125` | `0.78165957` |

Delta, GSMP minus baseline:

| Metric | Delta |
|---|---:|
| Val accuracy | `-0.00013423` |
| Test accuracy | `+0.00121392` |

On seed `0`, GSMP1 improves test accuracy while validation accuracy is nearly tied but slightly lower.

## Context: Seeds 0-10

If seed `0` is combined with the existing seeds `1-10`, the aggregate becomes:

| Method | Val accuracy | Test accuracy |
|---|---:|---:|
| Baseline + GPT ensemble | `0.78580733 +/- 0.00057854` | `0.78118822 +/- 0.00112841` |
| GSMP first-layer + GPT ensemble | `0.78499889 +/- 0.00088860` | `0.78094693 +/- 0.00169106` |

Combined seeds `0-10` delta, GSMP minus baseline:

| Metric | Delta |
|---|---:|
| Test accuracy | `-0.00024129` |

Seed `0` is favorable to GSMP, but the aggregate over seeds `0-10` still shows a small baseline advantage.
