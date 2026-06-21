# Results Summary

This summarizes the completed full TAPE runs in `260610_2`. All full runs used
official TAPE `TA_P_E` features, 3 seeds, 200 epochs, 1 V100 GPU, and the
`ogbn-arxiv` dataset.

Smoke runs are excluded from the main comparison because they used only 5
epochs and `ogb` features.

## Main Comparison

| Method | Val Acc | Test Acc | Slurm time | GPU-hours | Billing-hours | Max CPU RAM | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TAPE+RevGAT baseline | `0.7789 +/- 0.0016` | `0.7707 +/- 0.0026` | `15m13s` | `0.254` | `2.03` | `~5.1 GB` | `~8.0 GB` |
| TAPE+LinearRevGAT | `0.7764 +/- 0.0020` | `0.7668 +/- 0.0035` | `8m54s` | `0.148` | `1.19` | `~5.1 GB` | `~6.7 GB` |
| TAPE+LinearRevGAT+GSMP | `0.7767 +/- 0.0004` | `0.7678 +/- 0.0026` | `9m00s` | `0.150` | `1.20` | `~5.0 GB` | `~6.7 GB` |

## LinearRevGAT vs LinearRevGAT+GSMP

GSMP slightly improved the final TAPE ensemble:

```text
Val improvement:  +0.0003
Test improvement: +0.0010
```

The improvement is positive but small. It should be described cautiously unless
more repeats or ablations show a consistent effect.

## Feature-Level Breakdown

| Method | Feature | Val Acc | Test Acc |
| --- | --- | ---: | ---: |
| TAPE+RevGAT baseline | TA | `0.7682 +/- 0.0014` | `0.7539 +/- 0.0012` |
| TAPE+RevGAT baseline | P | `0.7579 +/- 0.0011` | `0.7566 +/- 0.0020` |
| TAPE+RevGAT baseline | E | `0.7660 +/- 0.0013` | `0.7579 +/- 0.0028` |
| TAPE+RevGAT baseline | ensemble | `0.7789 +/- 0.0016` | `0.7707 +/- 0.0026` |
| TAPE+LinearRevGAT | TA | `0.7659 +/- 0.0022` | `0.7501 +/- 0.0042` |
| TAPE+LinearRevGAT | P | `0.7564 +/- 0.0004` | `0.7522 +/- 0.0032` |
| TAPE+LinearRevGAT | E | `0.7653 +/- 0.0020` | `0.7579 +/- 0.0040` |
| TAPE+LinearRevGAT | ensemble | `0.7764 +/- 0.0020` | `0.7668 +/- 0.0035` |
| TAPE+LinearRevGAT+GSMP | TA | `0.7642 +/- 0.0028` | `0.7483 +/- 0.0054` |
| TAPE+LinearRevGAT+GSMP | P | `0.7562 +/- 0.0014` | `0.7539 +/- 0.0021` |
| TAPE+LinearRevGAT+GSMP | E | `0.7638 +/- 0.0012` | `0.7550 +/- 0.0080` |
| TAPE+LinearRevGAT+GSMP | ensemble | `0.7767 +/- 0.0004` | `0.7678 +/- 0.0026` |

## Completed Jobs

| Job ID | Job name | Method | State | Elapsed |
| ---: | --- | --- | --- | ---: |
| `48153926` | `tape-revgat-base` | TAPE+RevGAT baseline | `COMPLETED` | `15m13s` |
| `48154196` | `tape-linear-ablate` | TAPE+LinearRevGAT | `COMPLETED` | `8m54s` |
| `48154335` | `tape-linear-gsmp` | TAPE+LinearRevGAT+GSMP | `COMPLETED` | `9m00s` |

## Output Locations

| Method | Result directory | Log |
| --- | --- | --- |
| TAPE+RevGAT baseline | `results/tape_revgat_gsmp/20260610_112640_baseline_RevGAT` | `logs/tape-revgat-base_48153926.out` |
| TAPE+LinearRevGAT | `results/tape_revgat_gsmp/20260610_114935_linear_LinearRevGAT` | `logs/tape-linear-ablate_48154196.out` |
| TAPE+LinearRevGAT+GSMP | `results/tape_revgat_gsmp/20260610_120753_gsmp_LinearRevGAT` | `logs/tape-linear-gsmp_48154335.out` |

## Short Takeaway

LinearRevGAT is much faster than the full RevGAT baseline and gets close to its
accuracy. Adding GSMP to LinearRevGAT had almost no resource overhead and gave a
small positive ensemble gain, especially on test accuracy.
