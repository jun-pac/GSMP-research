# Env-Matched GSMP Stage 0 Summary

Completed run folder:
`260615_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0`

Source progress file:
`results/envmatched_p1_baseline_gsmp_ordered_live_progress.tsv`

This run compared the env-matched HGAMLP-HOPE baseline (`none`) against GSMP with impact active only in stage 0. Both methods used seeds 1, 2, and 3. The Slurm jobs completed successfully and the stderr files were empty.

## Aggregate Final-Stage Results

Metric is test accuracy at the validation-best checkpoint.

| Method | Seeds | Best validation mean | Test-at-best-validation mean |
| --- | ---: | ---: | ---: |
| `none` | 3 | 59.989% +/- 0.029 | 57.998% +/- 0.116 |
| `gsmp` | 3 | 59.769% +/- 0.090 | 57.743% +/- 0.061 |

GSMP trailed the matched baseline by 0.220 percentage points on validation and 0.255 percentage points on test-at-best-validation.

## Per-Seed Final-Stage Results

| Seed | Baseline test | GSMP test | GSMP - baseline |
| ---: | ---: | ---: | ---: |
| 1 | 58.130% | 57.753% | -0.377 pp |
| 2 | 57.953% | 57.798% | -0.155 pp |
| 3 | 57.910% | 57.677% | -0.234 pp |

## Latest Slurm Job

The last job was `48265422`, running `gsmp` seed 3. It finished on June 15, 2026 at 8:55 PM EDT.

| Checkpoint | Epoch | Validation | Test |
| --- | ---: | ---: | ---: |
| Best validation | 267 | 59.856% | 57.677% |
| Final logged | 300 | 59.805% | 57.782% |

## Takeaway

In this env-matched stage-0 setting, GSMP did not improve HGAMLP-HOPE over the matched baseline. The next priority is to test GSMP active across stages 0 through 3 using only GSMP seeds 1, 2, and 3, since baseline seeds 1 through 3 are already available here.
