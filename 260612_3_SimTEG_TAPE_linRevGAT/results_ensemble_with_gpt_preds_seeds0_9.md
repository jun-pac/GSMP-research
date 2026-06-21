# Results: Ensemble With GPT Predictions, Seeds 0-9

## Experiment Setting

This file summarizes the `SimTeG+TAPE+linearRevGAT` GPT-prediction ensemble over seeds `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`.

The comparison is between:

- `baseline`: `SimTeG+TAPE+linearRevGAT`
- `gsmp_first_layer`: `SimTeG+TAPE+linearRevGAT+GSMP1`, with GSMP applied only in the first message-passing layer

The ensemble inputs are OGBN-Arxiv E5, OGBN-Arxiv RoBERTa, OGBN-Arxiv-TAPE E5, OGBN-Arxiv-TAPE RoBERTa, and GPT-prediction features. The ensemble weights are `2:2:1:1:1`.

## Source Files

- Seed `0`: `results/ensemble_with_gpt_preds_seed0.json`
- Seeds `1-3`: `results_ensemble_with_gpt_preds.json`
- Seeds `4-9`: `results/ensemble_with_gpt_preds_seeds4_10_fallback.json`

## Main Results, Seeds 0-9

Validation and test accuracies are mean +/- sample standard deviation over ten seeds.

| Method | Val accuracy | Test accuracy |
|---|---:|---:|
| Baseline + GPT ensemble | `0.78570086 +/- 0.00048306` | `0.78122750 +/- 0.00118150` |
| GSMP first-layer + GPT ensemble | `0.78514380 +/- 0.00078784` | `0.78129951 +/- 0.00128765` |

Delta, GSMP minus baseline:

| Metric | Delta |
|---|---:|
| Val accuracy | `-0.00055707` |
| Test accuracy | `+0.00007201` |

For seeds `0-9`, the test result is essentially tied. GSMP1 is ahead by only `+0.00007201` absolute on mean test accuracy, while the baseline has higher mean validation accuracy.

## Per-Seed Results

| Seed | Baseline val | Baseline test | GSMP1 val | GSMP1 test | Delta val | Delta test |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.78539548` | `0.78044565` | `0.78526125` | `0.78165957` | `-0.00013423` | `+0.00121392` |
| 1 | `0.78610020` | `0.78028105` | `0.78459009` | `0.78275004` | `-0.00151012` | `+0.00246898` |
| 2 | `0.78606665` | `0.77908771` | `0.78519413` | `0.78309981` | `-0.00087251` | `+0.00401210` |
| 3 | `0.78599953` | `0.78277061` | `0.78485855` | `0.78133037` | `-0.00114098` | `-0.00144024` |
| 4 | `0.78546260` | `0.78028105` | `0.78593241` | `0.78017818` | `+0.00046981` | `-0.00010287` |
| 5 | `0.78489211` | `0.78186532` | `0.78616732` | `0.78079542` | `+0.00127521` | `-0.00106989` |
| 6 | `0.78589886` | `0.78122750` | `0.78549616` | `0.77943748` | `-0.00040270` | `-0.00179001` |
| 7 | `0.78512702` | `0.78250314` | `0.78371757` | `0.78071312` | `-0.00140944` | `-0.00179001` |
| 8 | `0.78563039` | `0.78155669` | `0.78428806` | `0.78009588` | `-0.00134233` | `-0.00146082` |
| 9 | `0.78643579` | `0.78225624` | `0.78593241` | `0.78293521` | `-0.00050337` | `+0.00067897` |

GSMP1 improves test accuracy on seeds `0`, `1`, `2`, and `9`; it is lower on the other six seeds.

## Leaderboard Baseline Note

The leaderboard `SimTeG+TAPE+RevGAT = 0.7803 +/- 0.0007` test accuracy is useful as context, but the primary GSMP baseline here is the matched no-GSMP `linearRevGAT` ensemble. The leaderboard model uses RevGAT learned attention, while this experiment applies GSMP to the first layer of a linearRevGAT harness.
