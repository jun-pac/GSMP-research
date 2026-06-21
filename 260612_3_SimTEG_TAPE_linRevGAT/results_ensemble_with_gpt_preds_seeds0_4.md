# Results: Ensemble With GPT Predictions, Seeds 0-4

## Experiment Setting

This file summarizes the `SimTeG+TAPE+linearRevGAT` GPT-prediction ensemble over seeds `0, 1, 2, 3, 4`.

The comparison is between:

- `baseline`: `SimTeG+TAPE+linearRevGAT`
- `gsmp_first_layer`: `SimTeG+TAPE+linearRevGAT+GSMP1`, with GSMP applied only in the first message-passing layer

The ensemble inputs are OGBN-Arxiv E5, OGBN-Arxiv RoBERTa, OGBN-Arxiv-TAPE E5, OGBN-Arxiv-TAPE RoBERTa, and GPT-prediction features. The ensemble weights are `2:2:1:1:1`.

## Source Files

- Seed `0`: `results/ensemble_with_gpt_preds_seed0.json`
- Seeds `1-3`: `results_ensemble_with_gpt_preds.json`
- Seed `4`: `results/ensemble_with_gpt_preds_seeds4_10_fallback.json`

## Main Results, Seeds 0-4

Validation and test accuracies are mean +/- sample standard deviation over five seeds.

| Method | Val accuracy | Test accuracy |
|---|---:|---:|
| Baseline + GPT ensemble | `0.78580489 +/- 0.00034583` | `0.78057322 +/- 0.00134371` |
| GSMP first-layer + GPT ensemble | `0.78516729 +/- 0.00050583` | `0.78180359 +/- 0.00116861` |

Delta, GSMP minus baseline:

| Metric | Delta |
|---|---:|
| Val accuracy | `-0.00063761` |
| Test accuracy | `+0.00123038` |

For seeds `0-4`, GSMP1 improves mean test accuracy but has lower mean validation accuracy.

## Per-Seed Results

| Seed | Baseline val | Baseline test | GSMP1 val | GSMP1 test | Delta val | Delta test |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.78539548` | `0.78044565` | `0.78526125` | `0.78165957` | `-0.00013423` | `+0.00121392` |
| 1 | `0.78610020` | `0.78028105` | `0.78459009` | `0.78275004` | `-0.00151012` | `+0.00246898` |
| 2 | `0.78606665` | `0.77908771` | `0.78519413` | `0.78309981` | `-0.00087251` | `+0.00401210` |
| 3 | `0.78599953` | `0.78277061` | `0.78485855` | `0.78133037` | `-0.00114098` | `-0.00144024` |
| 4 | `0.78546260` | `0.78028105` | `0.78593241` | `0.78017818` | `+0.00046981` | `-0.00010287` |

GSMP1 improves test accuracy on seeds `0`, `1`, and `2`; it is lower on seeds `3` and `4`.

## Leaderboard Baseline Note

The leaderboard `SimTeG+TAPE+RevGAT = 0.7803 +/- 0.0007` test accuracy is useful as context, but the primary GSMP baseline here is the matched no-GSMP `linearRevGAT` ensemble. The leaderboard model uses RevGAT learned attention, while this experiment applies GSMP to the first layer of a linearRevGAT harness.
