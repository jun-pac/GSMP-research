# Results: Ensemble With GPT Predictions, Seeds 1-10

## Experiment Setting

This file summarizes the `SimTeG+TAPE+linearRevGAT` GPT-prediction ensemble over seeds `1, 2, 3, 4, 5, 6, 7, 8, 9, 10`. It is intentionally separate from `results_ensemble_with_gpt_preds.md`, which contains only the earlier seeds `1, 2, 3` summary.

The comparison is between:

- `baseline`: `SimTeG+TAPE+linearRevGAT`
- `gsmp_first_layer`: `SimTeG+TAPE+linearRevGAT+GSMP1`, with GSMP applied only in the first message-passing layer

The ensemble inputs are:

1. OGBN-Arxiv E5 embeddings
2. OGBN-Arxiv RoBERTa embeddings
3. OGBN-Arxiv-TAPE E5 embeddings
4. OGBN-Arxiv-TAPE RoBERTa embeddings
5. GPT-prediction features

The ensemble weights are `2:2:1:1:1`. Logits are converted to softmax probabilities, weighted, averaged, and then converted to labels by argmax. P-GSMP is not included because GPT-pred label features are not real-valued node embeddings in the way P-GSMP assumes.

## Source Files

Seeds `1-3`:

- `results_ensemble_with_gpt_preds.json`
- `results_ensemble_with_gpt_preds.csv`

Seeds `4-10`:

- `results/ensemble_with_gpt_preds_seeds4_10_fallback.json`
- `results/ensemble_with_gpt_preds_seeds4_10_fallback.csv`

For seeds `4-10`, the GSMP text-component logits were split between the original prefix and a retry prefix:

- Main prefix: `20260618_183228_seeds4_10`
- Retry prefix: `20260618_183228_seeds4_10_retry_nop0240`

The seeds `4-10` saved summary uses the main prefix where available and falls back to the retry prefix for missing GSMP text logits.

## Main Results, Seeds 1-10

Validation and test accuracies are mean +/- sample standard deviation over ten seeds.

| Method | Val accuracy | Test accuracy |
|---|---:|---:|
| Baseline + GPT ensemble | `0.78584852 +/- 0.00059260` | `0.78126247 +/- 0.00116078` |
| GSMP first-layer + GPT ensemble | `0.78497265 +/- 0.00093217` | `0.78087567 +/- 0.00176503` |

Delta, GSMP minus baseline:

| Metric | Delta |
|---|---:|
| Val accuracy | `-0.00087587` |
| Test accuracy | `-0.00038681` |

Across seeds `1-10`, the GPT ensemble does not show a GSMP improvement. The no-GSMP baseline is slightly higher on both validation and test accuracy.

## Per-Seed Results

| Seed | Baseline val | Baseline test | GSMP1 val | GSMP1 test | Delta val | Delta test |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.78610020` | `0.78028105` | `0.78459009` | `0.78275004` | `-0.00151012` | `+0.00246898` |
| 2 | `0.78606665` | `0.77908771` | `0.78519413` | `0.78309981` | `-0.00087251` | `+0.00401210` |
| 3 | `0.78599953` | `0.78277061` | `0.78485855` | `0.78133037` | `-0.00114098` | `-0.00144024` |
| 4 | `0.78546260` | `0.78028105` | `0.78593241` | `0.78017818` | `+0.00046981` | `-0.00010287` |
| 5 | `0.78489211` | `0.78186532` | `0.78616732` | `0.78079542` | `+0.00127521` | `-0.00106989` |
| 6 | `0.78589886` | `0.78122750` | `0.78549616` | `0.77943748` | `-0.00040270` | `-0.00179001` |
| 7 | `0.78512702` | `0.78250314` | `0.78371757` | `0.78071312` | `-0.00140944` | `-0.00179001` |
| 8 | `0.78563039` | `0.78155669` | `0.78428806` | `0.78009588` | `-0.00134233` | `-0.00146082` |
| 9 | `0.78643579` | `0.78225624` | `0.78593241` | `0.78293521` | `-0.00050337` | `+0.00067897` |
| 10 | `0.78687204` | `0.78079542` | `0.78354978` | `0.77742115` | `-0.00332226` | `-0.00337428` |

GSMP1 improves test accuracy on seeds `1`, `2`, and `9`; it is lower on the other seven seeds.

## Interpretation

The earlier seeds `1-3` summary was favorable to GSMP on test accuracy. After adding seeds `4-10`, that advantage does not hold. The ten-seed result is a small baseline advantage:

```text
Baseline test: 0.78126247 +/- 0.00116078
GSMP1 test:    0.78087567 +/- 0.00176503
Delta:         -0.00038681
```

This is close enough that it should be described as "no reliable improvement" rather than a strong negative result, but it is not evidence that GSMP improves the GPT ensemble.

## Leaderboard Baseline Note

The leaderboard number `SimTeG+TAPE+RevGAT = 0.7803 +/- 0.0007` test accuracy is useful as an external anchor, but it should not be treated as the primary baseline for the GSMP comparison in this harness.

The fair primary baseline is the matched no-GSMP `linearRevGAT` ensemble run with the same components, seeds, training code, saved-logit ensemble rule, and local environment. That matched baseline is the `Baseline + GPT ensemble` row above.

Reasons not to use the leaderboard number as the direct baseline:

- The leaderboard model is `RevGAT` with learned attention, while this experiment is `linearRevGAT`.
- GSMP here is inserted as first-layer source-year-balanced edge reweighting; it is not applied to learned RevGAT attention coefficients.
- The leaderboard number is an external reproduction anchor, not a same-run, same-seed ablation against this GSMP implementation.
- Using the external `0.7803 +/- 0.0007` as the baseline would make the comparison depend on architecture and protocol differences, not only on GSMP.

It is legitimate to report the leaderboard number for context, for example: "The official SimTeG+TAPE+RevGAT anchor is `0.7803 +/- 0.0007`; our matched linearRevGAT+GPT baseline is `0.7813 +/- 0.0012` over seeds `1-10`." But claims about whether GSMP helps should use the matched no-GSMP linearRevGAT baseline.
