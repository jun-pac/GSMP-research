# Results: Ensemble With GPT Predictions

## Experiment Setting

We evaluate GSMP on OGBN-Arxiv using the `SimTeG+TAPE+linearRevGAT` experimental harness. The comparison is between the baseline `linearRevGAT` model and `linearRevGAT+GSMP1`, where GSMP is applied only in the first message-passing layer as source-year-balanced edge reweighting. P-GSMP is not included in this comparison.

The graph is the OGBN-Arxiv citation graph with bidirected edges and self-loops. All runs use cached node features rather than language-model fine-tuning. The five ensemble inputs are:

1. OGBN-Arxiv E5 embeddings
2. OGBN-Arxiv RoBERTa embeddings
3. OGBN-Arxiv-TAPE E5 embeddings
4. OGBN-Arxiv-TAPE RoBERTa embeddings
5. GPT-prediction features

Training uses seeds `1, 2, 3` and `200` epochs per run. The GNN uses 2 layers, 2 heads, hidden size 256, dropout 0.58, input dropout 0.37, learning rate 0.002, weight decay 0, label smoothing 0.02, 2 label-propagation iterations, and labels as input features. The reported single-component result is test accuracy at the best validation epoch, averaged over the three seeds.

The ensemble follows the SimTeG paper/code strategy: apply softmax to each saved logits tensor, compute a weighted average of probabilities, and take the argmax. The ensemble weights are `2:2:1:1:1` for the five inputs listed above.

## Single-Component Results

Validation and test accuracies are mean +/- standard deviation over seeds `1, 2, 3`.

| Ensemble input | Baseline val | Baseline test | GSMP1 val | GSMP1 test |
|---|---:|---:|---:|---:|
| OGBN-Arxiv E5 | 0.7782 +/- 0.0010 | 0.7694 +/- 0.0041 | 0.7768 +/- 0.0008 | 0.7718 +/- 0.0029 |
| OGBN-Arxiv RoBERTa | 0.7760 +/- 0.0011 | 0.7726 +/- 0.0022 | 0.7750 +/- 0.0007 | 0.7727 +/- 0.0010 |
| OGBN-Arxiv-TAPE E5 | 0.7721 +/- 0.0002 | 0.7645 +/- 0.0009 | 0.7724 +/- 0.0004 | 0.7669 +/- 0.0021 |
| OGBN-Arxiv-TAPE RoBERTa | 0.7728 +/- 0.0011 | 0.7670 +/- 0.0022 | 0.7726 +/- 0.0010 | 0.7693 +/- 0.0019 |
| GPT-prediction features | 0.7678 +/- 0.0011 | 0.7633 +/- 0.0008 | 0.7683 +/- 0.0009 | 0.7629 +/- 0.0011 |

## Final Ensemble Results

| Method | Ensemble inputs | Weights | Val accuracy | Test accuracy |
|---|---|---:|---:|---:|
| Baseline | E5, RoBERTa, TAPE-E5, TAPE-RoBERTa, GPT-preds | 2:2:1:1:1 | 0.7861 +/- 0.0001 | 0.7807 +/- 0.0019 |
| GSMP first layer | E5, RoBERTa, TAPE-E5, TAPE-RoBERTa, GPT-preds | 2:2:1:1:1 | 0.7849 +/- 0.0003 | 0.7824 +/- 0.0009 |

GSMP first-layer improves ensemble test accuracy by `+0.0017` absolute over the baseline, while the baseline has slightly higher validation accuracy. This suggests that first-layer GSMP improves held-out test performance in the paper-style ensemble setting with GPT-prediction features included.

## Per-Seed Ensemble Results

| Method | Seed | Val accuracy | Test accuracy |
|---|---:|---:|---:|
| Baseline | 1 | 0.7861 | 0.7803 |
| Baseline | 2 | 0.7861 | 0.7791 |
| Baseline | 3 | 0.7860 | 0.7828 |
| GSMP first layer | 1 | 0.7846 | 0.7828 |
| GSMP first layer | 2 | 0.7852 | 0.7831 |
| GSMP first layer | 3 | 0.7849 | 0.7813 |

## Output Files

- `results_ensemble_with_gpt_preds.md`: paper-ready summary.
- `results_ensemble_with_gpt_preds.json`: machine-readable ensemble summary.
- `results_ensemble_with_gpt_preds.csv`: per-seed ensemble rows.
