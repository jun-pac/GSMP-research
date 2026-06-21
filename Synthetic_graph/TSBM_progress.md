# TSBM GSMP Progress

Strategy: make the synthetic graph follow the latest TSBM description while
stress-testing temporal message-passing skew. Labels and timestamps are sampled
independently, node features are label-conditioned Gaussians, and edge
probabilities use `P0[y,y~] * gamma[y,y~]^|t-t~|`. The selected stress setting
uses high same-time cross-label probability with very fast cross-label temporal
decay, plus lower same-label probability with slow temporal decay. Ordinary
mean aggregation is then dominated by abundant same-time noisy neighbors, while
GSMP rebalances each target node's source-time buckets and recovers the
cross-time homophilous signal.

Current best setting: `data_tsbm/iter01.pt`. The 3-seed confirmation below
exceeds the requested 10-point absolute test-accuracy gap for all requested
pairs.

## Run 2026-06-21 08:28:36

- dataset: `data_tsbm/iter01.pt`
- result_json: `results_tsbm/iter01_seed0_results.json`
- result_tsv: `results_tsbm/iter01_seed0_rows.tsv`
- graph setting: nodes=3000, edges=163276, classes=5, times=12, same_p0=0.018, cross_p0=0.22, same_gamma=0.94, cross_gamma=0.025, feature_scale=0.55, feature_noise=[1.2, 2.0]
- split: train_times=[0, 1, 2, 3, 4, 5, 6], val_times=[7, 8], test_times=[9, 10, 11]
- train setting: models=all, seeds=0, epochs=150, hidden_dim=64, lr=0.01, dropout=0.25, self_loop=False, gsmp_scope=all, gsmp_mode=scale_preserve

| model | baseline test | GSMP test | delta | relative delta |
| --- | ---: | ---: | ---: | ---: |
| sgc | 0.1747 +/- 0.0000 | 0.7520 +/- 0.0000 | +0.5773 | +330.53% |
| gcn | 0.1973 +/- 0.0000 | 0.8453 +/- 0.0000 | +0.6480 | +328.38% |
| graphsage | 0.3747 +/- 0.0000 | 0.8840 +/- 0.0000 | +0.5093 | +135.94% |
| linearrevgat | 0.2373 +/- 0.0000 | 0.9093 +/- 0.0000 | +0.6720 | +283.15% |

Per-seed rows are in the TSV above.

## Run 2026-06-21 08:44:43

- dataset: `data_tsbm/iter01.pt`
- result_json: `results_tsbm/iter01_seeds0_2_results.json`
- result_tsv: `results_tsbm/iter01_seeds0_2_rows.tsv`
- graph setting: nodes=3000, edges=163276, classes=5, times=12, same_p0=0.018, cross_p0=0.22, same_gamma=0.94, cross_gamma=0.025, feature_scale=0.55, feature_noise=[1.2, 2.0]
- split: train_times=[0, 1, 2, 3, 4, 5, 6], val_times=[7, 8], test_times=[9, 10, 11]
- train setting: models=all, seeds=0,1,2, epochs=160, hidden_dim=64, lr=0.01, dropout=0.25, self_loop=False, gsmp_scope=all, gsmp_mode=scale_preserve

| model | baseline test | GSMP test | delta | relative delta |
| --- | ---: | ---: | ---: | ---: |
| sgc | 0.2036 +/- 0.0220 | 0.7573 +/- 0.0022 | +0.5538 | +272.05% |
| gcn | 0.1951 +/- 0.0072 | 0.8471 +/- 0.0025 | +0.6520 | +334.17% |
| graphsage | 0.3569 +/- 0.0242 | 0.8911 +/- 0.0082 | +0.5342 | +149.69% |
| linearrevgat | 0.2453 +/- 0.0214 | 0.8991 +/- 0.0115 | +0.6538 | +266.49% |

Per-seed rows are in the TSV above.
