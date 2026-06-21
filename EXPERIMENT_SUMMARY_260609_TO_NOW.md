# GSMP / STAMP Experiment Summary, 2026-06-09 To 2026-06-21

This document summarizes the experiment compartments created from `260609`
through the current workspace state on 2026-06-21.

## Scope And Main Goal

The common goal was to test whether timestamp-aware message passing improves
chronological generalization on citation/social temporal graphs without changing
the rest of each leaderboard or baseline pipeline.

The main scientific metric throughout is validation-selected test accuracy:
`test_at_best_val`, `test_at_best_valid`, or `best_test_at_best_val` depending
on the logger. Raw best-test values are diagnostic only.

Terms used below:

- `baseline`: original model/path with ordinary propagation.
- `SMP`: timestamp proximity weighting.
- `UMP`: future-to-past edge removal or temporal direction filtering.
- `GSMP`: target-side inverse source-timestamp-frequency weighting.
- `GSMP1` / `STAMP`: first-layer-only GSMP. This is the renamed first-layer
  method in the newer writing.
- `P-GSMP`: preprocessing-only GSMP feature smoothing before the normal model.

All accuracies are fractions unless explicitly shown as percentages. Reported
standard deviations are sample standard deviations when more than one seed was
available.

## High-Level Takeaways

| Area | Best matched conclusion |
| --- | --- |
| `ogbn-arxiv`, SimTeG/TAPE LinearRevGAT | First-layer GSMP / STAMP is the strongest positive arXiv signal. The 4-text-component ensemble improved by `+0.00182` test over baseline; the 10-seed GPT ensemble was essentially tied at `+0.00007`. |
| `ogbn-arxiv`, official TAPE LinearRevGAT | GSMP gave a small positive test gain: `+0.0010`. |
| `ogbn-arxiv`, GLEM frozen GNN-pretrain | GSMP was neutral to mildly positive on test: `+0.00039` over LinearRevGAT on seeds 0-3, stronger on seeds 1-3. |
| `ogbn-arxiv`, LD | GSMP was neutral to slightly worse: `-0.00071` over LinearRevGAT on seeds 1-3. |
| `ogbn-arxiv`, tunedGNN GCN | GSMP and P-GSMP did not beat the seed-0 baseline in the full selected runs. |
| Pokec temporal GCN | First-layer GSMP improved the 3-seed GCN test mean by about `+0.0010`. |
| `ogbn-mag`, HGAMLP-HOPE | GSMP did not improve the matched baseline. Stage-0 feature-only priority0 across seeds 0-4 and 9 was `-0.00123` test; env-matched stage0 feature+label was `-0.00255` test. |
| `ogbn-mag`, SeHGNN-HOPE canary | Baseline seed 1 remained ahead of SMP and GSMP paper-stack canaries. |
| `ogbn-mag`, RpHGNN/PSHGCN | Pipelines were prepared, but no reportable full result yet. |

## Timeline / Compartments

| Folder | Dataset | Model / pipeline | Main goal | Status |
| --- | --- | --- | --- | --- |
| `260609` | `ogbn-arxiv` | SimTeG/TAPE cached embeddings + GraphSAGE | First controlled SMP/UMP/GSMP comparison and leaderboard-style component runs | Completed preliminary runs |
| `260610` | `ogbn-arxiv` | SimTeG/TAPE cached features + LinearRevGAT | Test GSMP in five-component LinearRevGAT ensemble | Completed, GSMP hurt |
| `260610_2` | `ogbn-arxiv` | Official TAPE RevGAT / LinearRevGAT | Reproduce official TAPE feature stack and isolate GSMP on LinearRevGAT | Completed, small positive |
| `260611` | `ogbn-arxiv` | LD RevGAT / LinearRevGAT | Test GSMP inside LD frozen hidden-state pipeline | Completed seeds 1-3, slightly negative |
| `260611_2` | `ogbn-arxiv` | GLEM RevGAT / LinearRevGAT | Frozen-output GNN-pretrain ablation | Completed seeds 0-3, mildly positive on test |
| `260612` | `ogbn-arxiv` | tunedGNN GCN | Clean GCN ablation without attention-removal confound | Seed-0 full results, negative |
| `260612_2` | `ogbn-arxiv` | tunedGNN GCN + P-GSMP | Preprocessing-only GSMP feature ablation | Seed-0 full result, negative |
| `260612_3_SimTEG_TAPE_linRevGAT` | `ogbn-arxiv` | SimTeG/TAPE LinearRevGAT | Test first-layer GSMP / STAMP and P-GSMP with text/GPT ensembles | Completed 10-seed GPT comparison |
| `260613_pokec` | Pokec | Split/connectivity analysis | Build chronological Pokec protocol | Completed analysis |
| `260613_2_pokec_GCN` | Pokec | GCN vs first-layer GSMP-GCN | Paper-aligned GCN temporal comparison | Completed 3-seed 100-epoch run |
| `260614_2_ogbn_mag_HGAMLP_HOPE_v2` | `ogbn-mag` | HGAMLP-HOPE | Stage-aware GSMP/SMP propagation in HOPE | Priority0/1 partial-to-completed results |
| `260615_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0` | `ogbn-mag` | HGAMLP-HOPE env-matched | Matched baseline vs stage0 GSMP | Completed seeds 1-3 |
| `260616_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3` | `ogbn-mag` | HGAMLP-HOPE env-matched | Priority2 all-stage GSMP follow-up | Incomplete/early evidence poor |
| `260617_ogbn_mag_RpHGNN` | `ogbn-mag` | RpHGNN | Prepare baseline/SMP/GSMP paper propagation experiments | Pipeline only |
| `260617_2_ogbn_mag_PSHGCN` | `ogbn-mag` | PSHGCN | Prepare ComplEx + PSHGCN GSMP experiments | Blocked by missing ComplEx files |
| `260618_ogbn_mag_SeHGNN_HOPE` | `ogbn-mag` | SeHGNN-HOPE | Test GSMP paper-stack with HOPE-style SeHGNN | Canary seed 1 completed |

## 260609 - SimTeG/TAPE GraphSAGE On `ogbn-arxiv`

Goal: establish a clean first pipeline comparing `baseline`, `smp`, `ump`, and
`gsmp` on cached SimTeG/TAPE embeddings without language-model training.

Controlled single-source run:

- Dataset: `ogbn-arxiv`
- Embedding: `e5-large` cached `x_embs.pt`
- Model: GraphSAGE
- Seeds: `1, 2, 3`
- Epochs: 100
- Selection metric: validation accuracy

| Variant | Seeds | Val accuracy | Test at best val |
| --- | ---: | ---: | ---: |
| baseline | 3 | `0.76754 +/- 0.00066` | `0.75159 +/- 0.00271` |
| smp | 3 | `0.76723 +/- 0.00098` | `0.74879 +/- 0.00134` |
| ump | 3 | `0.76073 +/- 0.00033` | `0.75010 +/- 0.00220` |
| gsmp | 3 | `0.76792 +/- 0.00061` | `0.75212 +/- 0.00236` |

Takeaway: in this single-source GraphSAGE setting, GSMP was slightly positive
on test (`+0.00053` over baseline), while SMP and UMP were not.

Baseline calibration on seeds `42, 43, 44` produced:

| Variant | Val accuracy | Test at best val |
| --- | ---: | ---: |
| baseline | `0.77035 +/- 0.00102` | `0.76081 +/- 0.00062` |

Leaderboard-style component runs used seeds `42, 43, 44`.

| Component | Baseline test | GSMP test | Delta |
| --- | ---: | ---: | ---: |
| `ogbn-arxiv/e5-large` | `0.76373` | `0.76100` | `-0.00274` |
| `ogbn-arxiv/all-roberta-large-v1` | `0.76213` | `0.75648` | `-0.00566` |
| `ogbn-arxiv-tape/e5-large` | `0.75124` | `0.75216` | `+0.00092` |
| `ogbn-arxiv-tape/all-roberta-large-v1` | `0.74631` | `0.74894` | `+0.00263` |
| `ogbn-arxiv/gpt-preds` | `0.73496` | not run in this folder | n/a |

The 260609 ensemble results are not a perfectly matched comparison because the
baseline ensemble used five components including GPT predictions, while the GSMP
ensemble used four text components.

| Ensemble | Components | Weights | Val accuracy | Test accuracy |
| --- | ---: | --- | ---: | ---: |
| baseline | 5 | `2:2:1:1:1` | `0.77988 +/- 0.00048` | `0.77567 +/- 0.00182` |
| gsmp | 4 | `2:2:1:1` normalized over four components | `0.78093 +/- 0.00125` | `0.77375 +/- 0.00189` |

## 260610 - SimTeG/TAPE LinearRevGAT, Five Components

Goal: move from GraphSAGE to a LinearRevGAT shell using the SimTeG/TAPE
component stack, then test whether GSMP helps when attention is replaced by
fixed linear aggregation.

Important correction: despite some filenames saying `revgat`, this folder's
main comparison used `LinearRevGAT` vs `LinearRevGAT+GSMP`.

Setup:

- Dataset: `ogbn-arxiv`
- Seeds: `1, 2, 3`
- Components: `arxiv_e5`, `arxiv_roberta`, `arxiv_tape_e5`,
  `arxiv_tape_roberta`, `arxiv_gpt_preds`
- Ensemble weights: `2:2:1:1:1`
- Max epochs: 200
- Early stopping: min epoch 80, patience 40

| Method | Val accuracy | Test accuracy |
| --- | ---: | ---: |
| LinearRevGAT | `0.7813 +/- 0.0006` | `0.7789 +/- 0.0013` |
| LinearRevGAT+GSMP | `0.7810 +/- 0.0012` | `0.7764 +/- 0.0008` |

Delta: validation `-0.0002`, test `-0.0025`.

Component breakdown:

| Component | Linear test | GSMP test | Delta |
| --- | ---: | ---: | ---: |
| `arxiv_e5` | `0.7687` | `0.7670` | `-0.0017` |
| `arxiv_roberta` | `0.7686` | `0.7637` | `-0.0048` |
| `arxiv_tape_e5` | `0.7644` | `0.7626` | `-0.0017` |
| `arxiv_tape_roberta` | `0.7627` | `0.7650` | `+0.0023` |
| `arxiv_gpt_preds` | `0.7622` | `0.7570` | `-0.0052` |

Takeaway: GSMP slightly hurt this cached-feature five-component ensemble. The
likely interpretation is that temporal neighbor density itself was useful in
this setting, so flattening source-year mass removed signal.

Resource note:

- Slurm job `48147351`
- Completed in `45m32s` on one V100
- Approx. `0.759` GPU-hours and `4.55` billing-hours/credits
- Peak CPU RAM approx. `6.0 GB`

## 260610_2 - Official TAPE RevGAT / LinearRevGAT

Goal: repeat the arXiv experiment inside the official TAPE feature stack and
compare full RevGAT, LinearRevGAT, and LinearRevGAT+GSMP.

Setup:

- Dataset: `ogbn-arxiv`
- Official TAPE `TA_P_E` features
- Seeds: 3
- Epochs: 200
- GPU: one V100

| Method | Val accuracy | Test accuracy | Slurm time | GPU-hours | Billing-hours | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TAPE+RevGAT baseline | `0.7789 +/- 0.0016` | `0.7707 +/- 0.0026` | `15m13s` | `0.254` | `2.03` | `~8.0 GB` |
| TAPE+LinearRevGAT | `0.7764 +/- 0.0020` | `0.7668 +/- 0.0035` | `8m54s` | `0.148` | `1.19` | `~6.7 GB` |
| TAPE+LinearRevGAT+GSMP | `0.7767 +/- 0.0004` | `0.7678 +/- 0.0026` | `9m00s` | `0.150` | `1.20` | `~6.7 GB` |

Feature-level breakdown:

| Method | Feature | Val accuracy | Test accuracy |
| --- | --- | ---: | ---: |
| RevGAT baseline | TA | `0.7682 +/- 0.0014` | `0.7539 +/- 0.0012` |
| RevGAT baseline | P | `0.7579 +/- 0.0011` | `0.7566 +/- 0.0020` |
| RevGAT baseline | E | `0.7660 +/- 0.0013` | `0.7579 +/- 0.0028` |
| RevGAT baseline | ensemble | `0.7789 +/- 0.0016` | `0.7707 +/- 0.0026` |
| LinearRevGAT | ensemble | `0.7764 +/- 0.0020` | `0.7668 +/- 0.0035` |
| LinearRevGAT+GSMP | ensemble | `0.7767 +/- 0.0004` | `0.7678 +/- 0.0026` |

Takeaway: GSMP gave a small positive test gain over LinearRevGAT:
`+0.0010`.

## 260611 - LD RevGAT / LinearRevGAT

Goal: test GSMP in the LD `ogbn-arxiv` pipeline while keeping LD hidden states
and using a fair LinearRevGAT vs LinearRevGAT+GSMP comparison.

Official LD target:

- validation: `0.7762 +/- 0.0008`
- test: `0.7726 +/- 0.0017`

Completed LinearRevGAT comparison, seeds `1, 2, 3`:

| Seed | LD+LinearRevGAT | LD+LinearRevGAT+GSMP | Delta |
| ---: | ---: | ---: | ---: |
| 1 | `0.770549` | `0.768718` | `-0.001831` |
| 2 | `0.768368` | `0.769129` | `+0.000761` |
| 3 | `0.768615` | `0.767545` | `-0.001070` |

Aggregate:

| Method | Mean test |
| --- | ---: |
| LD+LinearRevGAT | `0.769177` |
| LD+LinearRevGAT+GSMP | `0.768464` |

Takeaway: neutral to slightly worse (`-0.000713`). Runtime was nearly identical:
about `486.3s` vs `487.2s`.

## 260611_2 - GLEM RevGAT / LinearRevGAT

Goal: test GSMP in GLEM without paying for full EM language-model retraining.

Caveat: this is a frozen-output GNN-pretrain ablation, not a full official GLEM
leaderboard reproduction. The official GLEM+RevGAT target is:

- validation: `0.7746 +/- 0.0018`
- test: `0.7694 +/- 0.0025`

All full runs used:

- stage: `GNN-pretrain`
- frozen LM outputs
- no full EM retraining
- epoch budget: 2000
- early stopping patience: 300

Aggregate, seeds `0, 1, 2, 3`:

| Method | Mean val | Val std | Mean test at best val | Test std | Mean runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLEM+RevGAT | `0.768239` | `0.000589` | `0.758780` | `0.002822` | `211.7s` |
| GLEM+LinearRevGAT | `0.767970` | `0.000518` | `0.759506` | `0.002788` | `154.1s` |
| GLEM+LinearRevGAT+GSMP | `0.767081` | `0.000637` | `0.759897` | `0.002268` | `152.7s` |

GSMP minus LinearRevGAT:

- seeds 0-3: `+0.000391` test, `-0.000889` val
- seeds 1-3: `+0.001804` test, `-0.000638` val

Takeaway: mixed but mildly positive on test in this frozen-output GNN-pretrain
setting. It does not establish that GSMP improves full GLEM.

## 260612 / 260612_2 - tunedGNN GCN And P-GSMP

Goal: test GSMP on a clean linear GCN where there is no attention-removal
confound.

Official tunedGNN GCN target:

- validation: `0.7447 +/- 0.0014`
- test: `0.7360 +/- 0.0018`

Selected seed-0 full results:

| Method | GSMP/P-GSMP scope | Best epoch | Val accuracy | Test at best val | Best raw test | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GCN | none | 898 | `0.746938` | `0.731004` | `0.736415` | `1865.6s` |
| GCN+GSMP | all layers, weighted GCN norm | 885 | `0.743246` | `0.727486` | `0.732671` | `2477.6s` |
| GCN+GSMP | first layer only / STAMP-like | 831 | `0.748381` | `0.729399` | `0.738103` | `2811.3s` |
| GCN+P-GSMP | preprocessing only, alpha `0.5`, depth `1` | 828 | `0.740998` | `0.724503` | `0.731210` | `1858.9s` |

Takeaway: the baseline reproduced a healthy validation score. GSMP improved
validation in the first-layer run but did not improve validation-selected test
accuracy over the baseline.

## 260612_3 - SimTeG/TAPE LinearRevGAT With GSMP1 / STAMP

Goal: isolate the first-layer-only variant on the strongest SimTeG/TAPE
LinearRevGAT-style arXiv stack and compare against P-GSMP.

The official SimTeG+TAPE+RevGAT anchor is:

- validation: `0.7846 +/- 0.0004`
- test: `0.7803 +/- 0.0007`

The direct comparator here is not official RevGAT. The fair comparison is among
LinearRevGAT variants.

Four text components, seeds `1, 2, 3`, official weights `2:2:1:1`:

| Method | Val accuracy | Test at best val |
| --- | ---: | ---: |
| baseline LinearRevGAT | `0.786033 +/- 0.000232` | `0.780212 +/- 0.002203` |
| GSMP1 / STAMP | `0.784210 +/- 0.000279` | `0.782030 +/- 0.000375` |
| P-GSMP | `0.782711 +/- 0.001286` | `0.776674 +/- 0.001633` |

Delta versus baseline:

- GSMP1 / STAMP: `+0.001817` test, `-0.001823` val
- P-GSMP: `-0.003539` test, `-0.003322` val

Four text components, uniform weights `1:1:1:1`:

| Method | Val accuracy | Test at best val |
| --- | ---: | ---: |
| baseline LinearRevGAT | `0.784534 +/- 0.000554` | `0.778484 +/- 0.002163` |
| GSMP1 / STAMP | `0.784109 +/- 0.000228` | `0.780480 +/- 0.000935` |
| P-GSMP | `0.782129 +/- 0.000822` | `0.776482 +/- 0.002201` |

Five-component GPT ensemble, seeds `0-9`, weights `2:2:1:1:1`:

| Method | Val accuracy | Test accuracy |
| --- | ---: | ---: |
| baseline + GPT ensemble | `0.78570086 +/- 0.00048306` | `0.78122750 +/- 0.00118150` |
| GSMP1 / STAMP + GPT ensemble | `0.78514380 +/- 0.00078784` | `0.78129951 +/- 0.00128765` |

Delta, GSMP1 minus baseline:

- validation: `-0.00055707`
- test: `+0.00007201`

Per-seed GPT ensemble:

| Seed | Baseline val | Baseline test | GSMP1 val | GSMP1 test | Delta test |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.78539548` | `0.78044565` | `0.78526125` | `0.78165957` | `+0.00121392` |
| 1 | `0.78610020` | `0.78028105` | `0.78459009` | `0.78275004` | `+0.00246898` |
| 2 | `0.78606665` | `0.77908771` | `0.78519413` | `0.78309981` | `+0.00401210` |
| 3 | `0.78599953` | `0.78277061` | `0.78485855` | `0.78133037` | `-0.00144024` |
| 4 | `0.78546260` | `0.78028105` | `0.78593241` | `0.78017818` | `-0.00010287` |
| 5 | `0.78489211` | `0.78186532` | `0.78616732` | `0.78079542` | `-0.00106989` |
| 6 | `0.78589886` | `0.78122750` | `0.78549616` | `0.77943748` | `-0.00179001` |
| 7 | `0.78512702` | `0.78250314` | `0.78371757` | `0.78071312` | `-0.00179001` |
| 8 | `0.78563039` | `0.78155669` | `0.78428806` | `0.78009588` | `-0.00146082` |
| 9 | `0.78643579` | `0.78225624` | `0.78593241` | `0.78293521` | `+0.00067897` |

Takeaway: GSMP1 / STAMP is promising but not uniformly positive. It clearly
helped the 4-text-component official-weight ensemble, and tied the 10-seed
five-component GPT ensemble.

## 260613 - Pokec Temporal Analysis And GCN

Goal: define a chronological Pokec split using registration year, then test
first-layer GSMP on a paper-aligned GCN.

Chronological split:

| Split | Nodes | Fraction | Years | Rule |
| --- | ---: | ---: | --- | --- |
| train | `1,218,296` | `74.62%` | 1999-2010 | year `<= 2010` |
| validation | `297,038` | `18.19%` | 2011 | year `== 2011` |
| test | `117,306` | `7.19%` | 2012 | year `== 2012` |

Graph statistics:

- profile rows: `1,632,803`
- valid registration-year users: `1,632,640`
- raw directed edges: `30,622,564`
- edges with valid endpoint years: `30,622,117`

GCN setup:

- model: tunedGNN-style GCN
- hidden dimension: 256
- layers: 7
- dropout: 0.2
- learning rate: 0.0005
- weight decay: 0
- batch size: 250000 in the completed run
- epochs: 100 in the completed run
- seeds: `1, 2, 3`
- GSMP scope: first GCN layer only

| Method | Seeds | Val accuracy | Test at best val | Best epochs | GPU memory | Runtime |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| GCN | 3 | `0.7885 +/- 0.0010` | `0.7665 +/- 0.0011` | `90, 85, 85` | `5.9650 +/- 0.0004 GB` | `1171.8 +/- 0.7s` |
| GCN+GSMP first layer | 3 | `0.7887 +/- 0.0005` | `0.7675 +/- 0.0007` | `100, 85, 85` | `5.9893 +/- 0.0003 GB` | `1281.0 +/- 1.3s` |

Takeaway: first-layer GSMP improved test by about `+0.0010`, with modest
runtime overhead.

## 260614 / 260615 / 260616 - HGAMLP-HOPE On `ogbn-mag`

Goal: patch official HOPE/HGAMLP propagation to test timestamp-aware
paper-paper propagation without changing the HOPE prediction head.

Implementation:

- Official HOPE code is preserved under each `HOPE/` folder.
- SMP/GSMP are inserted into meta-path propagation preprocessing.
- No labels/classes are used to construct weights.
- Feature propagation caches are reused under `impact_cache*/`.
- GSMP counts source paper years target-side:
  `raw_weight[u, v] = 1 / C_v[time[u]]`.

### Env-Matched Stage0, Feature+Label Scope

Folder: `260615_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0`

Setup:

- methods: `none` vs `gsmp`
- seeds: `1, 2, 3`
- GSMP active in stage 0
- impact applied to feature and label propagation
- stages: 300 each

| Method | Seeds | Best validation mean | Test-at-best-validation mean |
| --- | ---: | ---: | ---: |
| none | 3 | `59.989% +/- 0.029` | `57.998% +/- 0.116` |
| gsmp | 3 | `59.769% +/- 0.090` | `57.743% +/- 0.061` |

Delta: GSMP trailed by `0.220` validation percentage points and `0.255` test
percentage points.

Per-seed test:

| Seed | Baseline test | GSMP test | Delta |
| ---: | ---: | ---: | ---: |
| 1 | `58.130%` | `57.753%` | `-0.377 pp` |
| 2 | `57.953%` | `57.798%` | `-0.155 pp` |
| 3 | `57.910%` | `57.677%` | `-0.234 pp` |

### Priority0 Stage0 Feature-Only GSMP, Current Seeds 0-4 And 9

Folder: `260614_2_ogbn_mag_HGAMLP_HOPE_v2`

Setup:

- priority: `priority0`
- GSMP stages: `0`
- impact scope: feature propagation only
- GSMP direct paper-paper hops: all eligible direct `P-P` stack updates
- completed selected seeds: `0, 1, 2, 3, 4, 9`
- seed `5` was in progress in `priority0_seed5_8_compare_live_progress.tsv`
  and had only reached baseline stage 1 in the inspected file, so seeds 5-8
  are excluded from the aggregate.

Per-seed final-stage test:

| Seed | Baseline test | GSMP test | GSMP - baseline |
| ---: | ---: | ---: | ---: |
| 0 | `57.991%` | `57.815%` | `-0.176 pp` |
| 1 | `58.130%` | `58.192%` | `+0.062 pp` |
| 2 | `57.762%` | `57.755%` | `-0.007 pp` |
| 3 | `58.115%` | `57.941%` | `-0.174 pp` |
| 4 | `58.249%` | `57.734%` | `-0.515 pp` |
| 9 | `57.791%` | `57.865%` | `+0.074 pp` |

Aggregate:

| Method | Seeds | Best validation | Test at best validation |
| --- | ---: | ---: | ---: |
| baseline none | 6 | `60.088% +/- 0.146` | `58.006% +/- 0.196` |
| GSMP priority0 | 6 | `59.988% +/- 0.154` | `57.884% +/- 0.169` |

Delta: GSMP priority0 was `-0.123` test percentage points on the current
completed seed set.

### Priority1 Stage0 All-Hops GSMP

Folder: `260614_2_ogbn_mag_HGAMLP_HOPE_v2`

Setup:

- GSMP stages: `0`
- impact scope: feature and label propagation
- direct paper-paper GSMP hops: all eligible hops
- seeds: `1, 2, 3`

Completed GSMP test values from `priority1_stage0_gsmp_all_hops_live_progress.tsv`:

| Seed | GSMP test |
| ---: | ---: |
| 1 | `57.908%` |
| 2 | `57.870%` |
| 3 | `57.822%` |

The env-matched stage0 summary above is the cleaner matched comparison for
baseline-vs-GSMP, and it was negative for GSMP.

### Priority2 Stage0-3 All-Hops GSMP

Folder: `260616_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3`

Status: incomplete/early. The inspected progress file had:

| Seed | Latest stage | Latest epoch | Best validation | Test at best validation |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 238 | `0.571587` | `0.553280` |
| 2 | 0 | 271 | `0.558224` | `0.541453` |

Takeaway: no reportable full aggregate yet, and the partial values were far
below the stage0 baseline.

## 260617 - RpHGNN On `ogbn-mag`

Goal: prepare a stronger `ogbn-mag` heterogeneous baseline for SMP/GSMP
experiments.

Preserved official settings:

- method: `rphgnn`
- `use_nrl=True`
- `use_label=True`
- `train_strategy=cl`
- `hidden_size=512`
- `num_epochs=500`
- `max_patience=50`
- `use_all_feat=True`
- OGB evaluator: `Evaluator("ogbn-mag")`

Prepared variants:

- `baseline`
- `smp_layerwise`
- `gsmp_first_layer`
- `gsmp_paper_added`, which adds exact source-year-bucketed GSMP to effective
  `P-A-P` and `P-F-P` paper outputs when destination paper representations are
  formed.

Status: no full result has been launched from the reproduction report. The next
step is baseline smoke, then a 10-seed baseline, then SMP/GSMP only after the
baseline is acceptable.

## 260617_2 - PSHGCN On `ogbn-mag`

Goal: prepare PSHGCN with ComplEx embeddings and GSMP first-linear feature
inputs.

Prepared variants:

- `baseline`
- `smp_layerwise`
- `gsmp_first_linear`
- `gsmp_first_linear_stage0`
- smoke versions of all four

Status: no reportable result. The runner exits before GPU training because the
official ComplEx files are incomplete. Present file:

- `complEx/institution.pt`

Missing files:

- `complEx/author.pt`
- `complEx/field_of_study.pt`

## 260618 - SeHGNN-HOPE On `ogbn-mag`

Goal: test GSMP paper-stack feature propagation with SeHGNN-HOPE while
preserving label propagation, HOPE experts, SeHGNN semantic fusion, and MLP
heads.

Baseline target from HGAMLP-HOPE Table 2:

- SeHGNN+HOPE test accuracy: `57.95 +/- 0.14`
- seeds: `1-10`

GSMP scope:

- `paper-stack`
- feature propagation only
- direct `P-P` stack updates such as `P->PP`, `PA->PPA`, `PF->PPF`, and
  `PP->PPP`
- not applied to label propagation

Seed-1 canary results:

| Method | Best validation | Test at best validation | Best epoch | Latest stage/epoch |
| --- | ---: | ---: | ---: | --- |
| baseline | `60.181%` | `58.125%` | 194 | stage 3 / epoch 294 |
| SMP paper-stack | `60.006%` | `57.939%` | 190 | stage 3 / epoch 290 |
| GSMP paper-stack retry | `59.938%` | `57.860%` | 174 | stage 3 / epoch 274 |

There was one failed GSMP canary before retry:

| File | Status |
| --- | --- |
| `full_canary_gsmp_paperstack_seed1_live_progress.tsv` | collapsed near zero accuracy (`best_val=0.00248154`, `test=0.00102530`) |
| `full_canary_gsmp_paperstack_seed1_retry1_live_progress.tsv` | valid retry, shown above |

Short pilot, 30 epochs, stage 0 only:

| Method | Seed | Best validation | Test at best validation |
| --- | ---: | ---: | ---: |
| baseline | 1 | `0.553184` | `0.536088` |
| baseline | 2 | `0.551627` | `0.536660` |
| gsmp | 1 | `0.546171` | `0.528029` |
| gsmp | 2 | `0.544568` | `0.528768` |

Takeaway: the canary and pilot do not support GSMP paper-stack on SeHGNN-HOPE
yet. Baseline seed 1 is already near the expected target and remains ahead.

## Current Scoreboard

Positive or near-positive settings:

| Experiment | Comparator | Delta test |
| --- | --- | ---: |
| 260612_3 four text components, official weights | GSMP1 / STAMP - LinearRevGAT | `+0.001817` |
| 260610_2 official TAPE LinearRevGAT | GSMP - LinearRevGAT | `+0.0010` |
| 260613_2 Pokec GCN | first-layer GSMP - GCN | about `+0.0010` |
| 260611_2 GLEM frozen GNN-pretrain, seeds 0-3 | GSMP - LinearRevGAT | `+0.000391` |
| 260612_3 GPT ensemble, seeds 0-9 | GSMP1 / STAMP - baseline | `+0.000072` |

Negative settings:

| Experiment | Comparator | Delta test |
| --- | --- | ---: |
| 260610 SimTeG/TAPE five-component LinearRevGAT | GSMP - LinearRevGAT | `-0.0025` |
| 260611 LD LinearRevGAT | GSMP - LinearRevGAT | `-0.000713` |
| 260612 tunedGNN GCN all-layer GSMP | GSMP - GCN | `-0.003518` |
| 260612 tunedGNN GCN first-layer GSMP | GSMP first layer - GCN | `-0.001605` |
| 260612_2 tunedGNN P-GSMP | P-GSMP - GCN | `-0.006502` |
| 260615 HGAMLP-HOPE stage0 feature+label | GSMP - none | `-0.00255` |
| 260614_2 HGAMLP-HOPE priority0 feature-only, seeds 0-4 and 9 | GSMP - none | `-0.00123` |
| 260618 SeHGNN-HOPE canary seed1 | GSMP - baseline | `-0.00265` |

## Interpretation

The arXiv evidence suggests that GSMP is most promising when it is used as a
first-layer temporal correction in a strong text-feature ensemble, rather than
as an all-layer replacement for every aggregation step. This aligns with the
current STAMP direction: correct the first neighborhood mixing step and let
later layers operate normally.

The MAG evidence is different. In HGAMLP-HOPE and SeHGNN-HOPE, stage-aware
paper-stack GSMP has consistently trailed matched baselines so far. A plausible
reason is that MAG paper-paper temporal frequency is itself predictive for the
HOPE/HGAMLP/SeHGNN propagation stack, so equalizing source-year groups can
remove useful signal.

The safest paper-facing claim from these experiments is:

> STAMP/first-layer GSMP can improve or tie strong chronological arXiv
> ensembles, but the effect is model- and dataset-dependent. On `ogbn-mag`
> HOPE-style heterogeneous propagation, the current evidence is negative, so
> GSMP should be presented as an ablation rather than a universal improvement.

## Source Files Used

- `260609/README_experiment.md`
- `260609/results/main_48112756/summary.csv`
- `260609/results/leaderboard3_*/summary.csv`
- `260609/results/ensembles/*.json`
- `260610/COMPARISON_WITH_260610.md`
- `260610_2/RESULTS_SUMMARY.md`
- `260611/ld_revgat_gsmp_seed1_3_comparison.md`
- `260611_2/RESULTS_GLEM_GSMP_COMPARISON.md`
- `260612/README_GSMP_TUNEDGCN_OGBN_ARXIV.md`
- `260612/results/tunedgcn_gsmp/*/final_summary.json`
- `260612_2/results/tunedgcn_pgsmp/*/final_summary.json`
- `260612_3_SimTEG_TAPE_linRevGAT/results_ensemble_with_gpt_preds_seeds0_9.md`
- `260612_3_SimTEG_TAPE_linRevGAT/results/ensembles/*.json`
- `260613_pokec/pokec_temporal_outputs/pokec_dataset_report.md`
- `260613_2_pokec_GCN/results/pokec_temporal_summary.md`
- `260614_2_ogbn_mag_HGAMLP_HOPE_v2/results/*.tsv`
- `260615_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0/STAGE0_RESULTS_SUMMARY.md`
- `260616_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3/results/*.tsv`
- `260617_ogbn_mag_RpHGNN/diagnostics/reproduction_report.md`
- `260617_2_ogbn_mag_PSHGCN/README.md`
- `260618_ogbn_mag_SeHGNN_HOPE/results/*.tsv`
