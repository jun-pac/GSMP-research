# Results And GSMP Explanation

This note explains the completed `260610` results and how they differ from the
newer `../260610_2` experiment. The short version is:

- `260610` did use `LinearRevGAT`, despite several file/job names containing
  `revgat`.
- In `260610`, GSMP slightly hurt the final ensemble.
- The likely reason is not simply the normalization name. The `active_years`
  normalization in `260610` and the final `scale_preserve` weights in
  `260610_2` are algebraically equivalent for LinearRevGAT message passing.
- The more important difference is the feature stack and experiment harness:
  `260610` used SimTeG cached embeddings plus GPT-preds over five components,
  while `260610_2` used official TAPE `.emb` features.

## Did `260610` Use LinearRevGAT?

Yes. The Slurm/job names contain `revgat`, but the executed full run used the
linearized trainer:

- `scripts/run_linear_component.sh` calls `linear_revgat_gsmp.py`.
- `scripts/run_linear_component.sh` passes `--model_variant linear`
  or `--model_variant gsmp`.
- `linear_revgat_gsmp.py` defines `LinearRevGAT`.
- `linear_revgat_gsmp.py` instantiates `LinearRevGAT` for both
  variants.
- The saved summaries report `model=LinearRevGAT` or
  `model=LinearRevGATGSMP`.

So the meaningful comparison in `260610` is:

```text
LinearRevGAT vs LinearRevGAT+GSMP
```

not official attention RevGAT vs GSMP.

## `260610` Main Result

The full `260610` run used:

- Dataset: `ogbn-arxiv`
- Seeds: `1 2 3`
- Components: 5
  - `arxiv_e5`
  - `arxiv_roberta`
  - `arxiv_tape_e5`
  - `arxiv_tape_roberta`
  - `arxiv_gpt_preds`
- Ensemble weights: `2 2 1 1 1`
- Max epochs: `200`
- Early stopping: min epoch `80`, patience `40`

Final ensemble:

| Method | Val Acc | Test Acc |
| --- | ---: | ---: |
| LinearRevGAT | `0.7813 +/- 0.0006` | `0.7789 +/- 0.0013` |
| LinearRevGAT+GSMP | `0.7810 +/- 0.0012` | `0.7764 +/- 0.0008` |

Delta:

```text
Val:  -0.0002
Test: -0.0025
```

So in `260610`, GSMP slightly hurt the final ensemble.

## `260610` Component Breakdown

| Component | Linear Test | GSMP Test | GSMP Delta |
| --- | ---: | ---: | ---: |
| `arxiv_e5` | `0.7687` | `0.7670` | `-0.0017` |
| `arxiv_roberta` | `0.7686` | `0.7637` | `-0.0048` |
| `arxiv_tape_e5` | `0.7644` | `0.7626` | `-0.0017` |
| `arxiv_tape_roberta` | `0.7627` | `0.7650` | `+0.0023` |
| `arxiv_gpt_preds` | `0.7622` | `0.7570` | `-0.0052` |

Only `arxiv_tape_roberta` improved with GSMP. The ensemble dropped because
four of five components got worse, including the high-weight `arxiv_e5` and
`arxiv_roberta` components.

## `260610` Resource Use

Full Slurm job:

```text
Job ID: 48147351
State: COMPLETED
Elapsed: 45m32s
GPU: 1 V100
Memory allocated: 32G
Max CPU RAM: ~6.0 GB
GPU-hours: ~0.759
Billing-hours / credits: ~4.55
Memory allocation-hours: ~24.3 GB-hours
```

Component training time from logs:

```text
Linear components total: ~18.7 min
GSMP components total: ~21.7 min
Combined training: ~40.3 min
Slurm wall time: 45.5 min
```

## Normalization: `active_years` vs `scale_preserve`

At first glance the old and new folders look different because:

- `260610` uses `gsmp_norm=active_years`.
- `260610_2` uses `gsmp_norm=scale_preserve`.

However, for the final LinearRevGAT message weights, these are equivalent.

For a target node `v`, define:

```text
N_v(t) = incoming neighbors of v whose source year is t
c_v(t) = |N_v(t)|
A_v = number of active source-year groups for v
d_v = in-degree of v
```

The `260610` `active_years` weight is:

```text
w(u -> v) = 1 / (A_v * c_v(year(u)))
```

The `260610_2` `scale_preserve` path first computes:

```text
mean_weight = 1 / d_v
base = 1 / c_v(year(u))
mean_base_for_v = A_v / d_v
scale = base / mean_base_for_v
      = d_v / (A_v * c_v(year(u)))
```

Then it multiplies by ordinary mean aggregation:

```text
final_weight = mean_weight * scale
             = (1 / d_v) * d_v / (A_v * c_v(year(u)))
             = 1 / (A_v * c_v(year(u)))
```

Therefore:

```text
active_years == scale_preserve final LinearRevGAT message weight
```

The normalization name alone does not explain the result difference between the
folders.

## Why GSMP Hurts In `260610`

GSMP imposes a fixed temporal balancing prior. For each receiver node, it gives
each active source-year group equal total message mass, regardless of how many
neighbors came from that year.

That prior is helpful if temporal concentration is mostly nuisance bias. But in
OGBN-Arxiv citation neighborhoods, temporal concentration can be predictive:

- papers often cite temporally nearby work;
- citation bursts may reflect active topic communities;
- dominant source-year groups can encode field, topic, and publication-cohort
  information;
- SimTeG/TAPE cached embeddings are already strong semantic features, so the
  graph layer may mainly refine or smooth strong node representations.

In that setting, mean aggregation preserves empirical citation density:

```text
many neighbors from a year => more total message mass from that year
```

GSMP removes that density signal:

```text
each active year group => equal total message mass
```

So GSMP can downweight useful majority/cohort evidence and upweight sparse
years that may be noisier or less relevant. That is a plausible explanation for
the `260610` component pattern: GSMP helps one component but hurts four,
especially `arxiv_roberta` and `arxiv_gpt_preds`.

Paper-level phrasing:

> In the `260610` SimTeG/TAPE cached-feature setting, GSMP's equal-temporal-group
> prior appears mismatched to the citation-neighborhood signal. The temporal
> frequency distribution itself seems to carry useful semantic or class
> information, so flattening source-year group mass slightly degrades ensemble
> performance.

This should be presented as a setting-specific empirical finding, not as a
claim that GSMP universally hurts.

## Relation To `260610_2`

The newer `260610_2` result is different:

| Method | Val Acc | Test Acc |
| --- | ---: | ---: |
| TAPE+LinearRevGAT | `0.7764 +/- 0.0020` | `0.7668 +/- 0.0035` |
| TAPE+LinearRevGAT+GSMP | `0.7767 +/- 0.0004` | `0.7678 +/- 0.0026` |

Delta:

```text
Val:  +0.0003
Test: +0.0010
```

This is a small positive result, but it is not directly comparable to `260610`
because the feature sources and harnesses differ. The safest statement is:

> GSMP is sensitive to the feature stack. It slightly hurts the older
> SimTeG/TAPE cached-feature ensemble in `260610`, but slightly improves the
> official TAPE-feature LinearRevGAT run in `260610_2`.
