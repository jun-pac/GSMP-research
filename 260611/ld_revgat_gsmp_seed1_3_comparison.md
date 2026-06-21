# LD + LinearRevGAT vs LD + LinearRevGAT + GSMP (Seeds 1–3)

## Summary

We compared the two variants under the 260611 experiment directory for seeds 1, 2, and 3.

Overall, adding GSMP does not show a consistent improvement on this set of runs. It is slightly better for seed 2, but slightly worse for seeds 1 and 3.

## Results

| Seed | LD + LinearRevGAT | LD + LinearRevGAT + GSMP | Difference |
|---|---:|---:|---:|
| 1 | 0.770549 | 0.768718 | -0.001831 |
| 2 | 0.768368 | 0.769129 | +0.000761 |
| 3 | 0.768615 | 0.767545 | -0.001070 |

## Averages across seeds 1–3

- LD + LinearRevGAT average test score: 0.769177
- LD + LinearRevGAT + GSMP average test score: 0.768464
- Net difference: -0.000713

## Runtime note

Runtime is very similar between the two variants:

- LD + LinearRevGAT: about 486.3s
- LD + LinearRevGAT + GSMP: about 487.2s

## Takeaway

For seeds 1–3, GSMP appears roughly neutral to slightly worse than the non-GSMP baseline in this experiment batch, with no clear advantage in the reported results.
