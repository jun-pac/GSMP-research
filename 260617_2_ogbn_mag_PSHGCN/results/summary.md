# PSHGCN ogbn-mag Results Summary

Date: 2026-06-17

Experiment folder:
`/users/PAS1289/jyp531/GSMP-research/260617_2_ogbn_mag_PSHGCN/PSHGCN/ogbn-mag`

## Leaderboard-Style 500-Epoch Result

This is the main PSHGCN result as of 2026-06-17.

Command family:

```bash
ARRAY_SPEC=1-3%1 EPOCHS=500 STAGE=4 LAYERS_X=2 \
EVAL_EVERY=1 TEST_EVERY=1 MEM=48G TIME_LIMIT=06:00:00 \
bash scripts/submit_ogbn_mag_pshgcn_smp_gsmp.sh baseline

ARRAY_SPEC=1-3%1 EPOCHS=500 STAGE=4 LAYERS_X=2 \
EVAL_EVERY=1 TEST_EVERY=1 MEM=48G TIME_LIMIT=06:00:00 \
bash scripts/submit_ogbn_mag_pshgcn_smp_gsmp.sh gsmp_first_linear_stage0
```

Metric below is stage 3 `test_at_best_val`. All `+/-` values are sample standard deviation across seeds.

| Variant | Seeds | Best Val Mean | Test @ Best Val Mean | Final/Last Test Mean |
|---|---:|---:|---:|---:|
| baseline | 1,2,3 | `0.59275986 +/- 0.00088975` | `0.57228355 +/- 0.00146656` | `0.57153644 +/- 0.00162846` |
| GSMP first-layer stage0 only | 1,2,3 | `0.59317088 +/- 0.00102847` | `0.57422288 +/- 0.00047349` | `0.57256173 +/- 0.00059196` |

Mean delta, GSMP stage0-only minus baseline:

| Metric | Delta |
|---|---:|
| Best Val | `+0.00041102` |
| Test @ Best Val | `+0.00193933` |
| Final/Last Test | `+0.00102529` |

Per-seed stage 3 results:

| Variant | Seed | End Epoch | Best Epoch | Best Val | Test @ Best Val | Final/Last Test | Job |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 1 | 270 | 171 | `0.59373603` | `0.57376189` | `0.56982761` | `48595861_1` |
| baseline | 2 | 307 | 208 | `0.59254921` | `0.57082906` | `0.57307041` | `48602792_2` |
| baseline | 3 | 267 | 168 | `0.59199433` | `0.57225971` | `0.57171129` | `48602794_3` |
| GSMP first-layer stage0 only | 1 | 203 | 104 | `0.59299619` | `0.57369036` | `0.57290350` | `48595862_1` |
| GSMP first-layer stage0 only | 2 | 192 | 93 | `0.59427550` | `0.57459644` | `0.57290350` | `48602793_2` |
| GSMP first-layer stage0 only | 3 | 247 | 148 | `0.59224094` | `0.57438184` | `0.57187820` | `48602795_3` |

Per-seed deltas, GSMP stage0-only minus baseline:

| Seed | Best Val Delta | Test @ Best Val Delta |
|---:|---:|---:|
| 1 | `-0.00073984` | `-0.00007153` |
| 2 | `+0.00172629` | `+0.00376738` |
| 3 | `+0.00024661` | `+0.00212213` |

Stage-by-stage `test_at_best_val` deltas:

| Seed | Stage | Baseline | GSMP stage0-only | Delta |
|---:|---:|---:|---:|---:|
| 1 | 0 | `0.55189680` | `0.54779561` | `-0.00410119` |
| 1 | 1 | `0.56551181` | `0.56665633` | `+0.00114452` |
| 1 | 2 | `0.56863540` | `0.57314194` | `+0.00450654` |
| 1 | 3 | `0.57376189` | `0.57369036` | `-0.00007153` |
| 2 | 0 | `0.55323208` | `0.55146761` | `-0.00176447` |
| 2 | 1 | `0.56553566` | `0.56730013` | `+0.00176447` |
| 2 | 2 | `0.56980376` | `0.57004220` | `+0.00023844` |
| 2 | 3 | `0.57082906` | `0.57459644` | `+0.00376738` |
| 3 | 0 | `0.55428122` | `0.55213524` | `-0.00214598` |
| 3 | 1 | `0.56679940` | `0.56639405` | `-0.00040535` |
| 3 | 2 | `0.57192589` | `0.56958917` | `-0.00233672` |
| 3 | 3 | `0.57225971` | `0.57438184` | `+0.00212213` |

Runtime/cost audit:

| Job | State | Elapsed | Approx Cost |
|---|---|---:|---:|
| baseline seed 1 | `COMPLETED` | `02:03:03` | `$12.30` |
| GSMP stage0 seed 1 | `COMPLETED` | `01:55:18` | `$11.53` |
| baseline seed 2 | `COMPLETED` | `01:45:40` | `$10.57` |
| GSMP stage0 seed 2 | `COMPLETED` | `02:14:46` | `$13.48` |
| baseline seed 3 | `COMPLETED` | `01:48:03` | `$10.80` |
| GSMP stage0 seed 3 | `COMPLETED` | `01:51:50` | `$11.18` |
| total | | | `$69.87` |

## Baseline Gap Audit

The README leaderboard command is:

```bash
python main.py --extra_emb --stage 4 --layers_x 2
```

The README reports `Val acc: 59.43 +/- 0.15, Test acc: 57.52 +/- 0.11`.

Our full baseline seeds 1-3 are lower: `Val 59.28 +/- 0.09`, `Test 57.23 +/- 0.15`.
I checked the completed seed logs and did not find an obvious launch/config mistake:

- logs show `extra_emb=True`, `stage=4`, `layers_x=2`, `epochs=500`, `eval_every=1`, `test_every=1`, `patience=100`, `runs=1`, and seeds `1`, `2`, `3`;
- all six jobs completed with exit code `0:0`;
- all required ComplEx files were present in `complEx/`;
- for `variant='baseline'`, temporal weights are empty, so feature and label propagation use the original unweighted DGL mean path;
- the main difference from the README command is that these were submitted as separate one-seed Slurm jobs instead of one `runs=10` process.

I then checked the reproduction environment. This is the strongest mismatch found.

| Item | README target | Actual Slurm run |
|---|---|---|
| Python | not specified | `3.10.9` |
| PyTorch | `1.12.1` | `2.6.0+cu124` |
| DGL | `0.9.1` | `2.5.0+cu124` |
| PyG | `2.1.0` | `2.7.0` |
| OGB | `1.3.6` | `1.3.6` |
| torch-sparse | not specified | `0.6.18+pt26cu124` |
| CUDA shown by PyTorch | likely CUDA 11-era stack | `12.4` |
| NVIDIA driver CUDA capability | not specified | `13.0` |
| GPU | not specified | Tesla V100-PCIE-16GB |

Additional audit notes:

- the required ComplEx tensors are present and are not the local TransE files; they are float64 tensors with shapes `author=(1134649,256)`, `field_of_study=(59965,256)`, `institution=(8740,256)`, and are cast to float32 by the loader;
- current ComplEx SHA256 hashes are:
  - `author.pt`: `7e56aabb73d318d986373bc0c7a5f174f4cd796ac48e33fc7ec70f34aed762e7`
  - `institution.pt`: `493c25d2a55650fc7996578436b5ab85923ecf39b6e6856fc01fcd64de09fb15`
- local code is modified from upstream in `main.py`, `processing.py`, and `utils.py`;
- baseline propagation still falls through to the original `fn.mean` path when temporal weights are empty;
- DGL node-count inference is not the issue: inferred node counts match explicit counts exactly for P/A/I/F;
- diagonal correction tensors in `cache/diag` look sane, but this is still a local cache path difference from upstream's `./data/*_diag.pt`;
- our local code clamps tiny negative diagonal-correction roundoff to zero after checking the minimum; observed baseline minima were only about `-5.96e-08`, so this is unlikely to explain a 0.3-point test gap by itself.

So I do not see evidence that the Slurm command was wrong. The most likely reason the baseline is low is that this is not the README environment: PyTorch `2.6`/DGL `2.5` is far from the target PyTorch `1.12.1`/DGL `0.9.1` stack. The second most likely reason is that the ComplEx files may not be byte-identical to the files used for the README result, because the README does not provide checksums.

I would not call this an exact leaderboard reproduction yet, because the 3-seed baseline is meaningfully below the published 10-run mean. The remaining plausible explanations are:

- environment mismatch, especially PyTorch/DGL;
- ComplEx file mismatch;
- seeds `1-3` may be an unlucky subset, and the README number is over seeds `1-10`;
- our repo has cache/logging/GSMP modifications, and although the baseline path appears functionally equivalent, a clean upstream replay would be the strongest check.

Conservative interpretation: compare GSMP stage0-only against the matched local baseline above, not directly against the README baseline. Before reporting a leaderboard-reproduction claim, run either baseline seeds `4-10` or one clean upstream baseline replay.

Main comparison:
- `baseline`: official PSHGCN-style run with ComplEx embeddings.
- `gsmp_first_linear_stage0`: GSMP first-layer propagation applied only to stage 0 features; stages 1-3 use baseline propagated features.

## Earlier 50-Epoch Pilot

Common pilot setting:

```bash
EPOCHS=50 STAGE=4 LAYERS_X=2 EVAL_EVERY=5 TEST_EVERY=5 MEM=48G TIME_LIMIT=01:00:00
```

Metric reported below is stage 3 `test_at_best_val`. All `+/-` values are sample standard deviation across seeds.

### 50-Epoch 3-Seed Result

| Variant | Seeds | Best Val Mean | Test @ Best Val Mean | Notes |
|---|---:|---:|---:|---|
| baseline | 0,1,2 | `0.58462677 +/- 0.00187492` | `0.56506672 +/- 0.00104080` | completed cleanly |
| GSMP first-layer stage0 only | 0,1,2 | `0.58672811 +/- 0.00192403` | `0.56784059 +/- 0.00213992` | completed cleanly |

Mean delta, GSMP stage0-only minus baseline:

| Metric | Delta |
|---|---:|
| Best Val | `+0.00210135` |
| Test @ Best Val | `+0.00277387` |

Interpretation: GSMP first-layer stage0-only is the current best PSHGCN variant. It improves the 3-seed mean test accuracy by about `+0.277` percentage points over baseline.

## Per-Seed Stage 3 Results

| Variant | Seed | Best Val | Test @ Best Val | Best Epoch | Job |
|---|---:|---:|---:|---:|---|
| baseline | 0 | `0.58676922` | `0.56527337` | 34 | `48562640` |
| baseline | 1 | `0.58328581` | `0.56598870` | 44 | `48582334` |
| baseline | 2 | `0.58382527` | `0.56393810` | 49 | `48582334` |
| GSMP first-layer stage0 only | 0 | `0.58737034` | `0.56765779` | 49 | `48582256` |
| GSMP first-layer stage0 only | 1 | `0.58456511` | `0.56579794` | 34 | `48591844` |
| GSMP first-layer stage0 only | 2 | `0.58824889` | `0.57006605` | 49 | `48591844` |

Per-seed deltas, GSMP stage0-only minus baseline:

| Seed | Best Val Delta | Test @ Best Val Delta |
|---:|---:|---:|
| 0 | `+0.00060112` | `+0.00238442` |
| 1 | `+0.00127930` | `-0.00019076` |
| 2 | `+0.00442362` | `+0.00612795` |

## Other Exploratory Runs

These are single-seed checks, so they are not the main result.

| Variant | Seed | Best Val | Test @ Best Val | Best Epoch | Slurm State | Job |
|---|---:|---:|---:|---:|---|---|
| SMP layerwise | 0 | `0.58379445` | `0.56369966` | 34 | `FAILED` after writing results, exit `127:0` | `48576993` |
| GSMP first-layer all stages | 0 | `0.58733951` | `0.56522568` | 44 | `COMPLETED` | `48579900` |
| GSMP first-layer all eligible P-P propagation | 0 | `0.58385610` | `0.56846849` | 44 | `COMPLETED` | `48672624` |

Single-seed takeaway:
- SMP layerwise underperformed baseline in seed 0.
- GSMP first-layer all stages was essentially tied with baseline in seed 0.
- GSMP first-layer all eligible P-P propagation is the best seed-0 50-epoch exploratory result so far, but it still needs a 3-seed pilot before replacing the current 3-seed stage0-only conclusion.
- GSMP first-layer stage0-only remains the current best confirmed 3-seed PSHGCN variant.

## 2026-06-19 All Eligible P-P GSMP Pilot

New variant: `gsmp_first_linear_allprop`.

Intent: apply first-layer GSMP weights to all eligible direct paper-to-paper propagation in PSHGCN stages 0-3. This includes the existing GSMP feature propagation cache and also enables weighted P-P label propagation, with a weighted `PPPP` diagonal correction.

Smoke command:

```bash
CONDA_ENV_NAME= VENV_PATH=/users/PAS1289/jyp531/GSMP-research/.venv PYTORCH_CUDA_ALLOC_CONF= \
ARRAY_SPEC=0-0 SMOKE_STAGE=4 EPOCHS=1 MEM=48G TIME_LIMIT=01:00:00 \
bash scripts/submit_ogbn_mag_pshgcn_smp_gsmp.sh gsmp_first_linear_allprop_smoke
```

Pilot command:

```bash
CONDA_ENV_NAME= VENV_PATH=/users/PAS1289/jyp531/GSMP-research/.venv PYTORCH_CUDA_ALLOC_CONF= \
ARRAY_SPEC=0-0 EPOCHS=50 STAGE=4 LAYERS_X=2 EVAL_EVERY=5 TEST_EVERY=5 MEM=48G TIME_LIMIT=01:00:00 \
bash scripts/submit_ogbn_mag_pshgcn_smp_gsmp.sh gsmp_first_linear_allprop
```

Both jobs used the same working environment as the matched local results above: Python `3.10.9`, PyTorch `2.6.0+cu124`, DGL `2.5.0+cu124`, V100 GPU, and the shared `.venv` at `/users/PAS1289/jyp531/GSMP-research/.venv`.

Stage-by-stage single-seed result for job `48672624`:

| Stage | Best Val | Test @ Best Val | Best Epoch |
|---:|---:|---:|---:|
| 0 | `0.56725597` | `0.54464818` | 39 |
| 1 | `0.57510134` | `0.55757171` | 39 |
| 2 | `0.57983323` | `0.55936002` | 29 |
| 3 | `0.58385610` | `0.56846849` | 44 |

Seed-0 50-epoch stage-3 comparison:

| Variant | Best Val | Test @ Best Val | Best Epoch |
|---|---:|---:|---:|
| baseline | `0.58676922` | `0.56527337` | 34 |
| GSMP first-layer all stages, feature propagation only | `0.58733951` | `0.56522568` | 44 |
| GSMP first-layer stage0 only | `0.58737034` | `0.56765779` | 49 |
| GSMP first-layer all eligible P-P propagation | `0.58385610` | `0.56846849` | 44 |

Runtime and bill:

| Job | State | Elapsed | MaxRSS | Approx Cost |
|---|---|---:|---:|---:|
| all eligible P-P seed 0 pilot | `COMPLETED` | `00:22:19` | `24.46 GB` | `$2.23` |

The implied charge rate from the previous audit is `$0.10/min` (`$6/hour`). A matched full 500-epoch, seeds `1-3`, `EVAL_EVERY=1`, `TEST_EVERY=1` run should cost about `$36` based on prior completed full GSMP runs (`$11.53`, `$13.48`, `$11.18` per seed). A practical budget range is `$35-$40`; the 6-hour time-limit worst case is `$108` for three seeds. For ten seeds, expect about `$120` with a practical range around `$110-$135`; the 6-hour time-limit worst case is `$360`.

Full matched command, not yet launched:

```bash
CONDA_ENV_NAME= VENV_PATH=/users/PAS1289/jyp531/GSMP-research/.venv PYTORCH_CUDA_ALLOC_CONF= \
ARRAY_SPEC=1-3%1 EPOCHS=500 STAGE=4 LAYERS_X=2 EVAL_EVERY=1 TEST_EVERY=1 MEM=48G TIME_LIMIT=06:00:00 \
bash scripts/submit_ogbn_mag_pshgcn_smp_gsmp.sh gsmp_first_linear_allprop
```

## Runtime And Memory

| Run | Jobs | State | Elapsed | MaxRSS |
|---|---|---|---:|---:|
| baseline seed 0 | `48562640_0` | `COMPLETED` | `00:23:20` | `33.52 GB` |
| baseline seed 1 | `48582334_1` | `COMPLETED` | `00:23:56` | `33.64 GB` |
| baseline seed 2 | `48582334_2` | `COMPLETED` | `00:24:56` | `33.67 GB` |
| GSMP stage0 seed 0 | `48582256_0` | `COMPLETED` | `00:23:26` | `29.48 GB` |
| GSMP stage0 seed 1 | `48591844_1` | `COMPLETED` | `00:24:04` | `45.04 GB` |
| GSMP stage0 seed 2 | `48591844_2` | `COMPLETED` | `00:23:53` | `45.03 GB` |

## Next Recommendation

For the method comparison, the current matched result is enough to say that `gsmp_first_linear_stage0` improves over the local PSHGCN baseline on seeds `1,2,3`.

For a leaderboard-reproduction claim, the baseline gap should be checked before spending on GSMP seeds `4-10`. The cheapest clean check is to run baseline seeds `4-10` only, one at a time:

```bash
cd /users/PAS1289/jyp531/GSMP-research/260617_2_ogbn_mag_PSHGCN/PSHGCN/ogbn-mag

ARRAY_SPEC=4-10%1 EPOCHS=500 STAGE=4 LAYERS_X=2 \
EVAL_EVERY=1 TEST_EVERY=1 MEM=48G TIME_LIMIT=06:00:00 \
bash scripts/submit_ogbn_mag_pshgcn_smp_gsmp.sh baseline
```

Estimated actual cost for those seven baseline jobs is about `$75-$85` based on completed seed runtimes; worst-case with the 6-hour cap is `$252`.
