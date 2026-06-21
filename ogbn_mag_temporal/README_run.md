# Running The HH ogbn-mag Ablation

## Overview

`run_ogbn_mag_hh_ablation.slurm` is the leaderboard-correct HH ablation runner.
It calls `../HGAMLP_MAG/main.py` with the same HGAMLP-MAG settings used by the
ogbn-mag leaderboard reproduction:

- LINE embeddings from `../HGAMLP_MAG/mag.p`
- heterogeneous metapath feature propagation
- propagated label features
- multi-stage self-training with `STAGES="400 400 400 500"`

Method mapping:

- `hh`: HGAMLP-HOPE-style baseline
- `hh_smp`: HH + SMP temporal propagation  
- `hh_ump`: HH + UMP bucket-balanced propagation
- `hh_gsmp`: HH + GSMP temporal ratio propagation

The local `train_hh_mag.py` runner is only a paper-citation PyG prototype. It is
useful for smoke tests, but it is not a leaderboard baseline and should not be
expected to reach `60%` validation / `58%` test accuracy.

### Resource Notes

The leaderboard-correct job requests `16` CPUs, `128G` memory, one GPU, and
`48:00:00` walltime. Those resources are intentional: the full HGAMLP-MAG path
uses non-paper embeddings, DGL metapaths, label propagation, and staged training.

## Quick Start

### Submit the Job

```bash
cd /users/PAS1289/jyp531/GSMP-research/ogbn_mag_temporal

# Default: sequential run of all 4 methods
sbatch run_ogbn_mag_hh_ablation.slurm

# Run only the HH leaderboard baseline
METHOD=hh SEEDS=0 sbatch run_ogbn_mag_hh_ablation.slurm

# Optional: override leaderboard stages
STAGES="400 400 400 500" SEEDS="0 1 2" sbatch run_ogbn_mag_hh_ablation.slurm

# Optional: get job ID
JID=$(sbatch run_ogbn_mag_hh_ablation.slurm | awk '{print $NF}')
echo "Job ID: $JID"
```

### Monitor in Real Time

```bash
# See if your job is queued/running
squeue -u $USER

# Watch the global SLURM log
tail -f logs/slurm_<JOBID>.out

# Watch individual method logs (clean format, suitable for tail -f)
tail -f logs/hh_<JOBID>.out
tail -f logs/hh_smp_<JOBID>.out
tail -f logs/hh_ump_<JOBID>.out
tail -f logs/hh_gsmp_<JOBID>.out

# Extract final best lines
grep "Best Epoch" logs/hh_<JOBID>.out | tail
```

### Cancel the Job (if needed)

```bash
# List active jobs
squeue -u $USER

# Cancel specific job
scancel <JOBID>

# Cancel all your jobs
scancel -u $USER
```

## Output Structure

After running, you will have:

```
logs/
├── slurm_<JOBID>.out              # Overall SLURM output
├── hh_<JOBID>.out                 # Method-specific HGAMLP_MAG log
├── hh_smp_<JOBID>.out
├── hh_ump_<JOBID>.out
└── hh_gsmp_<JOBID>.out

../HGAMLP_MAG/output/ogbn-mag/
└── <impact-method>/seed_<N>/      # HGAMLP checkpoints and raw predictions
```

## Legacy Local Prototype

The old local prototype can still be run directly for smoke tests, but it is not
the leaderboard implementation:

```bash
python -u train_hh_mag.py \
  --method hh_gsmp \
  --dataset ogbn-mag \
  --root ./data \
  --epochs 50 \
  --eval-every 5 \
  --log-every 5 \
  --seed 0 \
  --device cuda \
  --output-dir . \
  --hidden-dim 512 \
  --dropout 0.5 \
  --lr 0.001 \
  --weight-decay 0.0 \
  --num-hops 6 \
  --bucket-style yearly \
  --use-precomputed \
  2>&1 | tee logs/hh_gsmp_seed0_local.out
```

Legacy key arguments:
- `--method`: One of `hh`, `hh_smp`, `hh_ump`, `hh_gsmp`
- `--epochs`: Number of training epochs (default: 100)
- `--eval-every`: Evaluation frequency (default: 10)
- `--batch-size`: Paper node mini-batch size (0 = full batch, default: 16384)
- `--seed`: Random seed for reproducibility
- `--bucket-style`: Temporal bucketing for SMP/UMP/GSMP (`coarse` or `yearly`)
- `--use-precomputed`: Load cached propagation features if available
- `--force-recompute`: Recompute propagation features even if cached

## Customization

### Edit SLURM Resources

Open `run_ogbn_mag_hh_ablation.slurm` and modify the header:

```bash
#SBATCH --cpus-per-task=16     # Default: 16
#SBATCH --mem=128G             # Default: 128GB
#SBATCH --time=48:00:00        # Default: 48 hours
#SBATCH --partition=gpu        # Change if your cluster uses different names
#SBATCH --gres=gpu:1           # GPU type/count (adjust as needed)
```

### Override via Environment Variables

```bash
# Run only HH baseline with one seed
METHOD=hh SEEDS=0 sbatch run_ogbn_mag_hh_ablation.slurm

# Run custom leaderboard stages
STAGES="300 300 300 400" sbatch run_ogbn_mag_hh_ablation.slurm
```

### Run as Array Job (Optional)

To run all 4 methods **in parallel** on separate GPUs (not recommended unless you have multiple GPUs):

1. Uncomment the array line in `run_ogbn_mag_hh_ablation.slurm`:
   ```bash
   #SBATCH --array=0-3    # Uncomment this
   ```

2. Submit as usual:
   ```bash
   sbatch run_ogbn_mag_hh_ablation.slurm
   ```

**Warning:** This submits 4 independent jobs; only use if you have 4+ available GPUs.

## Understanding the Output Format

Each method log is the upstream HGAMLP_MAG output. The final lines to care about
look like this:

```
HH | Seed <N> | Stage 3 Best Epoch 191, Val 60.1227, Test 58.2250
```

This format is designed for easy parsing with `grep` or `tail -f`:

```bash
# Extract final best lines for the HH baseline
grep "Best Epoch" logs/hh_<JOBID>.out | tail
```

## Troubleshooting

### Job stuck in queue

```bash
squeue -u $USER      # Check priority and state
sinfo               # Check partition availability
```

### Out-of-memory (OOM) errors

Increase `--mem` in the SLURM script:
```bash
#SBATCH --mem=160G    # Increase from 128G
```

Or reduce batch size:
```bash
BATCH_SIZE=8192 sbatch run_ogbn_mag_hh_ablation.slurm
```

### GPU memory errors

Reduce batch size:
```bash
BATCH_SIZE=8192 sbatch run_ogbn_mag_hh_ablation.slurm
```

### Missing dataset

```bash
# Download ogbn-mag manually
python -c "from ogb.nodeproppred import DglNodePropPredDataset; DglNodePropPredDataset(name='ogbn-mag')"
```

## Performance Tips

1. **Run HH alone first**: Use `METHOD=hh SEEDS=0` to verify the baseline before launching all four methods.

2. **Keep leaderboard stages for final numbers**: The verified HH run used `STAGES="400 400 400 500"`.

3. **Use shorter stages only for smoke tests**: For example, `STAGES="5"` checks plumbing but will not be meaningful for accuracy.

4. **Batch processing**: Increase `BATCH_SIZE` if GPU memory is available; decrease it for GPU OOM errors.

## Method Definitions

The leaderboard runner implements these transforms inside `../HGAMLP_MAG/utils.py`.

### HH (HGAMLP-HOPE Baseline)

Uses ordinary paper-paper propagation with no temporal adjustments:

```
X_paper^(k) = A_pp^k X_paper
```

### HH+SMP (Symmetrized Message Passing)

Weights edges based on temporal proximity:

```
A_pp^SMP(u,v) = w_smp(t_u, t_v) * A_pp(u,v)
```

Normalized row-wise for stable propagation.

### HH+UMP (Unsymmetrized Message Passing)

Bucket-balanced variant to ablate whether gains come from balanced propagation:

```
A_pp^UMP(u,v) = 1 / count_edges_in_same_bucket(t_u, t_v) * A_pp(u,v)
```

### HH+GSMP (Generalized Source Message Passing)

Uses transition probabilities estimated from train→validation distribution:

```
A_pp^GSMP(u,v) ∝ r(t_u, t_v) * A_pp(u,v)

where r(t_u, t_v) = P_val(t_u, t_v) / P_train(t_u, t_v)
```

## Citation

If you use this pipeline in your research, please cite the relevant papers:

- HGAMLP-HOPE: [ref]
- ogbn-mag: Open Graph Benchmark
- Temporal GNN methods: [refs]
```

To enable array mode, uncomment this line in `run_ogbn_mag_hh_ablation.slurm`:

```bash
##SBATCH --array=0-3
```

and change it to:

```bash
#SBATCH --array=0-3
```

Array mapping is `0 -> hh`, `1 -> hh_smp`, `2 -> hh_ump`, `3 -> hh_gsmp`.
