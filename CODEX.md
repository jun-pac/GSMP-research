# Codex Experiment Policy

## Slurm Seed Scheduling

- Long-running experiments must use one seed per Slurm task or one seed per independent `sbatch` job.
- Do not submit one long job that runs multiple seeds sequentially inside a single Python process.
- Do not use array throttles such as `%1` for multi-seed production runs unless the user explicitly asks for serial execution.
- Set walltime close to the per-seed runtime, with a small buffer. For a roughly 6 hour seed, prefer an 8-12 hour request over a 30 hour serial seed bundle.
- If a run changes training behavior, encode the change in the experiment directory and run names. For example: `no_stage3_earlystop`.

## Current GSMP Stage-3 Policy

- Baseline HGAMLP-HOPE runs keep their original early-stopping behavior.
- GSMP HGAMLP-HOPE no-stage3-earlystop runs disable early stopping only in stage 3.
- GSMP no-stage3-earlystop runs must keep per-stage best checkpoints and raw predictions so stage-2 artifacts are reusable.
