#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SEEDS="${SEEDS:-1-3}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-1}"
SLURM_TIME="${SLURM_TIME:-01:30:00}"
SLURM_MEM="${SLURM_MEM:-16G}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-PAS1289}"
SLURM_PARTITION="${SLURM_PARTITION:-gpu}"

export_vars="ALL,FULL=1,SAVE_MODEL=false,LD_REQUIRE_HIDDEN_STATE=1,LD_DISABLE_TQDM=1,LD_GSMP_ROOT=${ROOT}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. This launcher is intended for Slurm submission." >&2
  exit 2
fi

echo "Submitting LD+LinearRevGAT seeds ${SEEDS} with array concurrency ${ARRAY_CONCURRENCY}"
linear_jid="$(
  sbatch --parsable \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION}" \
    --array="${SEEDS}%${ARRAY_CONCURRENCY}" \
    --time="${SLURM_TIME}" \
    --mem="${SLURM_MEM}" \
    --export="${export_vars}" \
    slurm/run_ld_linear_revgat_ablation.sbatch
)"

echo "Submitting LD+LinearRevGAT+GSMP seeds ${SEEDS} after ${linear_jid}"
gsmp_jid="$(
  sbatch --parsable \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION}" \
    --array="${SEEDS}%${ARRAY_CONCURRENCY}" \
    --time="${SLURM_TIME}" \
    --mem="${SLURM_MEM}" \
    --dependency="afterok:${linear_jid}" \
    --export="${export_vars}" \
    slurm/run_ld_linear_revgat_gsmp.sbatch
)"

cat <<EOF
Submitted:
  LD+LinearRevGAT:      ${linear_jid}
  LD+LinearRevGAT+GSMP: ${gsmp_jid}

Monitor:
  watch -n 30 squeue -u \$USER

Summarize after completion:
  python tools/compare_results.py --results-root results/ld_revgat_gsmp
EOF
