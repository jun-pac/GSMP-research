#!/bin/bash
set -euo pipefail

OLD_JOB_ID="${OLD_JOB_ID:-48238494}"
THRESHOLD="${THRESHOLD:-0.567}"
OLD_PROJECT="${OLD_PROJECT:-/users/PAS1289/jyp531/GSMP-research/260614_ogbn_mag_HGAMLP_HOPE}"
V2_PROJECT="${V2_PROJECT:-/users/PAS1289/jyp531/GSMP-research/260614_2_ogbn_mag_HGAMLP_HOPE_v2}"
OLD_LOG="${OLD_LOG:-${OLD_PROJECT}/logs/${OLD_JOB_ID}/gsmp_seed1.log}"
CHECK_SECONDS="${CHECK_SECONDS:-60}"

echo "[$(date)] watching ${OLD_LOG}"
echo "[$(date)] threshold=${THRESHOLD}; old_job=${OLD_JOB_ID}"

while [[ ! -f "${OLD_LOG}" ]]; do
    echo "[$(date)] waiting for log file to appear..."
    sleep "${CHECK_SECONDS}"
done

while ! grep -q '^finished method=gsmp seed=1' "${OLD_LOG}"; do
    latest="$(awk -F'\t' '$3=="gsmp" && $4=="1"{line=$0} END{print line}' "${OLD_PROJECT}/results/live_progress.tsv" 2>/dev/null || true)"
    if [[ -n "${latest}" ]]; then
        echo "[$(date)] latest: ${latest}"
    else
        echo "[$(date)] still waiting for gsmp seed=1 to finish..."
    fi
    sleep "${CHECK_SECONDS}"
done

final_test="$(
    awk '/^Test:/ {gsub(/\[|\]/, "", $2); test=$2} END{if(test == "") exit 1; print test}' "${OLD_LOG}"
)"

echo "[$(date)] final test=${final_test}; threshold=${THRESHOLD}"

if awk -v acc="${final_test}" -v threshold="${THRESHOLD}" 'BEGIN{exit !(acc <= threshold)}'; then
    echo "[$(date)] final test <= threshold; canceling remaining old array tasks..."
    scancel "${OLD_JOB_ID}" || true
    echo "[$(date)] submitting priority 1 v2 experiment..."
    cd "${V2_PROJECT}"
    sbatch slurm/run_priority1_stage0_gsmp_all_hops.sbatch
else
    echo "[$(date)] final test > threshold; leaving old array running and not submitting priority 1."
fi
