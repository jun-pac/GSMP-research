#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESEARCH_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
HOPE_DIR="${PROJECT_DIR}/HOPE"

DATA_DIR="${DATA_DIR:-${RESEARCH_DIR}/data/ogbn_mag}"
MAG_P="${MAG_P:-${RESEARCH_DIR}/HGAMLP_MAG/mag.p}"
SPARSE_TOOLS_DIR="${SPARSE_TOOLS_DIR:-${RESEARCH_DIR}/sparse_tools}"

mkdir -p "${HOPE_DIR}/dataset/ogbn_mag"

ln -sfn "${DATA_DIR}/processed" "${HOPE_DIR}/dataset/ogbn_mag/processed"
ln -sfn "${DATA_DIR}/raw" "${HOPE_DIR}/dataset/ogbn_mag/raw"
ln -sfn "${DATA_DIR}/split" "${HOPE_DIR}/dataset/ogbn_mag/split"
ln -sfn "${DATA_DIR}/mapping" "${HOPE_DIR}/dataset/ogbn_mag/mapping"
ln -sfn "${DATA_DIR}/RELEASE_v2.txt" "${HOPE_DIR}/dataset/ogbn_mag/RELEASE_v2.txt"
ln -sfn "${MAG_P}" "${HOPE_DIR}/dataset/ogbn_mag/mag.p"
ln -sfn "${SPARSE_TOOLS_DIR}" "${PROJECT_DIR}/sparse_tools"

for name in ogbn-mag_PAP_diag.pt ogbn-mag_PFP_diag.pt ogbn-mag_PPP_diag.pt; do
    if [[ -f "${HOPE_DIR}/${name}" ]]; then
        ln -sfn "../../${name}" "${HOPE_DIR}/dataset/ogbn_mag/${name}"
    elif [[ -f "${RESEARCH_DIR}/HGAMLP_MAG/${name}" ]]; then
        ln -sfn "${RESEARCH_DIR}/HGAMLP_MAG/${name}" "${HOPE_DIR}/dataset/ogbn_mag/${name}"
    fi
done

echo "Prepared data links:"
ls -lh "${HOPE_DIR}/dataset/ogbn_mag/mag.p"
ls -ld "${HOPE_DIR}/dataset/ogbn_mag/processed" "${PROJECT_DIR}/sparse_tools"
