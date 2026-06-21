#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESEARCH_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
HOPE_DIR="${PROJECT_DIR}/HOPE"

DATA_DIR="${DATA_DIR:-${RESEARCH_DIR}/data/ogbn_mag}"
MAG_P="${MAG_P:-${RESEARCH_DIR}/HGAMLP_MAG/mag.p}"
DEFAULT_SPARSE_TOOLS_DIR="${RESEARCH_DIR}/260616_ogbn_mag_HGAMLP_HOPE_env_matched_GSMP_stage_0to3/sparse_tools"
if [[ ! -d "${DEFAULT_SPARSE_TOOLS_DIR}" ]]; then
    DEFAULT_SPARSE_TOOLS_DIR="${RESEARCH_DIR}/sparse_tools"
fi
SPARSE_TOOLS_DIR="${SPARSE_TOOLS_DIR:-${DEFAULT_SPARSE_TOOLS_DIR}}"

mkdir -p "${HOPE_DIR}/dataset/ogbn_mag"

for name in processed raw split mapping RELEASE_v2.txt; do
    if [[ -e "${DATA_DIR}/${name}" ]]; then
        ln -sfn "${DATA_DIR}/${name}" "${HOPE_DIR}/dataset/ogbn_mag/${name}"
    fi
done

if [[ -f "${MAG_P}" ]]; then
    ln -sfn "${MAG_P}" "${HOPE_DIR}/dataset/ogbn_mag/mag.p"
else
    echo "WARNING: mag.p not found at ${MAG_P}; set MAG_P before running full jobs." >&2
fi

if [[ -d "${SPARSE_TOOLS_DIR}" ]]; then
    ln -sfn "${SPARSE_TOOLS_DIR}" "${PROJECT_DIR}/sparse_tools"
else
    echo "WARNING: sparse_tools not found at ${SPARSE_TOOLS_DIR}; set SPARSE_TOOLS_DIR if needed." >&2
fi

for name in ogbn-mag_PAP_diag.pt ogbn-mag_PFP_diag.pt ogbn-mag_PPP_diag.pt; do
    if [[ -f "${RESEARCH_DIR}/HGAMLP_MAG/${name}" ]]; then
        ln -sfn "${RESEARCH_DIR}/HGAMLP_MAG/${name}" "${HOPE_DIR}/dataset/ogbn_mag/${name}"
    elif [[ -f "${PROJECT_DIR}/HOPE/${name}" ]]; then
        ln -sfn "${PROJECT_DIR}/HOPE/${name}" "${HOPE_DIR}/dataset/ogbn_mag/${name}"
    fi
done

echo "Data links under ${HOPE_DIR}/dataset/ogbn_mag:"
ls -lh "${HOPE_DIR}/dataset/ogbn_mag" || true
