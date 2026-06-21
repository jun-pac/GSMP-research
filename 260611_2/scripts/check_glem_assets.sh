#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLEM_DIR="${GLEM_DIR:-$WORKDIR/upstream/GLEM}"

echo "GLEM_DIR=$GLEM_DIR"
echo "Checking likely GLEM cache locations:"
for path in \
  "$GLEM_DIR/data" \
  "$GLEM_DIR/temp/prt_lm" \
  "$GLEM_DIR/temp/prt_gnn" \
  "$GLEM_DIR/cache/gsmp" \
  "$WORKDIR/cache/gsmp" \
  "$WORKDIR/results/glem_revgat_gsmp"; do
  if [[ -e "$path" ]]; then
    du -sh "$path" || true
  else
    echo "missing $path"
  fi
done

echo
echo "Existing pretrained LM/GNN artifacts, if any:"
find "$GLEM_DIR/temp" -maxdepth 5 \( -name '*.emb' -o -name '*.pred' -o -name '*.result' -o -name '*.ckpt' \) 2>/dev/null | sed -n '1,80p' || true
