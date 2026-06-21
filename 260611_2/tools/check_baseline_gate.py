#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/glem_revgat_gsmp")
    parser.add_argument("--min-val", type=float, default=0.7700)
    parser.add_argument("--method", default="GLEM+RevGAT")
    args = parser.parse_args()

    root = Path(args.results_root)
    summaries = sorted(root.glob("*/final_summary.json"))
    candidates = []
    for path in summaries:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if data.get("method") == args.method:
            candidates.append((float(data.get("best_val", -1)), path, data))

    if not candidates:
        raise SystemExit(
            f"No {args.method} final_summary.json found under {root}. "
            "Run the baseline reproduction first or set SKIP_BASELINE_GATE=1."
        )

    best_val, path, data = max(candidates, key=lambda x: x[0])
    if best_val < args.min_val:
        raise SystemExit(
            f"Best {args.method} val_acc={best_val:.4f} is below gate {args.min_val:.4f} "
            f"({path}). Debug baseline before full GSMP, or set SKIP_BASELINE_GATE=1."
        )

    print(
        f"[BASELINE_GATE] passed method={args.method} best_val={best_val:.4f} "
        f"test_at_best_val={float(data.get('test_at_best_val', -1)):.4f} path={path}"
    )


if __name__ == "__main__":
    main()
