#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/ld_revgat_gsmp")
    parser.add_argument("--min-val", type=float, default=0.75)
    args = parser.parse_args()

    root = Path(args.results_root)
    candidates = []
    for path in root.glob("*/final_summary.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if payload.get("method") == "LD+RevGAT":
            val = payload.get("val_mean")
            if val is not None:
                candidates.append((float(val), path, payload))

    if not candidates:
        raise SystemExit(
            "No LD+RevGAT baseline final_summary.json found. "
            "Run the baseline reproduction before full GSMP."
        )

    best_val, path, payload = max(candidates, key=lambda item: item[0])
    print(f"Best LD+RevGAT baseline gate candidate: val={best_val:.6f} path={path}")
    if best_val < args.min_val:
        raise SystemExit(
            f"Baseline val_mean={best_val:.6f} is below gate {args.min_val:.6f}. "
            "Debug baseline reproduction before spending full GSMP GPU hours. "
            "Set SKIP_BASELINE_GATE=1 only if you intentionally override this."
        )

    print("Baseline gate passed.")


if __name__ == "__main__":
    main()
