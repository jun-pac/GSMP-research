import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fail fast if a baseline run is too far below target.")
    parser.add_argument("summary", type=Path, help="Path to final_summary.json")
    parser.add_argument("--min-val", type=float, default=0.735, help="minimum acceptable validation accuracy")
    parser.add_argument("--min-test", type=float, default=0.725, help="minimum acceptable test_at_best_val")
    args = parser.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)

    val = float(summary["val_mean"])
    test = float(summary["test_at_best_val_mean"])
    print(f"[BASELINE_GATE] val_mean={val:.6f} test_at_best_val_mean={test:.6f}")

    if val < args.min_val or test < args.min_test:
        print(
            "[BASELINE_GATE] baseline is below the configured floor; "
            "debug this before spending GPU time on GSMP.",
            file=sys.stderr,
        )
        return 1
    print("[BASELINE_GATE] baseline passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
