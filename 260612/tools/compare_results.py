import argparse
import json
from pathlib import Path


def fmt(mean, std):
    return f"{100 * mean:.2f}+/-{100 * std:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Print a compact comparison table from final_summary.json files.")
    parser.add_argument("summaries", nargs="+", type=Path)
    args = parser.parse_args()

    rows = []
    for path in args.summaries:
        with open(path) as f:
            summary = json.load(f)
        label = summary["method"]
        if summary.get("use_gsmp"):
            label = (
                f"{label}("
                f"{summary.get('gsmp_norm')}, "
                f"{summary.get('gcn_gsmp_mode')}, "
                f"{summary.get('gsmp_apply', 'all_layers')})"
            )
        rows.append(
            {
                "method": label,
                "val": fmt(summary["val_mean"], summary["val_std"]),
                "test": fmt(summary["test_at_best_val_mean"], summary["test_at_best_val_std"]),
                "raw": fmt(summary["best_raw_test_mean"], summary["best_raw_test_std"]),
                "epoch": f"{summary['best_epoch_mean']:.1f}",
                "runtime": f"{summary['runtime_mean']:.1f}s",
                "gpu": f"{summary['gpu_memory_peak_mean']:.1f}MB",
            }
        )

    headers = ["method", "val_acc_mean+/-std", "test_at_best_val_mean+/-std", "best_raw_test_mean+/-std", "best_epoch_mean", "runtime_mean", "gpu_memory_peak_mean"]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        print(
            " | ".join(
                [
                    row["method"],
                    row["val"],
                    row["test"],
                    row["raw"],
                    row["epoch"],
                    row["runtime"],
                    row["gpu"],
                ]
            )
        )


if __name__ == "__main__":
    main()
