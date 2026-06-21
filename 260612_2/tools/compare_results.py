import argparse
import json
from pathlib import Path


def fmt(mean, std):
    return f"{100 * float(mean):.2f}+/-{100 * float(std):.2f}"


def label(summary):
    name = summary["method"]
    if summary.get("use_pgsmp"):
        name = (
            f"{name}(alpha={summary.get('pgsmp_alpha')}, "
            f"depth={summary.get('pgsmp_depth')}, "
            f"norm={summary.get('pgsmp_norm')}, "
            f"self={summary.get('pgsmp_self_mode')})"
        )
    return name


def main():
    parser = argparse.ArgumentParser(description="Print a compact comparison table from final_summary.json files.")
    parser.add_argument("summaries", nargs="+", type=Path)
    args = parser.parse_args()

    headers = [
        "method",
        "val_acc_mean+/-std",
        "test_at_best_val_mean+/-std",
        "best_raw_test_mean+/-std",
        "best_epoch_mean",
        "runtime_mean",
        "gpu_memory_peak_mean",
        "pgsmp_preprocessing_time",
        "cache_reused",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for path in args.summaries:
        with open(path) as f:
            summary = json.load(f)
        row = [
            label(summary),
            fmt(summary["val_mean"], summary["val_std"]),
            fmt(summary["test_at_best_val_mean"], summary["test_at_best_val_std"]),
            fmt(summary["best_raw_test_mean"], summary["best_raw_test_std"]),
            f"{float(summary['best_epoch_mean']):.1f}",
            f"{float(summary['runtime_mean']):.1f}s",
            f"{float(summary['gpu_memory_peak_mean']):.1f}MB",
            f"{float(summary.get('pgsmp_preprocessing_time', 0.0)):.1f}s",
            str(summary.get("pgsmp_cache_reused", False)),
        ]
        print(" | ".join(row))


if __name__ == "__main__":
    main()
