#!/usr/bin/env python3
"""Compatibility launcher for HGAMLP-HOPE impact experiments."""

from __future__ import annotations

import argparse
import runpy
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HGAMLP-HOPE impact experiments.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", default="hgamlp_hope")
    parser.add_argument("--impact-method", choices=("baseline", "smp", "ump", "gsmp"), required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--eval-every", type=int, required=True)
    parser.add_argument("--log-every", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--resume", default="false")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    entrypoint = repo_root / "ogbn_mag_temporal" / "main.py"

    if not entrypoint.exists():
        raise FileNotFoundError(f"Could not find existing experiment entry point: {entrypoint}")

    output_root = Path(args.output_root)
    checkpoint_dir = Path(args.checkpoint_dir)
    result_file = Path(args.result_file)
    run_result_dir = result_file.with_suffix("")

    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    run_result_dir.mkdir(parents=True, exist_ok=True)

    existing_jsons = set(run_result_dir.glob("results_*.json"))

    model_arg = "hgamLP_hope" if args.model == "hgamlp_hope" else args.model
    sys.argv = [
        str(entrypoint),
        "--dataset",
        args.dataset,
        "--root",
        args.data_root,
        "--model",
        model_arg,
        "--method",
        args.impact_method,
        "--epochs",
        str(args.epochs),
        "--eval-steps",
        str(args.eval_every),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--num-runs",
        "1",
        "--device",
        args.device,
        "--save-dir",
        str(run_result_dir),
        *args.extra_args,
    ]

    runpy.run_path(str(entrypoint), run_name="__main__")

    new_jsons = sorted(
        (path for path in run_result_dir.glob("results_*.json") if path not in existing_jsons),
        key=lambda path: path.stat().st_mtime,
    )
    if not new_jsons:
        new_jsons = sorted(run_result_dir.glob("results_*.json"), key=lambda path: path.stat().st_mtime)
    if new_jsons:
        shutil.copyfile(new_jsons[-1], result_file)


if __name__ == "__main__":
    main()
