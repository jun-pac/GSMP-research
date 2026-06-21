#!/usr/bin/env python3
"""Prepare TransE embeddings expected by FGAMLP's ogbn-mag loader."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.datasets import OGB_MAG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyg-root", default="data/pyg_ogb_mag_transe")
    parser.add_argument("--out-dir", default="data/TransE_mag")
    args = parser.parse_args()

    pyg_root = Path(args.pyg_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading/downloading PyG OGB_MAG TransE data under {pyg_root}")
    dataset = OGB_MAG(root=str(pyg_root), preprocess="transe")
    data = dataset[0]

    for node_type in ("author", "field_of_study", "institution"):
        x = data[node_type].x
        if x is None:
            raise RuntimeError(f"PyG OGB_MAG did not provide x for {node_type}")
        path = out_dir / f"{node_type}.pt"
        torch.save(x.cpu().float(), path)
        print(f"Wrote {path} shape={tuple(x.shape)}")


if __name__ == "__main__":
    main()
