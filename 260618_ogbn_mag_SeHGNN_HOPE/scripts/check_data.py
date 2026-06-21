#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
HOPE_DIR = PROJECT_DIR / "HOPE"
sys.path.insert(0, str(HOPE_DIR))

from utils import load_mag  # noqa: E402


EXPECTED_COUNTS = {
    "P": 736389,
    "A": 1134649,
    "I": 8740,
    "F": 59965,
}
REQUIRED_ETYPES = {
    ("A", "A-I", "I"),
    ("I", "I-A", "A"),
    ("A", "A-P", "P"),
    ("P", "P-A", "A"),
    ("P", "P-P", "P"),
    ("P", "P-F", "F"),
    ("F", "F-P", "P"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate ogbn-mag + HOPE LINE embedding inputs.")
    parser.add_argument("--root", default=str(HOPE_DIR / "dataset"))
    parser.add_argument("--emb-path", "--emb_path", dest="emb_path",
                        default=str(HOPE_DIR / "dataset" / "ogbn_mag"))
    parser.add_argument("--download-ok", action="store_true",
                        help="allow OGB to download data if it is not present")
    return parser.parse_args()


def main():
    args = parse_args()
    emb_path = Path(args.emb_path)
    mag_p = emb_path / "mag.p"
    if not mag_p.exists():
        raise FileNotFoundError(
            f"Missing LINE embedding file: {mag_p}. "
            "Do not regenerate it in this experiment; set MAG_P and run scripts/prepare_data_links.sh."
        )

    if not args.download_ok:
        processed = Path(args.root) / "ogbn_mag" / "processed"
        if not processed.exists():
            raise FileNotFoundError(
                f"Missing OGB processed directory: {processed}. "
                "Pass --download-ok only if you intentionally want OGB to download ogbn-mag."
            )

    load_args = SimpleNamespace(
        dataset="ogbn-mag",
        root=args.root,
        emb_path=str(emb_path),
        extra_embedding="Line",
        embed_size=256,
        use_sparse_tools=False,
        gsmp_first_layer=True,
        gsmp_apply_label_prop=False,
        gsmp_time_source="all",
        gsmp_derived_time="mode",
    )
    new_g, labels, num_papers, n_classes, train_nid, val_nid, test_nid, _ = load_mag(load_args)
    print(f"num_papers={num_papers} n_classes={n_classes} labels={tuple(labels.shape)}")
    print(f"split train={len(train_nid)} valid={len(val_nid)} test={len(test_nid)}")

    for ntype, expected in EXPECTED_COUNTS.items():
        actual = new_g.num_nodes(ntype)
        print(f"nodes[{ntype}]={actual}")
        if actual != expected:
            raise RuntimeError(f"Unexpected {ntype} node count: {actual}, expected {expected}")

    canonical = set(new_g.canonical_etypes)
    missing = REQUIRED_ETYPES - canonical
    print(f"canonical_etypes={sorted(canonical)}")
    if missing:
        raise RuntimeError(f"Missing compact graph edge types: {sorted(missing)}")

    time_dict = getattr(new_g, "_gsmp_time_dict")
    for ntype, values in time_dict.items():
        unknown = int((values == -1).sum().item())
        print(f"time[{ntype}] shape={tuple(values.shape)} unknown={unknown}")

    print("Data check passed.")


if __name__ == "__main__":
    main()
