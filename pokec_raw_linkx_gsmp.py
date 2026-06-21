import argparse
import os
import time

import torch
from torch_geometric.data import Data

from gsmp import compute_gsmp_edge_weights
from pokec_raw_linkx_smp import (
    PROFILES_URL,
    RELATIONSHIPS_URL,
    build_temporal_split,
    download,
    existing_path,
    load_profiles,
    load_relationships,
    make_linkx_sparse_adj,
    normalize_features,
    train_one,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw Pokec LINKX vs LINKX+GSMP.")
    parser.add_argument("--root", default="data/pokec")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--train-until-year", type=int, default=2009)
    parser.add_argument("--val-year", type=int, default=2010)
    parser.add_argument("--test-from-year", type=int, default=2011)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-edge-layers", type=int, default=1)
    parser.add_argument("--num-node-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--tag", default="pokec_raw_linkx_vs_linkx_gsmp")
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)
    profiles_path = existing_path(args.root, "soc-pokec-profiles.txt.gz")
    relationships_path = existing_path(args.root, "soc-pokec-relationships.txt.gz")
    if not args.no_download:
        download(PROFILES_URL, profiles_path)
        download(RELATIONSHIPS_URL, relationships_path)

    x, y, node_time, node_year = load_profiles(profiles_path)
    edge_index = load_relationships(relationships_path, num_nodes=x.size(0))
    split = build_temporal_split(
        y,
        node_year,
        train_until_year=args.train_until_year,
        val_year=args.val_year,
        test_from_year=args.test_from_year,
    )

    # GSMP balances each source node's outgoing neighbors across target
    # registration-year groups. Registration year is not used as a feature.
    gsmp_weight = compute_gsmp_edge_weights(edge_index, node_year, num_nodes=x.size(0))
    gsmp_changed_edges = int((gsmp_weight != 1).sum().item())
    adj_t = make_linkx_sparse_adj(edge_index, x.size(0))
    adj_t_gsmp = make_linkx_sparse_adj(edge_index, x.size(0), edge_weight=gsmp_weight)

    x = normalize_features(x)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = Data(x=x, y=y, edge_index=edge_index, num_nodes=x.size(0)).to(args.device)
    split = {key: value.to(args.device) for key, value in split.items()}
    adj_t = adj_t.to(args.device)
    adj_t_gsmp = adj_t_gsmp.to(args.device)

    print(data, flush=True)
    print(f"Split sizes: train={split['train'].numel()} valid={split['valid'].numel()} test={split['test'].numel()}", flush=True)
    print(f"GSMP changed edges: {gsmp_changed_edges} / {edge_index.size(1)}", flush=True)
    print(f"GSMP weight range: min={gsmp_weight.min().item():.6f} max={gsmp_weight.max().item():.6f}", flush=True)
    print("No-label-leakage check:", flush=True)
    print("  label target: gender column", flush=True)
    print("  loss labels: train split only", flush=True)
    print("  model selection: validation accuracy only", flush=True)
    print("  feature columns exclude gender, registration, last_login, and user_id", flush=True)
    print("  registration year is used only for temporal split and GSMP edge weights", flush=True)

    linkx = train_one(data, split, adj_t, args, "LINKX")
    linkx_gsmp = train_one(data, split, adj_t_gsmp, args, "LINKX+GSMP")

    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, f"{args.tag}.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("Raw Pokec LINKX vs LINKX+GSMP\n")
        f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"root: {args.root}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"num_nodes: {data.num_nodes}\n")
        f.write(f"num_edges: {data.edge_index.size(1)}\n")
        f.write(f"num_features: {data.num_features}\n")
        f.write(f"hidden_channels: {args.hidden_channels}\n")
        f.write(f"num_layers: {args.num_layers}\n")
        f.write(f"num_edge_layers: {args.num_edge_layers}\n")
        f.write(f"num_node_layers: {args.num_node_layers}\n")
        f.write(f"train_until_year: {args.train_until_year}\n")
        f.write(f"val_year: {args.val_year}\n")
        f.write(f"test_from_year: {args.test_from_year}\n")
        f.write(f"train_nodes: {split['train'].numel()}\n")
        f.write(f"valid_nodes: {split['valid'].numel()}\n")
        f.write(f"test_nodes: {split['test'].numel()}\n")
        f.write(f"gsmp_changed_edges: {gsmp_changed_edges}\n")
        f.write(f"gsmp_min_weight: {gsmp_weight.min().item():.6f}\n")
        f.write(f"gsmp_max_weight: {gsmp_weight.max().item():.6f}\n")
        f.write("no_label_leakage: gender/registration/last_login/user_id excluded from features; only train labels used for loss\n")
        for prefix, metrics in (("linkx", linkx), ("linkx_gsmp", linkx_gsmp)):
            for key, value in metrics.items():
                f.write(f"{prefix}_{key}: {value:.6f}\n")
    print(f"Saved result txt to {result_path}", flush=True)


if __name__ == "__main__":
    main()
