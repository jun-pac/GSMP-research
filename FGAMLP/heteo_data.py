#the entire file is apdated from https://github.com/facebookresearch/NARS/blob/main/data.py
import os
import numpy as np
import torch
import dgl
import dgl.function as fn
from collections import defaultdict


###############################################################################
# Loading Relation Subsets
###############################################################################

def read_relation_subsets(fname):
    print("Reading Relation Subsets:")
    rel_subsets = []
    with open(fname) as f:
        for line in f:
            relations = line.strip().split(',')
            rel_subsets.append(relations)
            print(relations)
    return rel_subsets


def write_default_mag_relation_subsets(fname):
    """Write a small NARS-style relation subset file for ogbn-mag."""
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    rels = ["cites", "writes", "has_topic", "affiliated_with"]
    subsets = []
    for mask in range(1, 1 << len(rels)):
        subset = [rel for i, rel in enumerate(rels) if mask & (1 << i)]
        if any(rel in {"cites", "writes", "has_topic"} for rel in subset):
            subsets.append(subset)
    with open(fname, "w") as f:
        for subset in subsets:
            f.write(",".join(subset) + "\n")


def compute_mag_timestamps(g, strategy="mean"):
    """Compute per-node timestamps for ogbn-mag node types."""
    timestamps = {}
    if "year" not in g.nodes["paper"].data:
        raise RuntimeError("ogbn-mag paper year is missing; cannot run temporal methods.")

    paper_year = g.nodes["paper"].data["year"].squeeze().float().cpu()
    timestamps["paper"] = paper_year

    for ntype in g.ntypes:
        if ntype == "paper":
            continue
        num_nodes = g.number_of_nodes(ntype)
        out = torch.full((num_nodes,), -1.0)
        if strategy == "mean":
            sums = torch.zeros(num_nodes)
            counts = torch.zeros(num_nodes)
        elif strategy == "min":
            vals = torch.full((num_nodes,), float("inf"))
        elif strategy == "max":
            vals = torch.full((num_nodes,), float("-inf"))
        else:
            raise ValueError(f"Unknown timestamp strategy: {strategy}")

        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)
            if stype == ntype and dtype == "paper":
                node_ids, paper_ids = g.all_edges(etype=etype)
            elif stype == "paper" and dtype == ntype:
                paper_ids, node_ids = g.all_edges(etype=etype)
            else:
                continue

            connected_year = paper_year[paper_ids.cpu()]
            node_ids = node_ids.cpu()
            if strategy == "mean":
                sums.index_add_(0, node_ids, connected_year)
                counts.index_add_(0, node_ids, torch.ones_like(connected_year))
            elif strategy == "min":
                vals.scatter_reduce_(0, node_ids, connected_year, reduce="amin", include_self=True)
            elif strategy == "max":
                vals.scatter_reduce_(0, node_ids, connected_year, reduce="amax", include_self=True)

        if strategy == "mean":
            mask = counts > 0
            out[mask] = (sums[mask] / counts[mask]).round()
        else:
            mask = torch.isfinite(vals)
            out[mask] = vals[mask]
        timestamps[ntype] = out

    return timestamps


def _edge_weights_for_method(src_t, dst_t, src_ids, dst_ids, method, t_min, t_max):
    if method == "baseline":
        return torch.ones_like(src_t, dtype=torch.float32)
    if method == "ump":
        return (src_t <= dst_t).float()
    if method == "smp":
        delta = torch.abs(src_t - dst_t)
        boundary = torch.minimum(t_max - dst_t, dst_t - t_min)
        single = (delta == 0) | (delta > boundary)
        return torch.where(single, torch.full_like(delta, 2.0), torch.ones_like(delta)).float()
    if method == "gsmp":
        # User-requested GSMP:
        # b_{u->v} = 1 / C_u(time(v)); then normalize by source-node average b.
        counts = defaultdict(int)
        for src, dst_time in zip(src_ids.tolist(), dst_t.tolist()):
            counts[(int(src), int(dst_time))] += 1

        raw = torch.empty(len(src_ids), dtype=torch.float32)
        sums = defaultdict(float)
        degs = defaultdict(int)
        for i, (src, dst_time) in enumerate(zip(src_ids.tolist(), dst_t.tolist())):
            src = int(src)
            b = 1.0 / max(1, counts[(src, int(dst_time))])
            raw[i] = b
            sums[src] += b
            degs[src] += 1

        weights = torch.zeros_like(raw)
        for i, src in enumerate(src_ids.tolist()):
            src = int(src)
            mu = sums[src] / degs[src] if degs[src] > 0 else 0.0
            weights[i] = raw[i] / mu if mu > 0 else 0.0
        return weights
    raise ValueError(f"Unknown impact method: {method}")


###############################################################################
# Generate multi-hop neighbor-averaged feature for each relation subset
###############################################################################

def gen_rel_subset_feature(g, rel_subset, args, device):
    """
    Build relation subgraph given relation subset and generate multi-hop
    neighbor-averaged feature on this subgraph
    """
    device = "cpu"
    new_edges = {}
    edge_weights = {}
    ntypes = set()
    timestamps = compute_mag_timestamps(g, args.non_paper_time_strategy)
    valid_ts = torch.cat([ts[ts >= 0] for ts in timestamps.values()])
    t_min = valid_ts.min()
    t_max = valid_ts.max()
    for etype in rel_subset:
        stype, _, dtype = g.to_canonical_etype(etype)
        src_ids, dst_ids = g.all_edges(etype=etype)
        src_ids = src_ids.cpu().long()
        dst_ids = dst_ids.cpu().long()
        src_t = timestamps[stype][src_ids]
        dst_t = timestamps[dtype][dst_ids]
        valid = (src_t >= 0) & (dst_t >= 0)

        forward_keep = torch.ones_like(valid, dtype=torch.bool)
        reverse_keep = torch.ones_like(valid, dtype=torch.bool)
        if args.impact_method == "ump":
            forward_keep = (~valid) | (src_t <= dst_t)
            reverse_keep = (~valid) | (dst_t <= src_t)

        src_f = src_ids[forward_keep]
        dst_f = dst_ids[forward_keep]
        src_r = src_ids[reverse_keep]
        dst_r = dst_ids[reverse_keep]

        if args.impact_method == "smp":
            direct_single = valid & (
                (torch.abs(src_t - dst_t) == 0)
                | (torch.abs(src_t - dst_t) > torch.minimum(t_max - dst_t, dst_t - t_min))
            )
            reverse_single = valid & (
                (torch.abs(dst_t - src_t) == 0)
                | (torch.abs(dst_t - src_t) > torch.minimum(t_max - src_t, src_t - t_min))
            )
            src_f = torch.cat([src_f, src_ids[direct_single]])
            dst_f = torch.cat([dst_f, dst_ids[direct_single]])
            src_r = torch.cat([src_r, src_ids[reverse_single]])
            dst_r = torch.cat([dst_r, dst_ids[reverse_single]])

        new_edges[(stype, etype, dtype)] = (src_f.numpy(), dst_f.numpy())
        new_edges[(dtype, etype + "_r", stype)] = (dst_r.numpy(), src_r.numpy())

        if args.impact_method == "gsmp":
            w = _edge_weights_for_method(src_t, dst_t, src_ids, dst_ids, args.impact_method, t_min, t_max)
            w = torch.where(valid, w, torch.ones_like(w))
            edge_weights[(stype, etype, dtype)] = w[forward_keep]
            w_r = _edge_weights_for_method(dst_t, src_t, dst_ids, src_ids, args.impact_method, t_min, t_max)
            w_r = torch.where(valid, w_r, torch.ones_like(w_r))
            edge_weights[(dtype, etype + "_r", stype)] = w_r[reverse_keep]
        ntypes.add(stype)
        ntypes.add(dtype)
    new_g = dgl.heterograph(new_edges)
    if "paper" not in new_g.ntypes:
        raise ValueError(f"Relation subset {rel_subset} does not contain paper nodes.")

    for canonical_etype, weights in edge_weights.items():
        if canonical_etype in new_g.canonical_etypes:
            new_g.edges[canonical_etype].data["w"] = weights[:new_g.num_edges(canonical_etype)]

    # set node feature and calc deg
    for ntype in ntypes:
        num_nodes = new_g.number_of_nodes(ntype)
        if num_nodes < g.nodes[ntype].data["feat"].shape[0]:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"][:num_nodes, :]
        else:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"]
        deg = 0
        for etype in new_g.etypes:
            canonical_etype = new_g.to_canonical_etype(etype)
            _, _, dtype = canonical_etype
            if ntype == dtype:
                if args.impact_method == "gsmp" and "w" in new_g.edges[canonical_etype].data:
                    _, dst = new_g.all_edges(etype=canonical_etype)
                    w = new_g.edges[canonical_etype].data["w"]
                    if isinstance(deg, int):
                        deg = torch.zeros(num_nodes)
                    deg.index_add_(0, dst.cpu(), w.cpu())
                else:
                    deg = deg + new_g.in_degrees(etype=etype)
        norm = 1.0 / deg.float()
        norm[torch.isinf(norm)] = 0
        new_g.nodes[ntype].data["norm"] = norm.view(-1, 1).to(device)

    res = []

    # compute k-hop feature
    for hop in range(1, args.num_hops + 1):
        ntype2feat = {}
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)
            canonical_etype = new_g.to_canonical_etype(etype)
            if args.impact_method == "gsmp":
                new_g[canonical_etype].update_all(
                    fn.u_mul_e(f'hop_{hop-1}', "w", 'm'), fn.sum('m', 'new_feat')
                )
            else:
                new_g[canonical_etype].update_all(fn.copy_u(f'hop_{hop-1}', 'm'), fn.sum('m', 'new_feat'))
            new_feat = new_g.nodes[dtype].data.pop("new_feat")
            assert("new_feat" not in new_g.nodes[stype].data)
            if dtype in ntype2feat:
                ntype2feat[dtype] += new_feat
            else:
                ntype2feat[dtype] = new_feat
        for ntype in new_g.ntypes:
            assert ntype in ntype2feat  # because subgraph is not directional
            feat_dict = new_g.nodes[ntype].data
            old_feat = feat_dict.pop(f"hop_{hop-1}")
            if ntype == "paper":
                res.append(old_feat.cpu())
            feat_dict[f"hop_{hop}"] = ntype2feat.pop(ntype).mul_(feat_dict["norm"])

    res.append(new_g.nodes["paper"].data.pop(f"hop_{args.num_hops}").cpu())
    return res


###############################################################################
# Dataset (ACM, MAG, OAG) loading
###############################################################################


def load_data(device, args):
    device = 'cpu'
    with torch.no_grad():
        if args.dataset == "ogbn-mag":
            return load_mag(device, args)
        else:
            raise RuntimeError(f"Dataset {args.dataset} not supported")


def load_mag(device, args):
    from ogb.nodeproppred import DglNodePropPredDataset
    path = os.path.join(args.emb_path, f"TransE_mag")
    required = ["author.pt", "field_of_study.pt", "institution.pt"]
    missing = [name for name in required if not os.path.isfile(os.path.join(path, name))]
    if missing:
        raise FileNotFoundError(
            "Missing TransE embeddings required by FGAMLP ogbn-mag loader: "
            + ", ".join(os.path.join(path, name) for name in missing)
            + ". Set --emb-path to the directory that contains TransE_mag/."
        )
    dataset = DglNodePropPredDataset(
        name="ogbn-mag", root=args.root)
    g, labels = dataset[0]
    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx["train"]['paper']
    val_nid = splitted_idx["valid"]['paper']
    test_nid = splitted_idx["test"]['paper']
    features = g.nodes['paper'].data['feat']
    author_emb = torch.load(os.path.join(path, "author.pt"), map_location=torch.device("cpu")).float()
    topic_emb = torch.load(os.path.join(path, "field_of_study.pt"), map_location=torch.device("cpu")).float()
    institution_emb = torch.load(os.path.join(path, "institution.pt"), map_location=torch.device("cpu")).float()

    g = g.to(device)
    g.nodes["author"].data["feat"] = author_emb.to(device)
    g.nodes["institution"].data["feat"] = institution_emb.to(device)
    g.nodes["field_of_study"].data["feat"] = topic_emb.to(device)
    g.nodes["paper"].data["feat"] = features.to(device)
    paper_dim = g.nodes["paper"].data["feat"].shape[1]
    author_dim = g.nodes["author"].data["feat"].shape[1]
    if paper_dim != author_dim:
        paper_feat = g.nodes["paper"].data.pop("feat")
        rand_weight = torch.Tensor(paper_dim, author_dim).uniform_(-0.5, 0.5)
        g.nodes["paper"].data["feat"] = torch.matmul(paper_feat, rand_weight.to(device))
        print(f"Randomly project paper feature from dimension {paper_dim} to {author_dim}")

    labels = labels['paper'].to(device).squeeze()
    n_classes = int(labels.max() - labels.min()) + 1

    return g, labels, n_classes, train_nid, val_nid, test_nid



def preprocess_features(g, rel_subsets, args, device):
    # pre-process heterogeneous graph g to generate neighbor-averaged features
    # for each relation subsets
    num_paper, feat_size = g.nodes["paper"].data["feat"].shape
    new_feats = [torch.zeros(num_paper, len(rel_subsets), feat_size) for _ in range(args.num_hops + 1)]
    print("Start generating features for each sub-metagraph:")
    for subset_id, subset in enumerate(rel_subsets):
        print(subset)
        feats = gen_rel_subset_feature(g, subset, args, device)
        for i in range(args.num_hops + 1):
            feat = feats[i]
            new_feats[i][:feat.shape[0], subset_id, :] = feat
        feats = None
    return new_feats
