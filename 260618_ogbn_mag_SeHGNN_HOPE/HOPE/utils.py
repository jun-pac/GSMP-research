import os
import gc
import random
import hashlib
from pathlib import Path

import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_sparse import remove_diag, set_diag

import numpy as np
from tqdm import tqdm
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from tqdm import tqdm

GSMP_UNKNOWN_TIME = -1


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # dgl.seed(seed)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def is_metapath_key(key):
    return isinstance(key, str) and key.isupper() and all(ch in "PAIF" for ch in key)


def _mode_by_destination(src_values, dst, num_dst, unknown=GSMP_UNKNOWN_TIME):
    src_values = src_values.cpu().long()
    dst = dst.cpu().long()
    out = torch.full((num_dst,), unknown, dtype=torch.long)
    if dst.numel() == 0:
        return out

    valid = src_values != unknown
    if int(valid.sum()) == 0:
        return out

    src_values = src_values[valid]
    dst = dst[valid]
    unique_values, value_inverse = torch.unique(src_values, sorted=True, return_inverse=True)
    pair_code = dst * unique_values.numel() + value_inverse.long()
    unique_pairs, _, pair_counts = torch.unique(
        pair_code, sorted=False, return_inverse=True, return_counts=True)

    pair_dst = (unique_pairs // unique_values.numel()).numpy()
    pair_value = unique_values[(unique_pairs % unique_values.numel()).long()].numpy()
    pair_counts = pair_counts.numpy()

    # Sort by destination, then largest count, then earliest year for deterministic ties.
    order = np.lexsort((pair_value, -pair_counts, pair_dst))
    sorted_dst = pair_dst[order]
    sorted_value = pair_value[order]
    first = np.ones(sorted_dst.shape[0], dtype=bool)
    first[1:] = sorted_dst[1:] != sorted_dst[:-1]
    out[torch.from_numpy(sorted_dst[first]).long()] = torch.from_numpy(sorted_value[first]).long()
    return out


def _time_summary(name, value, unknown=GSMP_UNKNOWN_TIME):
    value = value.cpu().long()
    unknown_count = int((value == unknown).sum().item())
    known = value[value != unknown]
    if known.numel():
        print(
            f"GSMP time[{name}]: nodes={value.numel()} unknown={unknown_count} "
            f"bins={known.unique().numel()} min={int(known.min())} max={int(known.max())}",
            flush=True,
        )
    else:
        print(
            f"GSMP time[{name}]: nodes={value.numel()} unknown={unknown_count} bins=0",
            flush=True,
        )


def build_mag_time_dict(original_g, new_g, args):
    """Build deterministic node-time bins for compact P/A/I/F ogbn-mag graph."""
    derived_time = getattr(args, "gsmp_derived_time", "mode")
    if derived_time != "mode":
        raise ValueError(f"Unsupported --gsmp-derived-time={derived_time}; only 'mode' is implemented.")

    if "year" not in original_g.nodes["paper"].data:
        raise RuntimeError("ogbn-mag paper year is missing; GSMP needs paper publication years.")

    paper_time = original_g.nodes["paper"].data["year"].squeeze().cpu().long()
    if getattr(args, "gsmp_time_source", "all") == "train-only":
        train_nid = getattr(args, "_gsmp_train_paper_nid", None)
        if train_nid is None:
            raise RuntimeError("--gsmp-time-source train-only requires training paper ids.")
        masked = torch.full_like(paper_time, GSMP_UNKNOWN_TIME)
        masked[train_nid.cpu().long()] = paper_time[train_nid.cpu().long()]
        paper_time = masked

    new_g.nodes["P"].data["year"] = paper_time

    a_src, p_dst = new_g.edges(etype="A-P", order="eid")
    author_time = _mode_by_destination(
        paper_time[p_dst.long()], a_src.long(), new_g.num_nodes("A"))

    p_src, f_dst = new_g.edges(etype="P-F", order="eid")
    field_time = _mode_by_destination(
        paper_time[p_src.long()], f_dst.long(), new_g.num_nodes("F"))

    a_src, i_dst = new_g.edges(etype="A-I", order="eid")
    institution_time = _mode_by_destination(
        author_time[a_src.long()], i_dst.long(), new_g.num_nodes("I"))

    time_dict = {
        "P": paper_time,
        "A": author_time,
        "I": institution_time,
        "F": field_time,
    }
    for ntype, time_value in time_dict.items():
        _time_summary(ntype, time_value)
    return time_dict


def _temporal_cache_path(new_g, etype, src_time, normalizer, cache_dir, method="gsmp",
                         dst_time=None):
    if cache_dir is None:
        return None
    if method == "gsmp":
        return _gsmp_cache_path(new_g, etype, src_time, normalizer, cache_dir)
    src_type, _, dst_type = new_g.to_canonical_etype(etype)
    src_time = src_time.cpu().long()
    if dst_time is not None:
        dst_time = dst_time.cpu().long()
    metadata = (
        method,
        str(etype),
        src_type,
        dst_type,
        int(new_g.num_nodes(src_type)),
        int(new_g.num_nodes(dst_type)),
        int(new_g.num_edges(etype)),
        normalizer,
        int(src_time.numel()),
        int(src_time.min().item()) if src_time.numel() else 0,
        int(src_time.max().item()) if src_time.numel() else 0,
        int(src_time.sum().item()) if src_time.numel() else 0,
        int(dst_time.numel()) if dst_time is not None else 0,
        int(dst_time.min().item()) if dst_time is not None and dst_time.numel() else 0,
        int(dst_time.max().item()) if dst_time is not None and dst_time.numel() else 0,
        int(dst_time.sum().item()) if dst_time is not None and dst_time.numel() else 0,
    )
    digest = hashlib.sha1(repr(metadata).encode("utf-8")).hexdigest()[:12]
    safe_etype = str(etype).replace("/", "_").replace(":", "_")
    return Path(cache_dir) / f"{safe_etype}_{method}_{normalizer}_{digest}.pt"


def _gsmp_cache_path(new_g, etype, src_time, normalizer, cache_dir):
    if cache_dir is None:
        return None
    src_type, _, dst_type = new_g.to_canonical_etype(etype)
    src_time = src_time.cpu().long()
    metadata = (
        str(etype),
        src_type,
        dst_type,
        int(new_g.num_nodes(src_type)),
        int(new_g.num_nodes(dst_type)),
        int(new_g.num_edges(etype)),
        normalizer,
        int(src_time.numel()),
        int(src_time.min().item()) if src_time.numel() else 0,
        int(src_time.max().item()) if src_time.numel() else 0,
        int(src_time.sum().item()) if src_time.numel() else 0,
    )
    digest = hashlib.sha1(repr(metadata).encode("utf-8")).hexdigest()[:12]
    safe_etype = str(etype).replace("/", "_").replace(":", "_")
    return Path(cache_dir) / f"{safe_etype}_{normalizer}_{digest}.pt"


def compute_gsmp_edge_weights(new_g, etype, src_time, normalizer="nonempty",
                              cache_dir=None, use_cache=True):
    if normalizer not in ("nonempty", "global"):
        raise ValueError(f"Unsupported GSMP normalizer: {normalizer}")

    cache_path = _gsmp_cache_path(new_g, etype, src_time, normalizer, cache_dir)
    if use_cache and cache_path is not None and cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    _, _, dst_type = new_g.to_canonical_etype(etype)
    src, dst = new_g.edges(etype=etype, order="eid")
    src = src.cpu().long()
    dst = dst.cpu().long()
    edge_time = src_time.cpu().long()[src]

    if edge_time.numel() == 0:
        weights = torch.empty((0,), dtype=torch.float32)
    else:
        global_num_bins = max(1, int(torch.unique(src_time.cpu().long()).numel()))
        unique_time, edge_time_inverse = torch.unique(edge_time, sorted=True, return_inverse=True)
        num_bins = int(unique_time.numel())
        pair_code = dst * num_bins + edge_time_inverse.long()
        unique_pairs, pair_inverse, pair_counts = torch.unique(
            pair_code, sorted=False, return_inverse=True, return_counts=True)

        count_per_edge = pair_counts[pair_inverse].float()
        if normalizer == "nonempty":
            pair_dst = unique_pairs // num_bins
            nonempty_bins = torch.bincount(
                pair_dst.long(), minlength=new_g.num_nodes(dst_type)).float()
            bin_norm = nonempty_bins[dst].clamp_min(1.0)
        else:
            bin_norm = torch.full_like(count_per_edge, float(global_num_bins))
        weights = (1.0 / (bin_norm * count_per_edge)).float()

    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
        torch.save(weights, tmp_path)
        os.replace(tmp_path, cache_path)
        print(f"Saved GSMP edge weights: {cache_path}", flush=True)

    return weights


def compute_smp_edge_weights(new_g, etype, src_time, dst_time, cache_dir=None,
                             use_cache=True):
    cache_path = _temporal_cache_path(
        new_g, etype, src_time, "target", cache_dir, method="smp", dst_time=dst_time)
    if use_cache and cache_path is not None and cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    _, _, dst_type = new_g.to_canonical_etype(etype)
    src, dst = new_g.edges(etype=etype, order="eid")
    src = src.cpu().long()
    dst = dst.cpu().long()
    src_edge_time = src_time.cpu().long()[src]
    dst_edge_time = dst_time.cpu().long()[dst]

    if src_edge_time.numel() == 0:
        weights = torch.empty((0,), dtype=torch.float32)
    else:
        all_time = torch.cat([src_time.cpu().long(), dst_time.cpu().long()])
        known_time = all_time[all_time != GSMP_UNKNOWN_TIME]
        if known_time.numel() == 0:
            raw = torch.ones(src_edge_time.numel(), dtype=torch.float32)
        else:
            t_min = int(known_time.min().item())
            t_max = int(known_time.max().item())
            valid = (src_edge_time != GSMP_UNKNOWN_TIME) & (dst_edge_time != GSMP_UNKNOWN_TIME)
            raw = torch.ones(src_edge_time.numel(), dtype=torch.float32)
            if int(valid.sum().item()) > 0:
                src_valid = src_edge_time[valid].float()
                dst_valid = dst_edge_time[valid].float()
                delta = torch.abs(src_valid - dst_valid)
                radius = torch.minimum(dst_valid - float(t_min), float(t_max) - dst_valid)
                single = (delta == 0) | (delta > radius)
                raw[valid] = torch.where(
                    single,
                    torch.full_like(src_valid, 2.0, dtype=torch.float32),
                    torch.ones_like(src_valid, dtype=torch.float32),
                )

        denom = torch.zeros(new_g.num_nodes(dst_type), dtype=torch.float32)
        denom.scatter_add_(0, dst.long(), raw)
        weights = raw / denom[dst].clamp_min(1e-12)

    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
        torch.save(weights, tmp_path)
        os.replace(tmp_path, cache_path)
        print(f"Saved SMP edge weights: {cache_path}", flush=True)

    return weights


def temporal_update_all(new_g, etype, k, current_dst_name, src_time, args,
                        dst_time=None):
    method = getattr(args, "temporal_first_layer_method", "gsmp")
    normalizer = getattr(args, "gsmp_normalizer", "nonempty")
    use_cache = not getattr(args, "no_gsmp_cache", False)
    cache_dir = getattr(args, "gsmp_cache_dir", None)
    if method == "gsmp":
        edge_weight = compute_gsmp_edge_weights(
            new_g, etype, src_time, normalizer=normalizer,
            cache_dir=cache_dir, use_cache=use_cache)
    elif method == "smp":
        if dst_time is None:
            raise RuntimeError("SMP update needs destination timestamps.")
        edge_weight = compute_smp_edge_weights(
            new_g, etype, src_time, dst_time,
            cache_dir=cache_dir, use_cache=use_cache)
    else:
        raise ValueError(f"Unsupported temporal first-layer method: {method}")
    edge_weight = edge_weight.to(dtype=new_g.nodes[new_g.to_canonical_etype(etype)[0]].data[k].dtype)
    weight_key = "_temporal_w"
    new_g.edges[etype].data[weight_key] = edge_weight
    try:
        new_g[etype].update_all(
            fn.u_mul_e(k, weight_key, "m"),
            fn.sum("m", current_dst_name), etype=etype)
    finally:
        if weight_key in new_g.edges[etype].data:
            del new_g.edges[etype].data[weight_key]


def should_use_gsmp_update(gsmp_first_layer, gsmp_scope, hop, tgt_type, src_type, dst_type,
                           is_label_propagation=False, gsmp_apply_label_prop=False):
    if not gsmp_first_layer:
        return False
    if tgt_type != "P" or src_type != "P" or dst_type != "P":
        return False
    if is_label_propagation and not gsmp_apply_label_prop:
        return False
    if gsmp_scope == "first-hop":
        return hop == 1
    if gsmp_scope == "paper-stack":
        return True
    raise ValueError(f"Unsupported GSMP scope: {gsmp_scope}")


def hg_propagate(new_g, tgt_type, num_hops, max_hops, extra_metapath, echo=False,
                 gsmp_first_layer=False, gsmp_time_dict=None,
                 gsmp_normalizer="nonempty", gsmp_scope="paper-stack",
                 is_label_propagation=False, gsmp_apply_label_prop=False, args=None,
                 temporal_first_layer_method="gsmp"):
    for hop in range(1, max_hops):
        reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)

            for k in list(new_g.nodes[stype].data.keys()):
                if len(k) == hop:
                    current_dst_name = f'{dtype}{k}'
                    if (hop == num_hops and dtype != tgt_type and k not in reserve_heads) \
                      or (hop > num_hops and k not in reserve_heads):
                        continue
                    if echo: print(k, etype, current_dst_name)
                    use_gsmp = should_use_gsmp_update(
                        gsmp_first_layer, gsmp_scope, hop, tgt_type, stype, dtype,
                        is_label_propagation=is_label_propagation,
                        gsmp_apply_label_prop=gsmp_apply_label_prop)
                    if use_gsmp:
                        if gsmp_time_dict is None or stype not in gsmp_time_dict:
                            raise RuntimeError(f"Missing GSMP time tensor for source node type {stype}.")
                        if temporal_first_layer_method == "smp" and dtype not in gsmp_time_dict:
                            raise RuntimeError(f"Missing SMP time tensor for destination node type {dtype}.")
                        if args is None:
                            class _Args:
                                pass
                            args = _Args()
                            args.gsmp_normalizer = gsmp_normalizer
                            args.temporal_first_layer_method = temporal_first_layer_method
                        else:
                            args.temporal_first_layer_method = temporal_first_layer_method
                        method_name = temporal_first_layer_method.upper()
                        print(
                            f"{method_name} {gsmp_scope} update etype={etype} src={stype} dst={dtype} "
                            f"key={k} out={current_dst_name} normalizer={gsmp_normalizer}",
                            flush=True,
                        )
                        temporal_update_all(
                            new_g, etype, k, current_dst_name, gsmp_time_dict[stype], args,
                            dst_time=gsmp_time_dict.get(dtype))
                    else:
                        new_g[etype].update_all(
                            fn.copy_u(k, 'm'),
                            fn.mean('m', current_dst_name), etype=etype)

        # remove no-use items
        for ntype in new_g.ntypes:
            if ntype == tgt_type: continue
            removes = []
            for k in new_g.nodes[ntype].data.keys():
                if len(k) <= hop:
                    removes.append(k)
            for k in removes:
                new_g.nodes[ntype].data.pop(k)
            if echo and len(removes): print('remove', removes)
        gc.collect()

        if echo: print(f'-- hop={hop} ---')
        for ntype in new_g.ntypes:
            for k, v in new_g.nodes[ntype].data.items():
                if echo: print(f'{ntype} {k} {v.shape}')
        if echo: print(f'------\n')

    return new_g


def clear_hg(new_g, echo=False):
    if echo: print('Remove keys left after propagation')
    for ntype in new_g.ntypes:
        keys = list(new_g.nodes[ntype].data.keys())
        if len(keys):
            if echo: print(ntype, keys)
            for k in keys:
                new_g.nodes[ntype].data.pop(k)
    return new_g


def check_acc(preds_dict, condition, init_labels, train_nid, val_nid, test_nid):
    mask_train, mask_val, mask_test = [], [], []
    remove_label_keys = []
    na, nb, nc = len(train_nid), len(val_nid), len(test_nid)

    for k, v in preds_dict.items():
        pred = v.argmax(1)

        a, b, c = pred[train_nid] == init_labels[train_nid], \
                  pred[val_nid] == init_labels[val_nid], \
                  pred[test_nid] == init_labels[test_nid]
        ra, rb, rc = a.sum() / len(train_nid), b.sum() / len(val_nid), c.sum() / len(test_nid)

        vv = torch.log((v / (v.sum(1, keepdim=True) + 1e-6)).clamp(1e-6, 1-1e-6))
        la, lb, lc = F.nll_loss(vv[train_nid], init_labels[train_nid]), \
                     F.nll_loss(vv[val_nid], init_labels[val_nid]), \
                     F.nll_loss(vv[test_nid], init_labels[test_nid])

        if condition(ra, rb, rc, k):
            mask_train.append(a)
            mask_val.append(b)
            mask_test.append(c)
        else:
            remove_label_keys.append(k)
        print(k, ra, rb, rc, la, lb, lc, (ra/rb-1)*100, (ra/rc-1)*100, (1-la/lb)*100, (1-la/lc)*100)

    print(set(list(preds_dict.keys())) - set(remove_label_keys))
    print((torch.stack(mask_train, dim=0).sum(0) > 0).sum() / len(train_nid))
    print((torch.stack(mask_val, dim=0).sum(0) > 0).sum() / len(val_nid))
    print((torch.stack(mask_test, dim=0).sum(0) > 0).sum() / len(test_nid))
    return remove_label_keys


def _finite_summary(name, value):
    value = value.detach()
    finite = torch.isfinite(value)
    if finite.any():
        finite_value = value[finite].float()
        return (
            f"{name}: shape={tuple(value.shape)} dtype={value.dtype} "
            f"finite={int(finite.sum().item())}/{value.numel()} "
            f"min={float(finite_value.min().item()):.6g} "
            f"max={float(finite_value.max().item()):.6g}"
        )
    return (
        f"{name}: shape={tuple(value.shape)} dtype={value.dtype} "
        f"finite=0/{value.numel()}"
    )


def _raise_if_nonfinite(loss, logits, context):
    if torch.isfinite(loss).all():
        return
    raise FloatingPointError(
        f"Non-finite training loss at {context}: loss={loss.detach().float().cpu().item()} "
        f"{_finite_summary('logits', logits)}"
    )


def train(model, train_loader, loss_fcn, optimizer, evaluator, device,
          feats, label_feats, labels_cuda, label_emb, mask=None, lamb=0.0, scalar=None):
    model.train()
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []

    for batch in train_loader:
        batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        # if mask is not None:
        #     batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
        # else:
        #     batch_mask = None
        batch_label_emb = label_emb[batch].to(device)
        batch_y = labels_cuda[batch]

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                x = model(batch_feats, batch_labels_feats, batch_label_emb)
                loss_train = loss_fcn(x, batch_y)
                if "HOPE" in model.aggregation:
                    loss_train = loss_train + lamb * model.mohe.expert_prototypes_loss
            _raise_if_nonfinite(loss_train, x, "stage=train")
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            x = model(batch_feats, batch_labels_feats, batch_label_emb)
            loss_train = loss_fcn(x, batch_y)
            if "HOPE" in model.aggregation:
                loss_train = loss_train + lamb * model.mohe.expert_prototypes_loss
            _raise_if_nonfinite(loss_train, x, "stage=train")
            loss_train.backward()
            optimizer.step()

        y_true.append(batch_y.cpu().to(torch.long))
        y_pred.append(x.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, acc


def train_multi_stage(model, train_loader, enhance_loader, loss_fcn, optimizer, evaluator, device,
                      feats, label_feats, labels, label_emb, predict_prob, gama, lamb=0.0, scalar=None):
    model.train()
    loss_fcn = nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    total_loss = 0
    loss_l1, loss_l2 = 0., 0.
    iter_num = 0
    for idx_1, idx_2 in zip(train_loader, enhance_loader):
        idx = torch.cat((idx_1, idx_2), dim=0)
        L1_ratio = len(idx_1) * 1.0 / (len(idx_1) + len(idx_2))
        L2_ratio = len(idx_2) * 1.0 / (len(idx_1) + len(idx_2))

        batch_feats = {k: x[idx].to(device) for k, x in feats.items()}
        batch_labels_feats = {k: x[idx].to(device) for k, x in label_feats.items()}
        batch_label_emb = label_emb[idx].to(device)
        y = labels[idx_1].to(torch.long).to(device)
        extra_weight, extra_y = predict_prob[idx_2].max(dim=1)
        extra_weight = extra_weight.to(device)
        extra_y = extra_y.to(device)

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                x = model(batch_feats, batch_labels_feats, batch_label_emb)
                L1 = loss_fcn(x[:len(idx_1)],  y)
                L2 = F.cross_entropy(x[len(idx_1):], extra_y, reduction='none')
                L2 = (L2 * extra_weight).sum() / len(idx_2)
                loss_train = L1_ratio * L1 + gama * L2_ratio * L2
                if "HOPE" in model.aggregation:
                    loss_train = loss_train + lamb * model.mohe.expert_prototypes_loss
            _raise_if_nonfinite(loss_train, x, "stage=train_multi_stage")
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            x = model(batch_feats, label_emb[idx].to(device))
            L1 = loss_fcn(x[:len(idx_1)],  y)
            L2 = F.cross_entropy(x[len(idx_1):], extra_y, reduction='none')
            L2 = (L2 * extra_weight).sum() / len(idx_2)
            loss_train = L1_ratio * L1 + gama * L2_ratio * L2
            if "HOPE" in model.aggregation:
                loss_train = loss_train + lamb * model.mohe.expert_prototypes_loss
            _raise_if_nonfinite(loss_train, x, "stage=train_multi_stage")
            loss_train.backward()
            optimizer.step()

        y_true.append(labels[idx_1].to(torch.long))
        y_pred.append(x[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        loss_l1 += L1.item()
        loss_l2 += L2.item()
        iter_num += 1

    print(loss_l1 / iter_num, loss_l2 / iter_num)
    loss = total_loss / iter_num
    approx_acc = evaluator(torch.cat(y_true, dim=0),torch.cat(y_pred, dim=0))
    return loss, approx_acc


@torch.no_grad()
def gen_output_torch(model, feats, label_feats, label_emb, test_loader, device):
    model.eval()
    preds = []
    for batch in tqdm(test_loader):
        batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        batch_label_emb = label_emb[batch].to(device)
        x = model(batch_feats, batch_labels_feats, batch_label_emb)
        preds.append(x.cpu())
    preds = torch.cat(preds, dim=0)
    return preds


def get_ogb_evaluator(dataset):
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        })["acc"]


def load_dataset(args):
    if args.dataset == 'ogbn-mag':
        # train/val/test 629571/64879/41939
        return load_mag(args)
    else:
        assert 0, 'Only allowed [ogbn-products, ogbn-proteins, ogbn-arxiv, ogbn-papers100M, ogbn-mag]'



def load_mag(args, symmetric=True):
    dataset = DglNodePropPredDataset(name=args.dataset, root=args.root)
    splitted_idx = dataset.get_idx_split()

    g, init_labels = dataset[0]
    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx['train']['paper']
    val_nid = splitted_idx['valid']['paper']
    test_nid = splitted_idx['test']['paper']

    features = g.nodes['paper'].data['feat']
    paper_year = None
    if 'year' in g.nodes['paper'].data:
        paper_year = g.nodes['paper'].data['year'].squeeze().cpu().long()
    if args.extra_embedding == "Line":
        print(f'Use extra embeddings generated with the Line method') 
        import pickle
        nrl_cache_path = os.path.join(args.emb_path, 'mag.p')
        with open(nrl_cache_path, "rb") as f:
            nrl_embedding_dict = pickle.load(f)
        author_emb = torch.Tensor(nrl_embedding_dict['author'])
        topic_emb = torch.Tensor(nrl_embedding_dict['field_of_study'])
        institution_emb = torch.Tensor(nrl_embedding_dict['institution'])
    else:
        author_emb = torch.Tensor(g.num_nodes('author'), args.embed_size).uniform_(-0.5, 0.5)
        topic_emb = torch.Tensor(g.num_nodes('field_of_study'), args.embed_size).uniform_(-0.5, 0.5)
        institution_emb = torch.Tensor(g.num_nodes('institution'), args.embed_size).uniform_(-0.5, 0.5)

    g.nodes['paper'].data['feat'] = features
    g.nodes['author'].data['feat'] = author_emb
    g.nodes['institution'].data['feat'] = institution_emb
    g.nodes['field_of_study'].data['feat'] = topic_emb

    init_labels = init_labels['paper'].squeeze()
    n_classes = int(init_labels.max()) + 1
    evaluator = get_ogb_evaluator(args.dataset)

    # for k in g.ntypes:
    #     print(k, g.ndata['feat'][k].shape)
    for k in g.ntypes:
        print(k, g.nodes[k].data['feat'].shape)

    adjs = []
    for i, etype in enumerate(g.etypes):
        src, dst, eid = g._graph.edges(i)
        adj = SparseTensor(row=dst, col=src)
        adjs.append(adj)
        print(g.to_canonical_etype(etype), adj)

    # F --- *P --- A --- I
    # paper : [736389, 128]
    # author: [1134649, 256]
    # institution [8740, 256]
    # field_of_study [59965, 256]

    new_edges = {}
    ntypes = set()

    etypes = [ # src->tgt
        ('A', 'A-I', 'I'),
        ('A', 'A-P', 'P'),
        ('P', 'P-P', 'P'),
        ('P', 'P-F', 'F'),
    ]

    if symmetric:
        adjs[2] = adjs[2].to_symmetric()
        assert torch.all(adjs[2].get_diag() == 0)

    for etype, adj in zip(etypes, adjs):
        stype, rtype, dtype = etype
        dst, src, _ = adj.coo()
        src = src.numpy()
        dst = dst.numpy()
        if stype == dtype:
            new_edges[(stype, rtype, dtype)] = (np.concatenate((src, dst)), np.concatenate((dst, src)))
        else:
            new_edges[(stype, rtype, dtype)] = (src, dst)
            new_edges[(dtype, rtype[::-1], stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)

    new_g = dgl.heterograph(new_edges)
    if paper_year is not None:
        new_g.nodes['P'].data['year'] = paper_year
    new_g.nodes['P'].data['P'] = g.nodes['paper'].data['feat']
    new_g.nodes['A'].data['A'] = g.nodes['author'].data['feat']
    new_g.nodes['I'].data['I'] = g.nodes['institution'].data['feat']
    new_g.nodes['F'].data['F'] = g.nodes['field_of_study'].data['feat']

    if (getattr(args, 'gsmp_first_layer', False)
            or getattr(args, 'smp_first_layer', False)
            or getattr(args, 'gsmp_apply_label_prop', False)):
        setattr(args, '_gsmp_train_paper_nid', train_nid)
        try:
            setattr(new_g, '_gsmp_time_dict', build_mag_time_dict(g, new_g, args))
        finally:
            if hasattr(args, '_gsmp_train_paper_nid'):
                delattr(args, '_gsmp_train_paper_nid')

    if args.use_sparse_tools:
        import sparse_tools
        IA, PA, PP, FP = adjs

        diag_name = f'{args.dataset}_PFP_diag.pt'
        if not os.path.exists(diag_name):
            PF = FP.t()
            PFP_diag = sparse_tools.spspmm_diag_sym_ABA(PF)
            torch.save(PFP_diag, diag_name)

        if symmetric:
            diag_name = f'{args.dataset}_PPP_diag.pt'
            if not os.path.exists(diag_name):
                # PP = PP.to_symmetric()
                # assert torch.all(PP.get_diag() == 0)
                PPP_diag = sparse_tools.spspmm_diag_sym_AAA(PP)
                torch.save(PPP_diag, diag_name)
        else:
            assert False

        diag_name = f'{args.dataset}_PAP_diag.pt'
        if not os.path.exists(diag_name):
            PAP_diag = sparse_tools.spspmm_diag_sym_ABA(PA)
            torch.save(PAP_diag, diag_name)

    return new_g, init_labels, new_g.num_nodes('P'), n_classes, train_nid, val_nid, test_nid, evaluator
