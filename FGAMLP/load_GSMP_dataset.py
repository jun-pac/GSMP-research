import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from heteo_data import load_data, read_relation_subsets, gen_rel_subset_feature, preprocess_features
import torch.nn.functional as F
import gc
import os
from multiprocessing import Pool, Array, Manager
import multiprocessing
import resource

test_time=2019
nums=torch.load("./num_year.pt")


def increase_file_limit(new_limit=4096):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))



def compute_local_edge_weights_chunk(args):
    g, paper_year, src_start, src_end = args
    import numpy as np
    adj = g.adj(scipy_fmt='csr')
    edge_weights = np.zeros(adj.indptr[src_end] - adj.indptr[src_start], dtype=np.float32)
    offset = 0
    for src in range(src_start, src_end):
        start = adj.indptr[src]
        end = adj.indptr[src + 1]
        neighbors = adj.indices[start:end]
        if len(neighbors) == 0:
            continue
        neighbor_years = paper_year[neighbors].cpu().numpy()
        year_counts = np.bincount(neighbor_years, minlength=2024)
        local_weights = np.zeros(len(neighbors), dtype=np.float32)
        for idx, dst in enumerate(neighbors):
            year = paper_year[dst].item()
            local_weights[idx] = 1.0 / (year_counts[year] if year_counts[year] > 0 else 1)
        mean_val = np.mean(local_weights)
        if mean_val > 0:
            local_weights = local_weights / mean_val
        edge_weights[offset:offset+len(neighbors)] = local_weights
        offset += len(neighbors)
    return edge_weights

def compute_local_edge_weights_parallel(g, paper_year, num_workers=48):
    import numpy as np
    import multiprocessing
    adj = g.adj(scipy_fmt='csr')
    num_nodes = g.number_of_nodes()
    chunk_size = (num_nodes + num_workers - 1) // num_workers
    args_list = []
    for i in range(num_workers):
        src_start = i * chunk_size
        src_end = min((i + 1) * chunk_size, num_nodes)
        if src_start >= src_end:
            continue
        args_list.append((g, paper_year, src_start, src_end))
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(compute_local_edge_weights_chunk, args_list)
    edge_weights = np.concatenate(results)
    return torch.tensor(edge_weights, dtype=torch.float32)


def prepare_label_emb(args, g, labels, n_classes, train_idx, valid_idx, test_idx, paper_year, label_teacher_emb=None):
    print(n_classes) # 172
    
    paper_year=paper_year.squeeze()
    print(f"paper_year.dtype ; {paper_year.dtype}") # torch.int64
    print(f"paper_year.shape ; {paper_year.shape}") # paper_year.shape ; torch.Size([111059956])

    labels=labels.long()
    print(f"labels.dtype ; {labels.dtype}") # torch.int64
    print(f"labels.shape ; {labels.shape}")


    if label_teacher_emb == None:
        y = np.zeros(shape=(labels.shape[0], int(n_classes)))
        y[train_idx] = F.one_hot(labels[train_idx].to(
            torch.long), num_classes=n_classes).float().squeeze(1)
        y = torch.Tensor(y)
    else:
        print("use teacher label")
        y = np.zeros(shape=(labels.shape[0], int(n_classes)))
        y[valid_idx] = label_teacher_emb[len(
            train_idx):len(train_idx)+len(valid_idx)]
        y[test_idx] = label_teacher_emb[len(
            train_idx)+len(valid_idx):len(train_idx)+len(valid_idx)+len(test_idx)]
        y[train_idx] = F.one_hot(labels[train_idx].to(
            torch.long), num_classes=n_classes).float().squeeze(1)
        y = torch.Tensor(y)


    
    if not os.path.isfile("./dgl_gsmp_edge_weight.pt"):
        increase_file_limit(4096)
        print("Computing local edge weights in parallel...")
        edge_weight = compute_local_edge_weights_parallel(g, paper_year, num_workers=48)
        print(f"len edge_weight: {edge_weight.shape}")
        print(f"edge_weight.dtype: {edge_weight.dtype}")
        torch.save(edge_weight,"./dgl_gsmp_edge_weight.pt")
    else:
        edge_weight=torch.load("./dgl_gsmp_edge_weight.pt")
    
    
    t0=time.time()
    for hop in range(args.label_num_hops):
        print(f"Compute {hop+1}th neighbor-averaged labels... {time.time()-t0:.4f}")
        y = gsmp_neighbor_average_labels(g, y.to(torch.float), edge_weight, args)
        # y = neighbor_average_labels(g, y.to(torch.float), args)
        gc.collect()
    res = y
    return torch.cat([res[train_idx], res[valid_idx], res[test_idx]], dim=0)


def gsmp_neighbor_average_labels(g, feat, edge_weight, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    g.ndata["f"] = feat
    g.edata["w"] = edge_weight
    g.update_all(fn.u_mul_e("f","w","msg"),
                 fn.mean("msg", "f"))
    feat = g.ndata.pop('f')

    return feat


def neighbor_average_labels(g, feat, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    g.ndata["f"] = feat
    g.update_all(fn.copy_u("f","msg"),
                 fn.mean("msg", "f"))
    feat = g.ndata.pop('f')

    return feat


def neighbor_average_features(g, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    g.ndata["feat_0"] = g.ndata["feat"]
    for hop in range(1, args.num_hops + 1):
        g.update_all(fn.copy_u(f"feat_{hop-1}", "msg"),
                     fn.mean("msg", f"feat_{hop}"))
    res = []
    for hop in range(args.num_hops + 1):
        res.append(g.ndata.pop(f"feat_{hop}"))
    return res

def batched_acc(labels,pred):
    # testing accuracy for single label multi-class prediction
    return (torch.argmax(pred, dim=1) == labels,)

def get_evaluator(dataset):
    # testing accuracy for single label multi-class prediction
    return batched_acc

def get_ogb_evaluator(dataset):
    """
    Get evaluator from Open Graph Benchmark based on dataset
    """
#    if dataset=='ogbn-mag':
#        return batched_acc
#    else:
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        })["acc"]


def load_dataset(name, device, args):
    print(f"name : {name}")
    if name!="ogbn-papers100M":
        raise RuntimeError("Dataset {} is not supported".format(name))
    dataset = DglNodePropPredDataset(name=name, root=args.root)
    splitted_idx = dataset.get_idx_split()

    train_nid = splitted_idx["train"]
    val_nid = splitted_idx["valid"]
    test_nid = splitted_idx["test"]
    g, labels = dataset[0]
    n_classes = dataset.num_classes
    labels = labels.squeeze()
    evaluator = get_ogb_evaluator(name)

    print(f"# Nodes: {g.number_of_nodes()}\n"
          f"# Edges: {g.number_of_edges()}\n"
          f"# Train: {len(train_nid)}\n"
          f"# Val: {len(val_nid)}\n"
          f"# Test: {len(test_nid)}\n"
          f"# Classes: {n_classes}\n")

    return g, labels, n_classes, train_nid, val_nid, test_nid, evaluator


def prepare_data(device, args, teacher_probs):
    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """
    data = load_dataset(args.dataset, device, args)

    g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
    print(g)
    print(g.ndata)
    paper_year=g.ndata['year']
    print(f"paper_year.shape : {paper_year.shape}")
    print(f"paper_year.dtype : {paper_year.dtype}")

    print(f"g.num_nodes : {g.num_nodes}")
    t0=time.time()
    g = dgl.add_reverse_edges(g, copy_ndata=True)
    feat=g.ndata.pop('feat')
    
    gc.collect()
    label_emb = None
    if args.use_rlu:
        label_emb = prepare_label_emb(args, g, labels, n_classes, train_nid, val_nid, test_nid, paper_year, teacher_probs)
    # move to device
    assert(args.dataset=='ogbn-papers100M')
    if args.node_emb_path is None:
        raise ValueError(f"for ogbn-papers100M, args.node_emb_path CAN NOT be None!")
    feats=[]
    for i in range(args.num_hops+1):
        feats.append(torch.load(f"{args.node_emb_path}_{i}.pt"))
    in_feats=feats[0].shape[1]
    
    train_nid = train_nid.to(device)
    val_nid = val_nid.to(device)
    test_nid = test_nid.to(device)
    labels = labels.to(device).to(torch.long)
    return feats, torch.cat([labels[train_nid], labels[val_nid], labels[test_nid]]), in_feats, n_classes, \
        train_nid, val_nid, test_nid, evaluator, label_emb
