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

    t0=time.time()
    for hop in range(args.label_num_hops):
        print(f"Compute {hop+1}th neighbor-averaged labels... {time.time()-t0:.4f}")
        y = neighbor_average_labels(g, y.to(torch.float), args)
        gc.collect()
    res = y
    # if(args.jjnorm):
    #     print(f"res.shape : {res.shape}")
    #     t0=time.time()

    #     # Calculte means
    #     print(f"Calculate means {time.time()-t0}",flush=True)
    #     train_mean = torch.zeros((193, 172, res.shape[1]))  
    #     train_cnt = torch.zeros((193, 172))
    #     train_time_mean = torch.zeros((193, res.shape[1]))
    #     train_time_cnt = torch.zeros(193)
    #     test_cnt = 0
    #     test_mean = torch.zeros((res.shape[1]))
    #     for u in test_idx:
    #         test_mean+=res[u]
    #         test_cnt+=1
    #     for u in train_idx:
    #         t=paper_year[u] - 1825
    #         train_cnt[t][labels[u]] += 1
    #         train_time_cnt[t] += 1
    #         train_mean[t][labels[u]] += res[u]                 
    #     for t in range(193):
    #         for l in range(172):
    #             train_time_mean[t]+=train_mean[t][l]
    #             train_mean[t][l]/=max(1,train_cnt[t][l])
    #         train_time_mean[t]/=max(1,train_time_cnt[t])
    #     test_mean/=max(1,test_cnt)


    #     # Add norm values
    #     print(f"Add norm values {time.time()-t0}",flush=True)
    #     test_var = 0
    #     rsq = torch.zeros(193)
    #     msq = torch.zeros(193)
    #     for u in test_idx:
    #         test_var += torch.norm(res[u] - test_mean) ** 2
    #     for u in train_idx:
    #         t = paper_year[u] - 1825
    #         msq[t] += torch.norm(train_mean[t][labels[u]] - train_time_mean[t]) ** 2
    #         rsq[t] += torch.norm(res[u] - train_mean[t][labels[u]]) ** 2


    #     # Calculate Statistics
    #     test_var/=max(1,test_cnt-1)
    #     for t in range(193):
    #         msq[t]/=max(1,train_time_cnt[t]-1)
    #         rsq[t]/=max(1,train_time_cnt[t]-1)
    #     alpha=torch.ones(193)
    #     for t in range(193):
    #         alpha_sq=(test_var-msq[t])/max(0.000001,rsq[t])
    #         if(alpha_sq>0):
    #             alpha[t]=torch.sqrt(alpha_sq)
    #         else:
    #             alpha[t]=0
    #         print(f"Time : {t+1825} | Number : {train_time_cnt[t]} | rsq : {rsq[t]:.4f} | msq : {msq[t]:.4f} | alpha_sq : {alpha_sq:.4f}")
    #     print(f"Test var : {test_var}")

    #     # Update modified vals
    #     print(f"Modify train feats {time.time()-t0}",flush=True)
    #     for u in train_idx:
    #         t = paper_year[u] - 1825
    #         res[u] = alpha[t] * res[u] + (1 - alpha[t]) * train_mean[t][labels[u]]


    return torch.cat([res[train_idx], res[valid_idx], res[test_idx]], dim=0)


def neighbor_average_labels(g, feat, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    g.ndata["f"] = feat
    g.update_all(fn.copy_u("f", "msg"),
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
    dataset = dataset.lower()
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
    src=np.load('./graph/pmp_src.npy')
    dst=np.load('./graph/pmp_dst.npy')    
    # new_g = dgl.graph((src,dst),num_nodes=g.num_nodes())
    new_g = dgl.graph((src,dst),num_nodes=111059956)
    print(f"New graph generated! : {time.time()-t0:.4f}")

    gc.collect()
    label_emb = None
    if args.use_rlu:
        label_emb = prepare_label_emb(args, new_g, labels, n_classes, train_nid, val_nid, test_nid, paper_year, teacher_probs)
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
