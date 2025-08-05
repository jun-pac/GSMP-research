#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import os
import random
import time
import logging
import uuid
import sys
import gc
from functools import reduce
import operator as op
import multiprocessing

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from examples.ogb_eff.ogbn_arxiv_dgl.loss import loss_kd_only
from examples.ogb_eff.ogbn_arxiv_dgl.model_weighted_rev import RevGAT
epsilon = 1 - math.log(2)

device = None

dataset = "ogbn-arxiv"
n_node_feats, n_classes = 0, 0


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def compute_edge_weights_chunk(args):
    """Helper function for parallel edge weight computation"""
    paper_year, row, col, src_start, src_end = args
    edge_weights = np.zeros(np.sum((row >= src_start) & (row < src_end)), dtype=np.float32)
    offset = 0
    for src in range(src_start, src_end):
        mask = (row == src)
        neighbors = col[mask]
        if len(neighbors) == 0:
            continue
        neighbor_years = paper_year[neighbors]
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


def parallel_compute(edge_weight, paper_year, row, col, num_nodes, num_workers=48):
    """
    Parallel computation of edge weights, exactly matching preprocess_GSMP_papers100m.py
    """
    chunk_size = (num_nodes + num_workers - 1) // num_workers
    args_list = []
    for i in range(num_workers):
        src_start = i * chunk_size
        src_end = min((i + 1) * chunk_size, num_nodes)
        if src_start >= src_end:
            continue
        args_list.append((paper_year, row, col, src_start, src_end))
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(compute_edge_weights_chunk, args_list)
    edge_weights = np.concatenate(results)
    edge_weight[:] = torch.from_numpy(edge_weights)


def compute_local_edge_weights_parallel(graph, paper_year, num_workers=None):
    """
    Compute edge weights based on local time distribution using parallel processing.
    Exactly matching the approach in preprocess_GSMP_papers100m.py
    """
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 48)
    
    src, dst = graph.edges()
    row = src.cpu().numpy()
    col = dst.cpu().numpy()
    paper_year = paper_year.cpu().numpy()
    
    num_nodes = graph.number_of_nodes()
    edge_weight = torch.zeros(len(row))
    
    parallel_compute(edge_weight, paper_year, row, col, num_nodes, num_workers)
    
    return edge_weight.float()


def compute_local_edge_weights(graph, paper_year):
    """
    Compute edge weights based on local time distribution of neighbors.
    Sequential version for debugging and small graphs.
    """
    src, dst = graph.edges()
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()
    paper_year = paper_year.cpu().numpy()
    
    edge_weights = np.zeros(len(src), dtype=np.float32)
    
    # Group edges by source node
    unique_src, src_indices = np.unique(src, return_index=True)
    
    for i, src_node in enumerate(unique_src):
        # Find all edges from this source node
        if i < len(unique_src) - 1:
            edge_start = src_indices[i]
            edge_end = src_indices[i + 1]
        else:
            edge_start = src_indices[i]
            edge_end = len(src)
        
        # Get all neighbors of this source node
        neighbors = dst[edge_start:edge_end]
        if len(neighbors) == 0:
            continue
            
        # Get years of all neighbors
        neighbor_years = paper_year[neighbors]
        
        # Count frequency of each year in the neighborhood
        year_counts = np.bincount(neighbor_years, minlength=2024)
        
        # Compute local weights for each edge
        local_weights = np.zeros(len(neighbors), dtype=np.float32)
        for idx, dst_node in enumerate(neighbors):
            year = paper_year[dst_node]
            # Weight is inversely proportional to the frequency of that year in the neighborhood
            local_weights[idx] = 1.0 / (year_counts[year] if year_counts[year] > 0 else 1)
        
        # Normalize weights by their mean
        mean_val = np.mean(local_weights)
        if mean_val > 0:
            local_weights = local_weights / mean_val
            
        edge_weights[edge_start:edge_end] = local_weights
    
    return torch.from_numpy(edge_weights).float()


def load_data(dataset,args):
    global n_node_feats, n_classes

    if args.data_root_dir == 'default':
        data = DglNodePropPredDataset(name=dataset)
    else:
        data = DglNodePropPredDataset(name=dataset,root=args.data_root_dir)

    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    paper_year=torch.squeeze(graph.ndata['year'])

    # Count frequency
    cnt=[0]*(2020-1971+1)
    for i in range(len(paper_year)):
        cnt[paper_year[i]-1971]+=1
    sum=0
    for i in range(2020-1971+1):
        sum+=cnt[i]

    # Count abstract difference matrix
    delta_matrix=[[0]*(2020-1971+1) for _ in range(2020-1971+1)]
    for i in range(2020-1971+1):
        for j in range(2020-1971+1):
            delta_matrix[i][abs(j-i)]+=cnt[j]
        
        # print(f"current year: {i+1971}")
        # for j in range(2020-1971+1):
        #     print(f"{delta_matrix[i][j]/sum:.4f} ",end='')
        
    
    

    if args.pretrain_path != 'None':
        graph.ndata["feat"] = torch.tensor(np.load(args.pretrain_path)).float()
        print("Pretrained node feature loaded! Path: {}".format(args.pretrain_path))        
    
    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()
    return graph, labels, train_idx, val_idx, test_idx, evaluator, paper_year, delta_matrix


def heize_preprocess(graph, paper_year, delta_matrix, use_parallel=True, num_workers=None, cache_file="./gsmp_edge_weight.pt"):
    global n_node_feats

    # make bidirected
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
        
    test_time=2019-1971
    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}") # 2864098
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}") # 3033441

    # Use local time distribution approach instead of global delta_matrix
    print("Computing edge weights using local time distribution...")
    start_time = time.time()
    
    # Check if cached edge weights exist
    if os.path.isfile(cache_file):
        print(f"Loading cached edge weights from {cache_file}")
        eweight = torch.load(cache_file)
        print(f"Loaded cached edge weights with shape: {eweight.shape}")
    else:
        print("Computing edge weights from scratch...")
        if use_parallel:
            print(f"Using parallel computation with {num_workers or 'auto'} workers...")
            eweight = compute_local_edge_weights_parallel(graph, paper_year, num_workers)
        else:
            print("Using sequential computation...")
            eweight = compute_local_edge_weights(graph, paper_year)
        
        # Save computed edge weights for future use
        print(f"Saving edge weights to {cache_file}")
        torch.save(eweight, cache_file)
    
    computation_time = time.time() - start_time
    print(f"Edge weight computation completed in {computation_time:.2f} seconds")
    print(f'eweight.dtype: {eweight.dtype}')
    print(f'eweight.shape: {eweight.shape}')
    print(f'eweight stats - min: {eweight.min():.4f}, max: {eweight.max():.4f}, mean: {eweight.mean():.4f}')

    src, dst = graph.edges()
    new_g = dgl.graph((src, dst))
    new_g.ndata['feat'] = feat
    new_g.edata['w'] = eweight
    print(f"new_g.edges()[0].shape, new_g.edges()[1].shape : {new_g.edges()[0].shape}, {new_g.edges()[1].shape}")
    print(f"new_g.edata['w'].shape: {new_g.edata['w'].shape}")

    new_g.create_formats_()

    return new_g


def gen_model(args):
    if args.use_labels:
        n_node_feats_ = n_node_feats + n_classes
    else:
        n_node_feats_ = n_node_feats

    if args.backbone == "rev":
        model = RevGAT(
                      n_node_feats_,
                      n_classes,
                      n_hidden=args.n_hidden,
                      n_layers=args.n_layers,
                      n_heads=args.n_heads,
                      activation=F.relu,
                      dropout=args.dropout,
                      input_drop=args.input_drop,
                      attn_drop=args.attn_drop,
                      edge_drop=args.edge_drop,
                      use_attn_dst=not args.no_attn_dst,
                      use_symmetric_norm=args.use_norm)
    else:
        raise Exception("Unknown backnone")

    return model


def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def add_labels(feat, labels, idx):
    onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(args, model, graph, labels, train_idx, val_idx, test_idx, optimizer,
          evaluator, mode='teacher', teacher_output=None):
    model.train()
    if mode == 'student':
        assert teacher_output != None

    alpha = args.alpha
    temp = args.temp

    feat = graph.ndata["feat"]

    if args.use_labels:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()

    if args.n_label_iters > 0:
        with torch.no_grad():
            pred = model(graph, feat)
    else:
        pred = model(graph, feat)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([train_pred_idx, val_idx, test_idx])
        for _ in range(args.n_label_iters):
            pred = pred.detach()
            torch.cuda.empty_cache()
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(graph, feat)

    if mode == 'teacher':
        loss = custom_loss_function(pred[train_pred_idx],
                                    labels[train_pred_idx])
    elif mode == 'student':
        loss_gt = custom_loss_function(pred[train_pred_idx],
                                       labels[train_pred_idx])
        loss_kd = loss_kd_only(pred, teacher_output, temp)
        loss = loss_gt * (1 - alpha) + loss_kd * alpha
    else:
        raise Exception('unkown mode')

    loss.backward()
    optimizer.step()

    return evaluator(pred[train_idx], labels[train_idx]), loss.item()


@torch.no_grad()
def evaluate(args, model, graph, labels, train_idx, val_idx, test_idx, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    if args.use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([val_idx, test_idx])
        for _ in range(args.n_label_iters):
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(graph, feat)

    train_loss = custom_loss_function(pred[train_idx], labels[train_idx])
    val_loss = custom_loss_function(pred[val_idx], labels[val_idx])
    test_loss = custom_loss_function(pred[test_idx], labels[test_idx])

    return (
        evaluator(pred[train_idx], labels[train_idx]),
        evaluator(pred[val_idx], labels[val_idx]),
        evaluator(pred[test_idx], labels[test_idx]),
        train_loss,
        val_loss,
        test_loss,
        pred,
    )


def save_pred(pred, run_num, kd_dir):
    if not os.path.exists(kd_dir):
        os.makedirs(kd_dir)
    fname = os.path.join(kd_dir, 'best_pred_run{}.pt'.format(run_num))
    torch.save(pred.cpu(), fname)


def run(args, graph, labels, train_idx, val_idx, test_idx,
        evaluator, n_running):
    evaluator_wrapper = lambda pred, labels: evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]

    # kd mode
    mode = args.mode

    # define model and optimizer
    model = gen_model(args).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training loop
    total_time = 0
    best_val_acc, final_test_acc, best_val_loss = 0, 0, float("inf")
    final_pred = None

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()
        if mode == 'student':
            teacher_output = torch.load('./{}/best_pred_run{}.pt'.format(
              args.kd_dir,
              n_running)).cpu().cuda()
        else:
            teacher_output = None

        adjust_learning_rate(optimizer, args.lr, epoch)

        acc, loss = train(args, model, graph, labels, train_idx,
                          val_idx, test_idx, optimizer, evaluator_wrapper,
                          mode=mode, teacher_output=teacher_output)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = evaluate(
            args, model, graph, labels, train_idx, val_idx, test_idx, evaluator_wrapper
        )

        toc = time.time()
        total_time += toc - tic

        if epoch == 1:
            peak_memuse = torch.cuda.max_memory_allocated(device) / float(1024 ** 3)
            logging.info('Peak memuse {:.2f} G'.format(peak_memuse))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            final_test_acc = test_acc
            final_pred = pred
            if mode == 'teacher':
                save_pred(final_pred, n_running, args.kd_dir)

        if epoch == args.n_epochs or epoch % args.log_every == 0:
            logging.info(
                f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
                f"Loss: {loss:.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
            )

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
            [acc, train_acc, val_acc, test_acc, loss, train_loss, val_loss, test_loss],
        ):
            l.append(e)

    logging.info("*" * 50)
    logging.info(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
    logging.info("*" * 50)

    # plot learning curves
    if args.plot_curves:
        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.set_yticks(np.linspace(0, 1.0, 101))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip([accs, train_accs, val_accs, test_accs], ["acc", "train acc", "val acc", "test acc"]):
            plt.plot(range(args.n_epochs), y, label=label, linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gat_acc_{n_running}.png")

        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip(
            [losses, train_losses, val_losses, test_losses], ["loss", "train loss", "val loss", "test loss"]
        ):
            plt.plot(range(args.n_epochs), y, label=label, linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gat_loss_{n_running}.png")

    if args.save_pred:
        os.makedirs(f"./output/{args.outputname}", exist_ok=True)
        torch.save(F.softmax(final_pred, dim=1), f"./output/{args.outputname}/{n_running}.pt")

    return best_val_acc, final_test_acc


def count_parameters(args):
    model = gen_model(args)
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def main():
    global device, n_node_feats, n_classes, epsilon

    argparser = argparse.ArgumentParser(
        "GAT implementation on ogbn-arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument("--n-runs", type=int, default=1, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=2000, help="number of epochs")
    argparser.add_argument(
        "--use-labels", action="store_true", help="Use labels in the training set as input features."
    )
    argparser.add_argument("--n-label-iters", type=int, default=0, help="number of label iterations")
    argparser.add_argument("--mask-rate", type=float, default=0.5, help="mask rate")
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--use-norm", action="store_true", help="Use symmetrically normalized adjacency matrix.")
    argparser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=3, help="number of layers")
    argparser.add_argument("--n-heads", type=int, default=3, help="number of heads")
    argparser.add_argument("--n-hidden", type=int, default=250, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.75, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.1, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention drop rate")
    argparser.add_argument("--edge-drop", type=float, default=0.0, help="edge drop rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--log-every", type=int, default=20, help="log every LOG_EVERY epochs")
    argparser.add_argument("--plot-curves", action="store_true", help="plot learning curves")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")
    argparser.add_argument("--save", type=str, default='exp', help="save exp")
    argparser.add_argument('--backbone', type=str, default='rev',
                           help='gcn backbone [deepergcn, wt, deq, rev, gr]')
    argparser.add_argument('--group', type=int, default=2,
                           help='num of groups for rev gnns')
    argparser.add_argument("--kd_dir", type=str, default='./kd_heize', help="kd path for pred")
    argparser.add_argument("--mode", type=str, default='teacher', help="kd mode [teacher, student]")
    argparser.add_argument("--alpha",type=float, default=0.5, help="ratio of kd loss")
    argparser.add_argument("--temp",type=float, default=1.0, help="temperature of kd")
    argparser.add_argument('--data_root_dir', type=str, default='default', help="dir_path for saving graph data. Note that this model use DGL loader so do not mix up with the dir_path for the Pyg one. Use 'default' to save datasets at current folder.")
    argparser.add_argument("--pretrain_path", type=str, default='None', help="path for pretrained node features")
    argparser.add_argument("--outputname", type=str, default='', help="path for pretrained node features")
    argparser.add_argument("--use-parallel-edge-weights", action="store_true", default=True, help="Use parallel processing for edge weight computation")
    argparser.add_argument("--num-workers", type=int, default=None, help="Number of workers for parallel edge weight computation")
    argparser.add_argument("--edge-weight-cache", type=str, default='./gsmp_edge_weight.pt', help="Cache file for edge weights")

    args = argparser.parse_args()
    
    # Adjust kd_dir here
    args.kd_dir = '{}/-L{}-H{}-Ptrn_{}'.format(args.kd_dir, args.n_layers, args.n_hidden, not args.pretrain_path=='None')
    
    args.save = '{}/-L{}-H{}-Ptrn_{}'.format(args.kd_dir, args.n_layers, args.n_hidden, not args.pretrain_path=='None')
    args.save = 'log/{}-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"), str(uuid.uuid4()))
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt='%m/%d %I:%M:%S %p')
    logging.getLogger().setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not args.use_labels and args.n_label_iters > 0:
        raise ValueError("'--use-labels' must be enabled when n_label_iters > 0")

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # load data & preprocess
    graph, labels, train_idx, val_idx, test_idx, evaluator, paper_year, delta_matrix = load_data(dataset,args)
    
    graph = heize_preprocess(graph, paper_year, delta_matrix, 
                           use_parallel=args.use_parallel_edge_weights, 
                           num_workers=args.num_workers,
                           cache_file=args.edge_weight_cache)

    graph, labels, train_idx, val_idx, test_idx = map(
        lambda x: x.to(device), (graph, labels, train_idx, val_idx, test_idx)
    )


    logging.info(args)
    logging.info(f"Number of params: {count_parameters(args)}")

    # run
    val_accs, test_accs = [], []

    for i in range(args.n_runs):
        seed(args.seed + i+2)
        val_acc, test_acc = run(args, graph, labels, train_idx, val_idx,
                                test_idx, evaluator, i + 1)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    logging.info(args)
    logging.info(f"Runned {args.n_runs} times")
    logging.info("Val Accs:")
    logging.info(val_accs)
    logging.info("Test Accs:")
    logging.info(test_accs)
    logging.info(f"Average val accuracy: {100*np.mean(val_accs):.2f} ± {100*np.std(val_accs):.2f}")
    logging.info(f"Average test accuracy: {100*np.mean(test_accs):.2f} ± {100*np.std(test_accs):.2f}")
    logging.info(f"Number of params: {count_parameters(args)}")

if __name__ == "__main__":
    main()


# conda activate deepgcn