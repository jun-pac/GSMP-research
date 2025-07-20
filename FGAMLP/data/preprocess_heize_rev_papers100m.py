import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, dropout_adj
import os
import os.path as osp
import time
from ogb.nodeproppred import PygNodePropPredDataset
from multiprocessing import Pool, Array, Manager
import multiprocessing




def compute_edge_weights(start_idx, end_idx):
    partial_edge_weight = torch.zeros(end_idx - start_idx)

    for idx in range(start_idx, end_idx):
        ys = min(2023, paper_year[row[idx]])
        yd = min(2023, paper_year[col[idx]])
        
        partial_edge_weight[idx - start_idx] = (
            delta_matrix[test_time][abs(ys - yd)] / 
            max(delta_matrix[ys][abs(ys - yd)], 1)
        )
    
    return partial_edge_weight



def parallel_compute(edge_weight, num_workers=48):
    pool = multiprocessing.Pool(processes=num_workers)
    chunk_size = len(row) // num_workers
    jobs = []

    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = len(row) if i == num_workers - 1 else (i + 1) * chunk_size 
        jobs.append(pool.apply_async(compute_edge_weights, (start_idx, end_idx)))

    for i, job in enumerate(jobs):
        partial_edge_weight = job.get()

        if i == num_workers - 1:
            edge_weight[i * chunk_size:] = partial_edge_weight
        else:
            edge_weight[i * chunk_size: (i + 1) * chunk_size] = partial_edge_weight

    pool.close()
    pool.join()




f_log=open('./txtlog/heize_rev_preprocess.txt','a')

parser = argparse.ArgumentParser()
parser.add_argument('--num_hops', type=int, default=6)
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--pretrained_emb_path', type=str, default=None)
parser.add_argument('--output_emb_prefix', type=str, default='./ogbn-papers100M-heize_rev.node-emb')
args = parser.parse_args()
print(args)

dataset = PygNodePropPredDataset('ogbn-papers100M', root=args.root)
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
data = dataset[0]
paper_year=data['node_year']
labels=data['y']
labels=labels.long()
paper_year=paper_year.squeeze()
labels=labels.squeeze()

x = None

print(f"pretrained_emb_path : {args.pretrained_emb_path}")

if len(args.pretrained_emb_path)>4:
    print(f"XRT feature address: {args.pretrained_emb_path}")
    x = np.load(args.pretrained_emb_path)
else:
    x = data.x.numpy()
N = data.num_nodes
print("Node Embeddings", x.shape)


print('Making the graph undirected.')
# Randomly drop some edges to save computation
data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

print(data)

row, col = data.edge_index

print('Computing adj...')

if not os.path.isfile("./heize_edge_weight_rev.pt"):
    edge_weight=torch.zeros(len(row))
    print(f"len edge_weight: {edge_weight.shape}")
    print(f"edge_weight.dtype: {edge_weight.dtype}")

    nums=torch.load("./num_year.pt")
    delta_matrix=[[0]*2024 for _ in range(2024)]
    for i in range(2024):
        for j in range(2024):
            delta_matrix[i][abs(j-i)]+=nums[j]

    test_time=2019
    # 3:01 AM begin (minimum 270G required)
    # 4:50 AM begin
    parallel_compute(edge_weight, num_workers=48)
    # 4:21 AM finish

    torch.save(edge_weight,"./heize_edge_weight_rev.pt")
else:
    edge_weight=torch.load("./heize_edge_weight_rev.pt")



adj = SparseTensor(row=row, col=col, value=edge_weight ,sparse_sizes=(N, N))
adj = adj.set_diag()
deg = adj.sum(dim=1).to(torch.float)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
# torch.save(adj, path)


adj = adj.to_scipy(layout='csr')

print('Start processing')

saved = np.concatenate((x[train_idx], x[valid_idx], x[test_idx]), axis=0)
torch.save(torch.from_numpy(saved).to(torch.float), f'{args.output_emb_prefix}_0.pt')


for i in range(args.num_hops):
    t0=time.time()
    x = adj @ x
    saved = np.concatenate((x[train_idx], x[valid_idx], x[test_idx]), axis=0)
    torch.save(torch.from_numpy(saved).to(torch.float), f'{args.output_emb_prefix}_{i+1}.pt')
    print(f"{i}th propagation finished... {time.time()-t0}")
