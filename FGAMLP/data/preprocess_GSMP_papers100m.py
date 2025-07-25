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

core_num=multiprocessing.cpu_count()



def compute_edge_weights_chunk(args):
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
    import numpy as np
    import multiprocessing
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






# begin 4:22 AM

f_log=open('./txtlog/gsmp_preprocess.txt','a')

parser = argparse.ArgumentParser()
parser.add_argument('--num_hops', type=int, default=6)
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--pretrained_emb_path', type=str, default=None)
parser.add_argument('--output_emb_prefix', type=str, default='./ogbn-papers100M-gsmp.node-emb')
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
# 4:35 AM (~13 min)

print('Computing adj...')

if not os.path.isfile("./gsmp_edge_weight.pt"):
    edge_weight=torch.zeros(len(row))
    print(f"len edge_weight: {edge_weight.shape}")
    print(f"edge_weight.dtype: {edge_weight.dtype}")

    nums=torch.load("./num_year.pt")
    # delta_matrix=[[0]*2024 for _ in range(2024)]
    # for i in range(2024):
    #     for j in range(2024):
    #         delta_matrix[i][abs(j-i)]+=nums[j]

    test_time=2019
    # 4:37 AM begin (minimum 270G required)
    parallel_compute(edge_weight, paper_year, row, col, N, num_workers=48)
    # 5:23 AM finish

    torch.save(edge_weight,"./gsmp_edge_weight.pt")
else:
    edge_weight=torch.load("./gsmp_edge_weight.pt")


adj = SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=(N, N))
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

edge_weight=torch.load("gsmp_edge_weight.pt")


for i in tqdm(range(args.num_hops)):
    t0=time.time()
    x = adj @ x
    saved = np.concatenate((x[train_idx], x[valid_idx], x[test_idx]), axis=0)
    torch.save(torch.from_numpy(saved).to(torch.float), f'{args.output_emb_prefix}_{i+1}.pt')
    print(f"{i}th propagation finished... {time.time()-t0}")
