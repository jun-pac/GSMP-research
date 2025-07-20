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
from tqdm import tqdm 
from multiprocessing import Pool
import multiprocessing
from itertools import repeat
sym_chunk_size=1000000
core_num=multiprocessing.cpu_count()
num_processes = multiprocessing.cpu_count()
num_processes-=2

f_log=open('./txtlog/pmp_preprocess.txt','a')


def process_data(start, end, result_queue):
    print(f"Process data... {start} {end}")
    train_mean_part = torch.zeros((218, 172, clone_x.shape[1]))
    train_cnt_part = torch.zeros((218, 172))
    train_time_mean_part = torch.zeros((218, clone_x.shape[1]))
    train_time_cnt_part = torch.zeros(218)
    test_cnt_part = 0
    test_mean_part = torch.zeros((clone_x.shape[1]))

    for u in range(start, end):
        if paper_year[u] == 2019:
            test_cnt_part += 1
            test_mean_part += clone_x[u]
        elif paper_year[u] <= 2017:
            t=max(0,paper_year[u] - 1800)
            if labels[u]<0 or labels[u]>=172:
                    continue
            train_cnt_part[t][labels[u]] += 1
            train_time_cnt_part[t] += 1
            train_mean_part[t][labels[u]] += clone_x[u]

    result_queue.put((train_mean_part, train_cnt_part, train_time_mean_part, train_time_cnt_part, test_cnt_part, test_mean_part))


def add_norm_data(start, end):
    print(f"Add norm data... {start} {end}")
    local_test_var = 0
    local_rsq = torch.zeros(218)
    local_msq = torch.zeros(218)

    for u in range(start, end):
        if paper_year[u] == 2019:
            local_test_var += torch.norm(clone_x[u] - test_mean) ** 2
        elif paper_year[u] <= 2017:
            t = max(0,paper_year[u] - 1800)
            if labels[u]<0 or labels[u]>=172:
                    continue
            local_msq[t] += torch.norm(train_mean[t][labels[u]] - train_time_mean[t]) ** 2
            local_rsq[t] += torch.norm(clone_x[u] - train_mean[t][labels[u]]) ** 2
    
    return local_test_var, local_rsq, local_msq


def update_data(start, end):
    print(f"Update data... {start} {end}")
    for u in range(start, end):
        if paper_year[u] <= 2017:
            t = max(0,paper_year[u] - 1800)
            if labels[u]<0 or labels[u]>=172:
                    continue
            clone_x[u] = alpha[t] * clone_x[u] + (1 - alpha[t]) * train_mean[t][labels[u]]


def sym_task(idx):
    newsrc=[]
    newdst=[]
    print(f"Persistent message passing... {idx}/{len_newsrc}")
    for i in range(idx,min(idx+sym_chunk_size,len_newsrc)):
        newsrc.append(dst[i])
        newdst.append(src[i])
        if abs(paper_year[src[i]]-paper_year[dst[i]])>min(2019-paper_year[dst[i]],max(0,paper_year[dst[i]] - 1800)):
            newsrc.append(src[i])
            newdst.append(dst[i])
        if abs(paper_year[src[i]]-paper_year[dst[i]])>min(2019-paper_year[src[i]],max(0,paper_year[src[i]] - 1800)):
            newsrc.append(dst[i])
            newdst.append(src[i])
    return np.array([newsrc, newdst])



parser = argparse.ArgumentParser()
parser.add_argument('--num_hops', type=int, default=6)
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--pretrained_emb_path', type=str, default=None)
parser.add_argument('--output_emb_prefix', type=str, default='./ogbn-papers100M_node-emb_w2v')
parser.add_argument("--jjnorm", action='store_true', default=False)

args = parser.parse_args()
t0=time.time()
print(args)

dataset = PygNodePropPredDataset('ogbn-papers100M', root=args.root)
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
data = dataset[0]
print(f"dataset load : {time.time()-t0:.4f}")
f_log.write(f"dataset load : {time.time()-t0:.4f}\n")
f_log.flush()

print(f"type(data) : {type(data)}") #  <class 'torch_geometric.data.data.Data'>
print(f"data : {data}") # data : Data(num_nodes=111059956, edge_index=[2, 1615685872], x=[111059956, 128], node_year=[111059956, 1], y=[111059956, 1])

# paper_year=data.node_year()
# labels=data.y()
paper_year=data['node_year']
labels=data['y']
labels=labels.long()
print(f"type(paper_year) : {type(paper_year)}") # type(paper_year) : <class 'torch.Tensor'>
print(f"type(labels) : {type(labels)}") # type(labels) : <class 'torch.Tensor'>
print(f"(paper_year.shape) : {paper_year.shape}") # torch.Size([111059956, 1])
print(f"(labels.shape) : {labels.shape}") # torch.Size([111059956, 1])

paper_year=paper_year.squeeze()
labels=labels.squeeze()
print(f"(paper_year.shape) : {paper_year.shape}") 
print(f"(labels.shape) : {labels.shape}")
print(f"paper_year.dtype : {paper_year.dtype}") 
print(f"labels.dtype : {labels.dtype}")
# print(f"torch.min(paper_year) {torch.min(paper_year)}")

# mn=2020
# for i in range(100000):
#     mn=min(mn,paper_year[i])
# print()
# print(f"Min of year in first 100000 : {mn}")
# f_log.write(f"Min of year in first 100000 : {mn}\n") : 1825
# f_log.flush()

x = None
# if args.pretrained_emb_path=='None':
#     args.pretrained_emb_path=None

print(f"pretrained_emb_path : {args.pretrained_emb_path}")
if args.pretrained_emb_path != None:
    x = np.load(args.pretrained_emb_path)
else:
    x = data.x.numpy()

x=data.x.numpy()
N = data.num_nodes
print("Node Embeddings", x.shape)
f_log.write(f"Node Embeddings : {x.shape}, {type(x)}\n")
f_log.flush()
print("problem here?")

# path = './adj_pmp_gcn.pt'
# if osp.exists(path):
#     adj = torch.load(path)
# else:
print('Making the graph undirected.')
# Randomly drop some edges to save computation
print("problem here??")


# src, dst = data.edge_index
# for i in range(150):
#     print(f"paper year - omuomu src, dst : {paper_year[src[i]]}, {paper_year[dst[i]]}") 
# paper year - omuomu src, dst : 2016, 1986 = src -> dst 

if not (os.path.exists('./graph/pmp_src.npy') and os.path.exists('./graph/pmp_dst.npy')):
    t1=time.time()
    #data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    print("problem here?????")
    #print(f"to_undirected : {time.time()-t0:.4f}")
    print(data) 

    src, dst = data.edge_index
    print(f"src.shape : {src.shape}")
    print(f"dst.shape : {dst.shape}")
    
    newsrc=[]
    newdst=[]
    
    len_newsrc=len(src)
    with Pool(core_num) as p:
        result = p.map(sym_task, range(0,len(src),sym_chunk_size))
    src_dst = np.concatenate(result, axis=1)
    newsrc, newdst = src_dst
    src=np.concatenate((src,newsrc))
    dst=np.concatenate((dst,newdst))

    np.save('./graph/pmp_src',src)
    np.save('./graph/pmp_dst',dst)
    print(f"PMP done! {time.time()-t1:.4f}")
    #new_edges[(stype, 'r'+rtype, dtype)] = (np.concatenate((src, newsrc)), np.concatenate((dst, newdst)))
else:
    src=np.load('./graph/pmp_src.npy')
    dst=np.load('./graph/pmp_dst.npy')

print(f'Computing adj... {time.time()-t0:.4f}')
f_log.write(f"src.dtype, dst.dtype : {src.dtype}, {dst.dtype}\n")
f_log.flush()

src=torch.from_numpy(src).long()
dst=torch.from_numpy(dst).long()
print(f'point 1 {time.time()-t0:.4f}')    

adj = SparseTensor(row=src, col=dst, sparse_sizes=(N, N))
print(f'point 2 {time.time()-t0:.4f}')   
adj = adj.set_diag()
print(f'point 3 {time.time()-t0:.4f}')   
deg = adj.sum(dim=1).to(torch.float)
print(f'point 4 {time.time()-t0:.4f}')   
deg_inv_sqrt = deg.pow(-0.5)
print(f'point 5 {time.time()-t0:.4f}')   
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
print(f'point 6 {time.time()-t0:.4f}')   
adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
print(f'point 7 {time.time()-t0:.4f}')   
# torch.save(adj, path)
# print(f'point 8 {time.time()-t0:.4f}')   


adj = adj.to_scipy(layout='csr')

print('Start processing')

saved = np.concatenate((x[train_idx], x[valid_idx], x[test_idx]), axis=0)
torch.save(torch.from_numpy(saved).to(torch.float), f'{args.output_emb_prefix}_0.pt')

for i in tqdm(range(args.num_hops)):
    x = adj @ x
    if(args.jjnorm):
        t1=time.time()
        print(f"Processing {i+1}th features... {time.time()-t0}", flush=True)
        f_log.write(f"Processing {i+1}th features... {time.time()-t0}\n")
        f_log.flush()

        clone_x=np.copy(x)
        shapes = list(clone_x.shape)
        print(f"shapes : {shapes}\n")
        f_log.write(f"shapes : {shapes}\n")
        f_log.flush()
        
        # Calculte means
        chunk_size = shapes[0] // num_processes
        processes = []
        result_queue = multiprocessing.Queue()
        for i in range(num_processes):
            start = i * chunk_size
            end = start + chunk_size if i < num_processes - 1 else shapes[0]
            p = multiprocessing.Process(target=process_data, args=(start, end, result_queue))
            processes.append(p)
            p.start()
        # Collect results from processes
        train_mean = torch.zeros((218, 172, clone_x.shape[1]))
        train_cnt = torch.zeros((218, 172))
        train_time_mean = torch.zeros((218, clone_x.shape[1]))
        train_time_cnt = torch.zeros(218)
        test_cnt = 0
        test_mean = torch.zeros((clone_x.shape[1]))
        for _ in processes:
            train_mean_part, train_cnt_part, train_time_mean_part, train_time_cnt_part, test_cnt_part, test_mean_part = result_queue.get()
            train_mean += train_mean_part
            train_cnt += train_cnt_part
            train_time_mean += train_time_mean_part
            train_time_cnt += train_time_cnt_part
            test_cnt += test_cnt_part
            test_mean += test_mean_part
        for p in processes:
            p.join()
        result_queue.close()
        result_queue.join_thread()



        # Add norm values
        processes = []
        result_queue = multiprocessing.Queue()
        for i in range(num_processes):
            start = i * chunk_size
            end = start + chunk_size if i < num_processes - 1 else shapes[0]
            p = multiprocessing.Process(target=add_norm_data, args=(start, end))
            processes.append(p)
            p.start()
        test_var = 0
        rsq = torch.zeros(218)
        msq = torch.zeros(218)
        for p in processes:
            p.join()
            local_test_var, local_rsq, local_msq = result_queue.get()
            test_var += local_test_var
            rsq += local_rsq
            msq += local_msq

        
        # Calculate Statistics
        test_var/=max(1,test_cnt-1)
        for t in range(218):
            msq[t]/=max(1,train_time_cnt[t]-1)
            rsq[t]/=max(1,train_time_cnt[t]-1)

        alpha=torch.zeros(218)
        for t in range(218):
            alpha_sq=(test_var-msq[t])/max(0.000001,rsq[t])
            if(alpha_sq>0):
                alpha[t]=torch.sqrt(alpha_sq)
            else:
                alpha[t]=0


        # Update modified vals
        processes = []
        for i in range(num_processes):
            start = i * chunk_size
            end = start + chunk_size if i < num_processes - 1 else shapes[0]
            p = multiprocessing.Process(target=update_data, args=(start, end))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()

        # Save
        saved = np.concatenate((clone_x[train_idx], clone_x[valid_idx], clone_x[test_idx]), axis=0)
        torch.save(torch.from_numpy(saved).to(torch.float), f'{args.output_emb_prefix}_{i+1}.pt')

    else:
        saved = np.concatenate((x[train_idx], x[valid_idx], x[test_idx]), axis=0)
        torch.save(torch.from_numpy(saved).to(torch.float), f'{args.output_emb_prefix}_{i+1}.pt')
