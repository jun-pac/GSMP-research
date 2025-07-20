import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, dropout_adj
import os
import os.path as osp
import gc
import time
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm 
from multiprocessing import Pool, Array, Manager
import multiprocessing
from itertools import repeat
chunk_size=1000000
core_num=multiprocessing.cpu_count()

f_log=open('./txtlog/pmp_preprocess.txt','a')

def sym_task(idx):
    newsrc=[]
    newdst=[]
    print(f"Persistent message passing... {idx}/{len_src}")
    for i in range(idx,min(idx+chunk_size,len_src)):
        newsrc.append(dst[i])
        newdst.append(src[i])
        if abs(paper_year[src[i]]-paper_year[dst[i]])>min(2020-paper_year[dst[i]],max(0,paper_year[dst[i]] - 1825)):
            newsrc.append(src[i])
            newdst.append(dst[i])
        if abs(paper_year[src[i]]-paper_year[dst[i]])>min(2020-paper_year[src[i]],max(0,paper_year[src[i]] - 1825)):
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
paper_year=data['node_year']
labels=data['y']
labels=labels.long()
paper_year=paper_year.squeeze()
labels=labels.squeeze()





x = None
# if args.pretrained_emb_path=='None':
#     args.pretrained_emb_path=None

# if args.pretrained_emb_path != None:
#     x = np.load(args.pretrained_emb_path)
# else:
#     x = data.x.numpy()

x=data.x.numpy()
N = data.num_nodes
print("Node Embeddings", x.shape)
f_log.write(f"Node Embeddings : {x.shape}, {type(x)}\n")
f_log.flush()


if not (os.path.exists('./graph/pmp_src.npy') and os.path.exists('./graph/pmp_dst.npy')):
    print('Building adjusted graph... 1.5hr')
    t1=time.time()
    print(data) 

    src, dst = data.edge_index
    print(f"src.shape : {src.shape}")
    print(f"dst.shape : {dst.shape}")
    
    len_src=len(src)
    with Pool(core_num) as p:
        result = p.map(sym_task, range(0,len(src),chunk_size))
    src_dst = np.concatenate(result, axis=1)
    newsrc, newdst = src_dst
    src=np.concatenate((src,newsrc))
    dst=np.concatenate((dst,newdst))

    np.save('./graph/pmp_src',src)
    np.save('./graph/pmp_dst',dst)
    print(f"PMP done! {time.time()-t1:.4f}")
    #new_edges[(stype, 'r'+rtype, dtype)] = (np.concatenate((src, newsrc)), np.concatenate((dst, newdst)))
else:
    print('Load adjusted graph... 10min')
    src=np.load('./graph/pmp_src.npy')
    dst=np.load('./graph/pmp_dst.npy')

print(f'Computing adj... 10min')
f_log.write(f"src.dtype, dst.dtype : {src.dtype}, {dst.dtype}\n")
f_log.flush()

src=torch.from_numpy(src).long()
dst=torch.from_numpy(dst).long()
adj = SparseTensor(row=src, col=dst, sparse_sizes=(N, N))
adj = adj.set_diag()
deg = adj.sum(dim=1).to(torch.float)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
adj = adj.to_scipy(layout='csr')

print('Start processing')

if not os.path.exists(f'{args.output_emb_prefix}_0.pt'):
    saved = np.concatenate((x[train_idx], x[valid_idx], x[test_idx]), axis=0)
    torch.save(torch.from_numpy(saved).to(torch.float), f'{args.output_emb_prefix}_0.pt')

for i in tqdm(range(args.num_hops)):
    x = adj @ x
    if(args.jjnorm):
        t1=time.time()
        print(f"Processing {i+1}th features... {time.time()-t0}", flush=True)

        clone_x=np.copy(x)
        shapes = list(clone_x.shape)
        print(f"shapes : {shapes}\n")


        # Calculte means
        print(f"Calculate means {time.time()-t0}",flush=True)
        train_mean = np.zeros((193, 172, clone_x.shape[1]))
        train_cnt = np.zeros((193, 172))
        train_time_mean = np.zeros((193, clone_x.shape[1]))
        train_time_cnt = np.zeros(193)
        test_cnt = 0
        test_mean = np.zeros((clone_x.shape[1]))
        for u in test_idx:
            test_mean+=clone_x[u]
            test_cnt+=1
        for u in train_idx:
            t=paper_year[u] - 1825
            train_cnt[t][labels[u]] += 1
            train_time_cnt[t] += 1
            train_mean[t][labels[u]] += clone_x[u]                 
        for t in range(193):
            for l in range(172):
                train_time_mean[t]+=train_mean[t][l]
                train_mean[t][l]/=max(1,train_cnt[t][l])
            train_time_mean[t]/=max(1,train_time_cnt[t])
        test_mean/=max(1,test_cnt)


        # Add norm values
        print(f"Add norm values {time.time()-t0}",flush=True)
        test_var = 0
        rsq = torch.zeros(193)
        msq = torch.zeros(193)
        for u in test_idx:
            test_var += np.linalg.norm(clone_x[u] - test_mean) ** 2
        for u in train_idx:
            t = paper_year[u] - 1825
            msq[t] += np.linalg.norm(train_mean[t][labels[u]] - train_time_mean[t]) ** 2
            rsq[t] += np.linalg.norm(clone_x[u] - train_mean[t][labels[u]]) ** 2


        # Calculate Statistics
        test_var/=max(1,test_cnt-1)
        for t in range(193):
            msq[t]/=max(1,train_time_cnt[t]-1)
            rsq[t]/=max(1,train_time_cnt[t]-1)
        alpha=torch.ones(193)
        for t in range(193):
            alpha_sq=(test_var-msq[t])/max(0.000001,rsq[t])
            if(alpha_sq>0):
                alpha[t]=torch.sqrt(alpha_sq)
            else:
                alpha[t]=0
            print(f"Time : {t+1825} | Number : {train_time_cnt[t]} | rsq : {rsq[t]:.4f} | msq : {msq[t]:.4f} | alpha_sq : {alpha_sq:.4f}")
        print(f"Test var : {test_var}")
        

        # Update modified vals
        print(f"Modify train feats {time.time()-t0}",flush=True)
        for u in train_idx:
            t = paper_year[u] - 1825
            clone_x[u] = alpha[t] * clone_x[u] + (1 - alpha[t]) * train_mean[t][labels[u]]
        

        # Save
        saved = np.concatenate((clone_x[train_idx], clone_x[valid_idx], clone_x[test_idx]), axis=0)
        torch.save(torch.from_numpy(saved).to(torch.float), f'{args.output_emb_prefix}_{i+1}.pt')
        x=clone_x

    else:
        saved = np.concatenate((x[train_idx], x[valid_idx], x[test_idx]), axis=0)
        torch.save(torch.from_numpy(saved).to(torch.float), f'{args.output_emb_prefix}_{i+1}.pt')
