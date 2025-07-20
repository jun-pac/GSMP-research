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

def process_data(idx):
    print(f"Process data... {idx}/{shapes[0]}")
    train_mean_part = np.zeros((218, 172, clone_x.shape[1]))
    train_cnt_part = np.zeros((218, 172))
    train_time_mean_part = np.zeros((218, clone_x.shape[1]))
    train_time_cnt_part = np.zeros(218)
    test_cnt_part = 0
    test_mean_part = np.zeros((clone_x.shape[1]))

    for u in range(idx,min(idx+chunk_size,shapes[0])):
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

    return train_mean_part, train_cnt_part, train_time_mean_part, train_time_cnt_part, test_cnt_part, test_mean_part


def add_norm_data(idx):
    print(f"Add norm data... {idx}/{shapes[0]}")
    local_test_var = 0
    local_rsq = np.zeros(218)
    local_msq = np.zeros(218)

    for u in range(idx,min(idx+chunk_size,shapes[0])):
        if paper_year[u] == 2019:
            local_test_var += np.linalg.norm(clone_x[u] - test_mean) ** 2
        elif paper_year[u] <= 2017:
            t = max(0,paper_year[u] - 1800)
            if labels[u]<0 or labels[u]>=172:
                    continue
            local_msq[t] += np.linalg.norm(train_mean[t][labels[u]] - train_time_mean[t]) ** 2
            local_rsq[t] += np.linalg.norm(clone_x[u] - train_mean[t][labels[u]]) ** 2
    
    return local_test_var, local_rsq, local_msq


def update_data(idx):
    print(f"Update data... {idx}/{shapes[0]}")
    global clone_x
    for u in range(idx,min(idx+chunk_size,shapes[0])):
        if paper_year[u] <= 2017:
            t = max(0,paper_year[u] - 1800)
            if labels[u]<0 or labels[u]>=172:
                    continue
            clone_x[u] = alpha[t] * clone_x[u] + (1 - alpha[t]) * train_mean[t][labels[u]]


def sym_task(idx):
    newsrc=[]
    newdst=[]
    print(f"Persistent message passing... {idx}/{len_src}")
    for i in range(idx,min(idx+chunk_size,len_src)):
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

print('Making the graph undirected.')
if not (os.path.exists('./graph/pmp_src.npy') and os.path.exists('./graph/pmp_dst.npy')):
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
    src=np.load('./graph/pmp_src.npy')
    dst=np.load('./graph/pmp_dst.npy')

print(f'Computing adj... {time.time()-t0:.4f}')
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
        f_log.write(f"Processing {i+1}th features... {time.time()-t0}\n")
        f_log.flush()

        clone_x=np.copy(x)
        shapes = list(clone_x.shape)
        print(f"shapes : {shapes}\n")
        f_log.write(f"shapes : {shapes}\n")
        f_log.flush()

        # Calculte means
        train_mean = np.zeros((218, 172, clone_x.shape[1]))
        train_cnt = np.zeros((218, 172))
        train_time_mean = np.zeros((218, clone_x.shape[1]))
        train_time_cnt = np.zeros(218)
        test_cnt = 0
        test_mean = np.zeros((clone_x.shape[1]))
        with Pool(core_num) as p:
            result = p.map(process_data, range(0,shapes[0],chunk_size))
        print(f"type(result) : {type(result)}")
        for k in range(len(result)):
            train_mean_part, train_cnt_part, train_time_mean_part, train_time_cnt_part, test_cnt_part, test_mean_part=result[k]
            train_mean += train_mean_part
            train_cnt += train_cnt_part
            train_time_mean += train_time_mean_part
            train_time_cnt += train_time_cnt_part
            test_cnt += test_cnt_part
            test_mean += test_mean_part



        # Add norm values
        test_var = 0
        rsq = torch.zeros(218)
        msq = torch.zeros(218)
        with Pool(core_num) as p:
            result = p.map(add_norm_data, range(0,shapes[0],chunk_size))
        for k in range(len(result)):
            local_test_var, local_rsq, local_msq=result[k]
            test_var += local_test_var
            rsq += local_rsq
            msq += local_msq

        # del train_time_mean
        # del train_mean
        # del test_mean
        # gc.collect()    

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
        # print(f"initial type(clone_x) : {type(clone_x)}")
        # print(f"core_num/2 : {core_num/2}")
        # with Manager() as manager:
        #     past=clone_x[0][0]
        #     clone_x=manager.list(clone_x)
        #     with Pool(12) as p:
        #         p.map(update_data, range(0,shapes[0],chunk_size))
        #     clone_x=np.array(clone_x)
        #     clone_x=torch.tensor(clone_x)
        #     cur=clone_x[0][0]
        # print(f"type(clone_x) : {type(clone_x)}")
        # print(f"clone_x : {clone_x.dtype}")
        # print(f"Is it changed? {past} {cur}")
        # f_log.write(f"type(clone_x) : {type(clone_x)}\n")
        # f_log.write(f"clone_x : {clone_x.dtype}\n")
        # f_log.write(f"Is it changed? {past} {cur}\n")
        # f_log.flush()
        for u in tqdm(range(shapes[0])):
            if paper_year[u] <= 2017:
                t = max(0,paper_year[u] - 1800)
                if labels[u]<0 or labels[u]>=172:
                        continue
                clone_x[u] = alpha[t] * clone_x[u] + (1 - alpha[t]) * train_mean[t][labels[u]]
        

        # Save
        saved = np.concatenate((clone_x[train_idx], clone_x[valid_idx], clone_x[test_idx]), axis=0)
        torch.save(torch.from_numpy(saved).to(torch.float), f'{args.output_emb_prefix}_{i+1}.pt')
        # del clone_x
        # gc.collect()
    else:
        saved = np.concatenate((x[train_idx], x[valid_idx], x[test_idx]), axis=0)
        torch.save(torch.from_numpy(saved).to(torch.float), f'{args.output_emb_prefix}_{i+1}.pt')
