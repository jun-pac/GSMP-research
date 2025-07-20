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
import pickle
chunk_size=1000000
core_num=multiprocessing.cpu_count()

# def process_data(idx):
#     print(f"Process data... {idx}/{shapes[0]}")
#     train_mean_part = np.zeros((193, 172, clone_x.shape[1]))
#     train_cnt_part = np.zeros((193, 172))
#     train_time_mean_part = np.zeros((193, clone_x.shape[1]))
#     train_time_cnt_part = np.zeros(193)
#     test_cnt_part = 0
#     test_mean_part = np.zeros((clone_x.shape[1]))

#     for u in range(idx,min(idx+chunk_size,shapes[0])):
#         if paper_year[u] == 2019:
#             test_cnt_part += 1
#             test_mean_part += clone_x[u]
#         elif paper_year[u] <= 2017:
#             t=max(0,paper_year[u] - 1800)
#             if labels[u]<0 or labels[u]>=172:
#                     continue
#             train_cnt_part[t][labels[u]] += 1
#             train_time_cnt_part[t] += 1
#             train_mean_part[t][labels[u]] += clone_x[u]

#     return train_mean_part, train_cnt_part, train_time_mean_part, train_time_cnt_part, test_cnt_part, test_mean_part


# def add_norm_data(idx):
#     print(f"Add norm data... {idx}/{shapes[0]}")
#     local_test_var = 0
#     local_rsq = np.zeros(193)
#     local_msq = np.zeros(193)

#     for u in range(idx,min(idx+chunk_size,shapes[0])):
#         if paper_year[u] == 2019:
#             local_test_var += np.linalg.norm(clone_x[u] - test_mean) ** 2
#         elif paper_year[u] <= 2017:
#             t = max(0,paper_year[u] - 1800)
#             if labels[u]<0 or labels[u]>=172:
#                     continue
#             local_msq[t] += np.linalg.norm(train_mean[t][labels[u]] - train_time_mean[t]) ** 2
#             local_rsq[t] += np.linalg.norm(clone_x[u] - train_mean[t][labels[u]]) ** 2
    
#     return local_test_var, local_rsq, local_msq


# def update_data(idx):
#     print(f"Update data... {idx}/{shapes[0]}")
#     global clone_x
#     for u in range(idx,min(idx+chunk_size,shapes[0])):
#         if paper_year[u] <= 2017:
#             t = max(0,paper_year[u] - 1800)
#             if labels[u]<0 or labels[u]>=172:
#                     continue
#             clone_x[u] = alpha[t] * clone_x[u] + (1 - alpha[t]) * train_mean[t][labels[u]]


# def sym_task(idx):
#     newsrc=[]
#     newdst=[]
#     print(f"Persistent message passing... {idx}/{len_src}")
#     for i in range(idx,min(idx+chunk_size,len_src)):
#         newsrc.append(dst[i])
#         newdst.append(src[i])
#         if abs(paper_year[src[i]]-paper_year[dst[i]])>min(2019-paper_year[dst[i]],max(0,paper_year[dst[i]] - 1800)):
#             newsrc.append(src[i])
#             newdst.append(dst[i])
#         if abs(paper_year[src[i]]-paper_year[dst[i]])>min(2019-paper_year[src[i]],max(0,paper_year[src[i]] - 1800)):
#             newsrc.append(dst[i])
#             newdst.append(src[i])
#     return np.array([newsrc, newdst])

t0=time.time()

dataset = PygNodePropPredDataset('ogbn-papers100M', root='./dataset')
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
# 1207179, 125265, 214338
data = dataset[0]
paper_year=data['node_year']
labels=data['y']
labels=labels.long()
paper_year=paper_year.squeeze()
labels=labels.squeeze()
print(f"labels.shape : {labels.shape}")
print(f"First labals : {labels[0]}")

print()
print("First 10 train idx")
for i in range(10):
    print(train_idx[i])
print("First 10 valid idx")
for i in range(10):
    print(valid_idx[i])
print("First 10 text idx")
for i in range(10):
    print(test_idx[i])

print()
nums=torch.ones(2024)
for u in range(len(labels)):
    nums[paper_year[u]]+=1
    if(u%1000000==999999):
        print(f"Counting... {u+1}/{len(labels)}",flush=True)
for t in range(2024):
    print(f"Time {t+1825} | Num : {nums[t]} | Ratio : {nums[t]/len(train_idx)}",flush=True) 

for t in range(200):
    print(f"{nums[t+1824]}",flush=True) 



print()
print(f"train : {train_idx.shape}")
print(f"max min ; {torch.max(paper_year[train_idx])}, {torch.min(paper_year[train_idx])}")
print(f"max min ; {torch.max(labels[train_idx])}, {torch.min(labels[train_idx])}")

print()
print(f"valid : {valid_idx.shape}")
print(f"Year max min ; {torch.max(paper_year[valid_idx])}, {torch.min(paper_year[valid_idx])}")
print(f"Label max min ; {torch.max(labels[valid_idx])}, {torch.min(labels[valid_idx])}")

print()
print(f"test : {test_idx.shape}")
print(f"Year max min ; {torch.max(paper_year[test_idx])}, {torch.min(paper_year[test_idx])}")
print(f"Label max min ; {torch.max(labels[test_idx])}, {torch.min(labels[test_idx])}")

print("Flush",flush=True)

nums=torch.tensor(nums).float()
torch.save(nums,"./num_year.pt")

print(f"nums.dtype: {nums.dtype}")



# python experiment_papers100m.py |&tee test_output