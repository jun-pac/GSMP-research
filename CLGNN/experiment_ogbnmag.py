
import torch
import os
import numpy as np

from tqdm import tqdm
import pickle
import time
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from torch_sparse import SparseTensor
    
# path = args.use_emb
home_dir = os.getenv("HOME")
dataset = DglNodePropPredDataset(
    name="ogbn-mag", root=os.path.join(home_dir, ".ogb", "dataset"))
g, labels = dataset[0]

# my
#g = g.to(device)

splitted_idx = dataset.get_idx_split()
train_nid = splitted_idx["train"]['paper']
val_nid = splitted_idx["valid"]['paper']
test_nid = splitted_idx["test"]['paper']
features = g.nodes['paper'].data['feat']
paper_year=g.nodes['paper'].data['year']

# g.nodes["paper"].data["feat"] = features.to(device)


labels = labels['paper'].squeeze()
n_classes = int(labels.max() - labels.min()) + 1
train_nid, val_nid, test_nid = np.array(train_nid), np.array(val_nid), np.array(test_nid)


nums=torch.zeros(10)
for i in range(labels.shape[0]):
    nums[paper_year[i]-2010]+=1
for t in range(10): 
    print(f"Time : {t+2010}, nums : {nums[t]}, ratio : {nums[t]/len(labels)}")
print()

print(f"Train num : {train_nid.shape}")
print(f"Valid num : {val_nid.shape}")
print(f"Test num : {test_nid.shape}")
print()

print(f"Train year min, max : {torch.min(paper_year[train_nid])}, {torch.max(paper_year[train_nid])}")
print(f"Valid year min, max : {torch.min(paper_year[val_nid])}, {torch.max(paper_year[val_nid])}")
print(f"Test year min, max : {torch.min(paper_year[test_nid])}, {torch.max(paper_year[test_nid])}")