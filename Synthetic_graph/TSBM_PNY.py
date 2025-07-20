import argparse

import dgl
import dgl.nn as dglnn
from dgl.data.utils import load_graphs

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
import os
import numpy as np
import random
import os
import time
import torch.nn.functional as F
from tqdm import tqdm 
from torch.multiprocessing import Pool
import multiprocessing
from itertools import repeat

chunk_size=100
sym_chunk_size=10000
core_num=multiprocessing.cpu_count()
len_newsrc=0
t0=time.time()

def task(idx):
    src=[]
    dst=[]
    for i in range(idx,min(idx+chunk_size,total_node)):
        for j in range(total_node):
            delta=abs(times[i]-times[j])
            yi=label[i]
            yj=label[j]
            if (i!=j and random.uniform(0, 1)<prob[yi][yj][delta]):
                src.append(i)
                dst.append(j)
    return np.array([src, dst])


def sym_task(idx):
    addsrc=[]
    adddst=[]
    for i in range(idx,min(idx+sym_chunk_size,len(newsrc))):
        if abs(times[newsrc[i]]-times[newdst[i]])>min(args.num_time-1-times[newdst[i]],times[newdst[i]]-0):
            addsrc.append(newsrc[i])
            adddst.append(newdst[i])
    return np.array([addsrc, adddst])


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size)) # Self connection?
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


class JJ_Norm(torch.nn.Module):
    def __init__(self, labels, times, num_time, num_label, split):
        super().__init__()
        self.labels = labels
        self.times = times
        self.num_time = num_time
        self.num_label = num_label
        self.split = split
        self.train_idx = (self.times<self.split).nonzero().squeeze()
        self.test_idx = (self.times>=self.split).nonzero().squeeze()
        #print(self.train_idx.shape)
        #print(self.test_idx.shape)

    def forward(self, x):
        clone_x=torch.clone(x)
        train_mean = torch.zeros((self.num_time, self.num_label, x.shape[1]), requires_grad=False)
        train_cnt = torch.zeros((self.num_time, self.num_label), requires_grad=False)
        train_time_mean = torch.zeros((self.num_time, x.shape[1]), requires_grad=False)
        train_time_cnt = torch.zeros(self.num_time, requires_grad=False)
        test_cnt = torch.zeros(1, requires_grad=False)
        test_mean = torch.zeros((x.shape[1]), requires_grad=False)
        for u in self.test_idx:
            test_cnt[0] += 1
            test_mean+=x[u]
        for u in self.train_idx:
            t=self.times[u]
            train_time_cnt[t] += 1
            train_cnt[t][self.labels[u]] += 1
            train_mean[t][self.labels[u]] += x[u]                 
        for t in range(self.num_time):
            for l in range(self.num_label):
                train_time_mean[t]+=train_mean[t][l]
                train_mean[t][l]=train_mean[t][l]/max(1,train_cnt[t][l])
            train_time_mean[t]/=max(1,train_time_cnt[t])
        test_mean/=max(1,test_cnt[0])


        # Add norm values
        test_var = torch.zeros(1,requires_grad=False)
        rsq = torch.zeros(self.num_time,requires_grad=False)
        msq = torch.zeros(self.num_time,requires_grad=False)
        for u in self.test_idx:
            test_var[0] += torch.norm(x[u] - test_mean) ** 2
        for u in self.train_idx:
            t = self.times[u]
            msq[t] += torch.norm(train_mean[t][self.labels[u]] - train_time_mean[t]) ** 2
            rsq[t] += torch.norm(x[u] - train_mean[t][self.labels[u]]) ** 2


        # Calculate Statistics
        test_var[0]/=max(1,test_cnt[0]-1)
        for t in range(self.split):
            msq[t]/=max(1,train_time_cnt[t]-1)
            rsq[t]/=max(1,train_time_cnt[t]-1)
        alpha=torch.ones(self.split,requires_grad=False)
        alpha_sq=torch.zeros(self.split,requires_grad=False)
        for t in range(self.split):
            alpha_sq[t]=(test_var[0]-msq[t])/max(0.000001,rsq[t])
            if(alpha_sq[t]>0):
                alpha[t]=torch.sqrt(alpha_sq[t])
            else:
                alpha[t]=0       


        # Update modified vals
        for u in self.train_idx:
            t = self.times[u]
            clone_x[u] = alpha[t] * x[u] + (1 - alpha[t]) * train_mean[t][self.labels[u]]
        return clone_x


class GCN2(nn.Module):
    def __init__(self, in_size, hid_size, out_size, labels, times):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size)) # Self connection?
        self.dropout = nn.Dropout(0.5)
        self.JJ_Norm=JJ_Norm(labels, times, args.num_time, args.num_label, args.split)


    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            h = self.JJ_Norm(h)
        return h


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model, f_log):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    for epoch in range(args.epoch):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%100==99:
            with torch.no_grad():
                acc = evaluate(g, features, labels, masks[2], model)
                print(f"{acc:.4f}", end=' ', flush=True)
                # f_log.write(f"{acc:.4f} ")
                # f_log.flush()
    print('',flush=True)
    # f_log.write(f"\n")
    # f_log.flush()

def train2(g, features, labels, masks, model, f_log):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    for epoch in range(args.epoch):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%100==99:
            with torch.no_grad():
                acc = evaluate(g, features, labels, masks[2], model)
                print(f"{acc:.4f}", end=' ', flush=True)
                # f_log.write(f"{acc:.4f} ")
                # f_log.flush()
    print('',flush=True)
    # f_log.write(f"\n")
    # f_log.flush()


def train_0(g, f_log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"].type(torch.bool), g.ndata["val_mask"].type(torch.bool), g.ndata["test_mask"].type(torch.bool)
    model = GCN(features.shape[1], 16, 10).to(device)
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)
    train(g, features, labels, masks, model, f_log)
    acc = evaluate(g, features, labels, masks[2], model)
    # print(f"{acc:.4f}", end=' ', flush=True)
    f_log.write(f"{acc:.4f}\n")
    f_log.flush()
    return acc


def train_1(g, f_log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"].type(torch.bool), g.ndata["val_mask"].type(torch.bool), g.ndata["test_mask"].type(torch.bool)
    model = GCN(features.shape[1], 16, 10).to(device) # Same model, but use symmetric data
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)
    train(g, features, labels, masks, model, f_log)
    acc = evaluate(g, features, labels, masks[2], model)
    # print(f"{acc:.4f}", end=' ', flush=True)
    f_log.write(f"{acc:.4f}\n")
    f_log.flush()
    return acc


def train_2(g, f_log, times):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"].type(torch.bool), g.ndata["val_mask"].type(torch.bool), g.ndata["test_mask"].type(torch.bool)
    model = GCN2(features.shape[1], 16, 10, labels, times).to(device)
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)
    train2(g, features, labels, masks, model, f_log)
    acc = evaluate(g, features, labels, masks[2], model)
    # print(f"{acc:.4f}", end=' ', flush=True)
    f_log.write(f"{acc:.4f}\n")
    f_log.flush()
    return acc



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=str, default='./data/')
    parser.add_argument("--feat_dim", type=int, default=5)
    parser.add_argument("--num_label", type=int, default=10)
    parser.add_argument("--num_time", type=int, default=10)
    parser.add_argument("--dt", type=str, default="float",help="data type(float, bfloat16)")
    parser.add_argument("--epoch", type=int, default=1500)
    parser.add_argument("--feat_std", type=float, default=4)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--split", type=int, default=8) # if t>=split, than node belongs to test set.
    parser.add_argument("--K", type=float, default=0.6)
    parser.add_argument("--G", type=float, default=0.4)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--L", type=int, default=9) # Number of label type... 
    
    args = parser.parse_args()
    f_log0=open('./txtlog/'+'NI_'+'L'+str(args.L)+'_'+'d'+str(args.decay)+'_'+str(args.K)+'_'+str(args.G)+'_0','a')
    f_log1=open('./txtlog/'+'NI_'+'L'+str(args.L)+'_'+'d'+str(args.decay)+'_'+str(args.K)+'_'+str(args.G)+'_1','a')
    f_log2=open('./txtlog/'+'NI_'+'L'+str(args.L)+'_'+'d'+str(args.decay)+'_'+str(args.K)+'_'+str(args.G)+'_2','a')

    f_log0.write('NI_'+'L'+str(args.L)+'_'+'d'+str(args.decay)+'_'+str(args.K)+'_'+str(args.G)+'_0\n')
    f_log0.flush()
    f_log1.write('NI_'+'L'+str(args.L)+'_'+'d'+str(args.decay)+'_'+str(args.K)+'_'+str(args.G)+'_1\n')
    f_log1.flush()
    f_log2.write('NI_'+'L'+str(args.L)+'_'+'d'+str(args.decay)+'_'+str(args.K)+'_'+str(args.G)+'_2\n')
    f_log2.flush()
    
    flag=0
    for i in range(args.repeat):
        total_node=0
        labels=[0]*args.num_label

        if args.L==0:
            temp=torch.randint(20,180,(args.num_label,))
            for l in range(args.num_label):
                labels[l]=int(temp[l])
                total_node+=labels[l]
        elif args.L==1:
            temp=torch.randint(50,150,(args.num_label,))
            for l in range(args.num_label):
                labels[l]=int(temp[l])
                total_node+=labels[l]
        elif args.L==2:
            temp=torch.randint(80,120,(args.num_label,))
            for l in range(args.num_label):
                labels[l]=int(temp[l])
                total_node+=labels[l]
        elif args.L==3:
            temp=torch.randn(args.num_label)
            temp=temp*50+100
            for l in range(args.num_label):
                labels[l]=max(0,int(temp[l]))
                total_node+=labels[l]
        elif args.L==4:
            temp=torch.randn(args.num_label)
            temp=temp*30+100
            for l in range(args.num_label):
                labels[l]=max(0,int(temp[l]))
                total_node+=labels[l]
        elif args.L==5:
            temp=torch.randn(args.num_label)
            temp=temp*20+100
            for l in range(args.num_label):
                labels[l]=max(0,int(temp[l]))
                total_node+=labels[l]
        else:
            for l in range(args.num_label):
                labels[l]=100
                total_node+=labels[l]


        label=torch.randint(0,args.num_label,(total_node,))
        idx=0
        for l in range(args.num_label):
            for k in range(labels[l]):
                label[idx]=l
                idx+=1
        label_cnt=torch.zeros(args.num_label)
        for i in range(total_node):
            label_cnt[label[i]]+=1
        times=torch.randint(0,args.num_time,(total_node,))
        print(f"label_cnt : ",end=' ')
        for i in range(args.num_label):
            print(f"{label_cnt[i]}",end=' ')
        print()

        train_mask=[0]*total_node
        valid_mask=[0]*total_node
        test_mask=[0]*total_node



        for i in range(total_node):
            train_mask[i]=(times[i]<args.split)
            valid_mask[i]=(times[i]>=args.split and times[i]<args.split)
            test_mask[i]=(times[i]>=args.split)
            
        label=torch.Tensor(label).type(torch.long)
        times=torch.Tensor(times).type(torch.long)
        train_mask=torch.Tensor(train_mask).type(torch.bool)
        valid_mask=torch.Tensor(valid_mask).type(torch.bool)
        test_mask=torch.Tensor(test_mask).type(torch.bool)

        prob=torch.zeros((args.num_label, args.num_label, args.num_time))
        print(f"decay factor : {args.decay:.4f}")

        for y in range(args.num_label):
            prob[y][y][0]=args.K
        for y1 in range(args.num_label):
            for y2 in range(y1):
                prob[y1][y2][0]=args.K * args.G
                prob[y2][y1][0]=args.K * args.G
        
        for t in range(1,args.num_time):
            for y1 in range(args.num_label):
                for y2 in range(args.num_label):
                    prob[y1][y2][t]=prob[y1][y2][t-1]*args.decay

        with Pool(core_num) as p:
            result = p.map(task, range(0,total_node,chunk_size))
        src_dst = np.concatenate(result, axis=1)
        src, dst = src_dst
        
        newsrc=np.concatenate((src,dst))
        newdst=np.concatenate((dst,src))

        with Pool(core_num) as p:
            result = p.map(sym_task, range(0,len(newsrc),sym_chunk_size))
        src_dst = np.concatenate(result, axis=1)
        addsrc, adddst = src_dst
        symsrc=np.concatenate((newsrc,addsrc))
        symdst=np.concatenate((newdst,adddst))

        print(f"\n Num edges : {len(newsrc)}, Num sym edges : {len(symsrc)}")

        new_edges={}
        sym_edges={}
        num_nodes={}
        new_edges[('node','-','node')]=(newsrc,newdst)
        sym_edges[('node','-','node')]=(symsrc,symdst)
        num_nodes['node']=total_node
        new_g = dgl.heterograph(new_edges,num_nodes_dict=num_nodes)
        sym_g = dgl.heterograph(sym_edges,num_nodes_dict=num_nodes)

        feat_center=torch.zeros([args.num_label,args.feat_dim])
        for i in range(args.num_label):
            feat_center[i]=torch.randn(args.feat_dim)

        feature=torch.zeros([total_node,args.feat_dim])
        for i in range(total_node):
            cur_label = label[i]
            feature[i] = torch.randn(args.feat_dim)*args.feat_std+feat_center[cur_label]
            
        new_g.nodes['node'].data["label"]=label
        new_g.nodes['node'].data["feat"]=feature
        new_g.nodes['node'].data["train_mask"]=train_mask
        new_g.nodes['node'].data["val_mask"]=valid_mask
        new_g.nodes['node'].data["test_mask"]=test_mask
        sym_g.nodes['node'].data["label"]=label
        sym_g.nodes['node'].data["feat"]=feature
        sym_g.nodes['node'].data["train_mask"]=train_mask
        sym_g.nodes['node'].data["val_mask"]=valid_mask
        sym_g.nodes['node'].data["test_mask"]=test_mask

        acc0=train_0(new_g, f_log0)
        acc1=train_1(sym_g, f_log1)
        acc2=train_2(sym_g, f_log2, times)


    

''' 
# 0430

conda activate vv

python TSBM_Nature9.py --L 0 --decay 0.45 --epoch 200
python TSBM_Nature9.py --L 0 --decay 0.65 --epoch 200
python TSBM_Nature9.py --L 1 --decay 0.45 --epoch 200
python TSBM_Nature9.py --L 1 --decay 0.65 --epoch 200
python TSBM_Nature9.py --L 2 --decay 0.45 --epoch 200
python TSBM_Nature9.py --L 2 --decay 0.65 --epoch 200
python TSBM_Nature9.py --L 3 --decay 0.45 --epoch 200
python TSBM_Nature9.py --L 3 --decay 0.65 --epoch 200
python TSBM_Nature9.py --L 4 --decay 0.45 --epoch 200
python TSBM_Nature9.py --L 4 --decay 0.65 --epoch 200
python TSBM_Nature9.py --L 5 --decay 0.45 --epoch 200
python TSBM_Nature9.py --L 5 --decay 0.65 --epoch 200


# 0501
python TSBM_Nature9.py --L 5 --decay 0.40 --epoch 200
// python TSBM_Nature9.py --L 5 --decay 0.45 --epoch 200
python TSBM_Nature9.py --L 5 --decay 0.50 --epoch 200
python TSBM_Nature9.py --L 5 --decay 0.55 --epoch 200
python TSBM_Nature9.py --L 5 --decay 0.60 --epoch 200
// python TSBM_Nature9.py --L 5 --decay 0.65 --epoch 200
python TSBM_Nature9.py --L 5 --decay 0.70 --epoch 200
python TSBM_Nature9.py --L 5 --decay 0.75 --epoch 200

python TSBM_Nature9.py --decay 0.40 --epoch 200
python TSBM_Nature9.py --decay 0.45 --epoch 200
python TSBM_Nature9.py --decay 0.50 --epoch 200
python TSBM_Nature9.py --decay 0.55 --epoch 200
python TSBM_Nature9.py --decay 0.60 --epoch 200
python TSBM_Nature9.py --decay 0.65 --epoch 200

'''