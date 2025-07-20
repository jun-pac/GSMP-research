import argparse

import dgl
import dgl.nn as dglnn
from GATmodule import GAT, GAT_JJ, GAT_PNY

from dgl.data.utils import load_graphs

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def mono_task(idx):
    addsrc=[]
    adddst=[]
    for i in range(idx,min(idx+sym_chunk_size,len(newsrc))):
        if times[newsrc[i]]>=times[newdst[i]]:
            addsrc.append(newsrc[i])
            adddst.append(newdst[i])
    return np.array([addsrc, adddst])


def sym_task(idx):
    addsrc=[]
    adddst=[]
    for i in range(idx,min(idx+sym_chunk_size,len(newsrc))):
        if abs(times[newsrc[i]]-times[newdst[i]])>min(args.num_time-1-times[newdst[i]],times[newdst[i]]-0):
            addsrc.append(newsrc[i])
            adddst.append(newdst[i])
    return np.array([addsrc, adddst])



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
        if epoch%20==19:
            with torch.no_grad():
                acc = evaluate(g, features, labels, masks[2], model)
                print(f"{acc:.4f}", end=' ', flush=True)
                # f_log.write(f"{acc:.4f} ")
                # f_log.flush()
    print('',flush=True)
    # f_log.write(f"\n")
    # f_log.flush()


def train_base(g, f_log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"].type(torch.bool), g.ndata["val_mask"].type(torch.bool), g.ndata["test_mask"].type(torch.bool)
    model = GAT(features.shape[1], 16, 10).to(device)
    train(g, features, labels, masks, model, f_log)
    acc = evaluate(g, features, labels, masks[2], model)
    # print(f"{acc:.4f}", end=' ', flush=True)
    f_log.write(f"{acc:.4f}\n")
    f_log.flush()
    return acc


def train_PMP(g, f_log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"].type(torch.bool), g.ndata["val_mask"].type(torch.bool), g.ndata["test_mask"].type(torch.bool)
    model = GAT(features.shape[1], 16, 10).to(device)
    train(g, features, labels, masks, model, f_log)
    acc = evaluate(g, features, labels, masks[2], model)
    # print(f"{acc:.4f}", end=' ', flush=True)
    f_log.write(f"{acc:.4f}\n")
    f_log.flush()
    return acc

def train_MMP(g, f_log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"].type(torch.bool), g.ndata["val_mask"].type(torch.bool), g.ndata["test_mask"].type(torch.bool)
    model = GAT(features.shape[1], 16, 10).to(device)
    train(g, features, labels, masks, model, f_log)
    acc = evaluate(g, features, labels, masks[2], model)
    # print(f"{acc:.4f}", end=' ', flush=True)
    f_log.write(f"{acc:.4f}\n")
    f_log.flush()
    return acc


def train_JJ(g, f_log, times):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"].type(torch.bool), g.ndata["val_mask"].type(torch.bool), g.ndata["test_mask"].type(torch.bool)
    model = GAT_JJ(labels, times, args.num_time, args.num_label, args.split, features.shape[1], 16, 10).to(device)
    train(g, features, labels, masks, model, f_log)
    acc = evaluate(g, features, labels, masks[2], model)
    # print(f"{acc:.4f}", end=' ', flush=True)
    f_log.write(f"{acc:.4f}\n")
    f_log.flush()
    return acc

def train_PNY(g, f_log, relative_conn, times):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"].type(torch.bool), g.ndata["val_mask"].type(torch.bool), g.ndata["test_mask"].type(torch.bool)
    model = GAT_PNY(labels, times, args.num_time, args.num_label, args.split, relative_conn, features.shape[1], 16, 10).to(device)
    train(g, features, labels, masks, model, f_log)
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
    parser.add_argument("--epoch", type=int, default=1500)
    parser.add_argument("--feat_std", type=float, default=4)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--split", type=int, default=8) # if t>=split, than node belongs to test set.
    parser.add_argument("--K", type=float, default=0.6)
    parser.add_argument("--G", type=float, default=0.4)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--L", type=int, default=0) # Number of label type... 
    
    args = parser.parse_args()
    f_log_base=open('./txtlog/'+'GAT_'+'L'+str(args.L)+'_'+'d'+str(args.decay)+'_'+str(args.K)+'_'+str(args.G)+'_b','a')
    f_log_MMP=open('./txtlog/'+'GAT_'+'L'+str(args.L)+'_'+'d'+str(args.decay)+'_'+str(args.K)+'_'+str(args.G)+'_mmp','a')
    f_log_PMP=open('./txtlog/'+'GAT_'+'L'+str(args.L)+'_'+'d'+str(args.decay)+'_'+str(args.K)+'_'+str(args.G)+'_pmp','a')
    f_log_JJ=open('./txtlog/'+'GAT_'+'L'+str(args.L)+'_'+'d'+str(args.decay)+'_'+str(args.K)+'_'+str(args.G)+'_jj','a')
    f_log_PNY=open('./txtlog/'+'GAT_'+'L'+str(args.L)+'_'+'d'+str(args.decay)+'_'+str(args.K)+'_'+str(args.G)+'_pny','a')

    # f_log_base.write('GAT_'+'L'+str(args.L)+'_'+'d'+str(args.decay)+'_'+str(args.K)+'_'+str(args.G)+'_b'+'\n')
    # f_log_base.flush()
    # f_log_MMP.write('GAT_'+'L'+str(args.L)+'_'+'d'+str(args.decay)+'_'+str(args.K)+'_'+str(args.G)+'_mmp'+'\n')
    # f_log_MMP.flush()
    # f_log_PMP.write('GAT_'+'L'+str(args.L)+'_'+'d'+str(args.decay)+'_'+str(args.K)+'_'+str(args.G)+'_pmp'+'\n')
    # f_log_PMP.flush()
    # f_log_JJ.write('GAT_'+'L'+str(args.L)+'_'+'d'+str(args.decay)+'_'+str(args.K)+'_'+str(args.G)+'_jj'+'\n')
    # f_log_JJ.flush()
    # f_log_PNY.write('GAT_'+'L'+str(args.L)+'_'+'d'+str(args.decay)+'_'+str(args.K)+'_'+str(args.G)+'_pny'+'\n')
    # f_log_PNY.flush()    
    

    for i in range(args.repeat):
        total_node=1000
        label=torch.randint(0,args.num_label,(total_node,))
        label_cnt=torch.zeros(args.num_label)
        for i in range(total_node):
            label_cnt[label[i]]+=1        

        times=torch.randint(0,args.num_time,(total_node,))
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

        
        print(f"decay factor : {args.decay:.4f}")
        prob=torch.zeros((args.num_label, args.num_label, args.num_time))
        
        if args.L==0:
            for y in range(args.num_label):
                prob[y][y][0]=args.K   
            for y1 in range(args.num_label):
                for y2 in range(y1):
                    prob[y1][y2][0]=args.K * args.G
                    prob[y2][y1][0]=prob[y1][y2][0]
            for t in range(1,args.num_time):
                for y1 in range(args.num_label):
                    for y2 in range(args.num_label):
                        prob[y1][y2][t]=prob[y1][y2][t-1]*args.decay
            
        elif args.L==1:
            for y in range(args.num_label):
                prob[y][y][0]=random.uniform(0, args.K)   
            for y1 in range(args.num_label):
                for y2 in range(y1):
                    prob[y1][y2][0]=random.uniform(0, args.K * args.G)
                    prob[y2][y1][0]=prob[y1][y2][0]
            for t in range(1,args.num_time):
                for y1 in range(args.num_label):
                    for y2 in range(args.num_label):
                        prob[y1][y2][t]=prob[y1][y2][t-1]*args.decay
            
        elif args.L==2:
            for y in range(args.num_label):
                prob[y][y][0]=random.uniform(0, args.K)   
            for y1 in range(args.num_label):
                for y2 in range(y1):
                    prob[y1][y2][0]=random.uniform(0,torch.sqrt(prob[y1][y1][0]*prob[y2][y2][0])*args.G)
                    prob[y2][y1][0]=prob[y1][y2][0]
            for t in range(1,args.num_time):
                for y1 in range(args.num_label):
                    for y2 in range(args.num_label):
                        prob[y1][y2][t]=prob[y1][y2][t-1]*args.decay
            
        elif args.L==3:
            for y in range(args.num_label):
                prob[y][y][0]=args.K   
            for y1 in range(args.num_label):
                for y2 in range(y1):
                    prob[y1][y2][0]=args.K * args.G
                    prob[y2][y1][0]=prob[y1][y2][0]    
            decay_matrix=torch.zeros((args.num_label,args.num_label))
            for y1 in range(args.num_label):
                for y2 in range(y1):
                    decay_matrix[y1][y2]=random.uniform(0.4,0.6)
                    decay_matrix[y2][y1]=decay_matrix[y1][y2]

            for t in range(1,args.num_time):
                for y1 in range(args.num_label):
                    for y2 in range(args.num_label):
                        prob[y1][y2][t]=prob[y1][y2][t-1]*decay_matrix[y1][y2]
            
        elif args.L==4:
            for y in range(args.num_label):
                prob[y][y][0]=args.K   
            for y1 in range(args.num_label):
                for y2 in range(y1):
                    prob[y1][y2][0]=args.K * args.G
                    prob[y2][y1][0]=prob[y1][y2][0]    
            decay_matrix=torch.zeros((args.num_label,args.num_label))
            for y1 in range(args.num_label):
                for y2 in range(y1):
                    decay_matrix[y1][y2]=random.uniform(0.3,0.9)
                    decay_matrix[y2][y1]=decay_matrix[y1][y2]

            for t in range(1,args.num_time):
                for y1 in range(args.num_label):
                    for y2 in range(args.num_label):
                        prob[y1][y2][t]=prob[y1][y2][t-1]*decay_matrix[y1][y2]
            
        else:
            for y in range(args.num_label):
                prob[y][y][0]=args.K   
            for y1 in range(args.num_label):
                for y2 in range(y1):
                    prob[y1][y2][0]=args.K * args.G
                    prob[y2][y1][0]=prob[y1][y2][0]    
            decay_matrix=torch.zeros((args.num_label,args.num_label))
            for y1 in range(args.num_label):
                for y2 in range(y1):
                    decay_matrix[y1][y2]=random.uniform(0,1)
                    decay_matrix[y2][y1]=decay_matrix[y1][y2]

            for t in range(1,args.num_time):
                for y1 in range(args.num_label):
                    for y2 in range(args.num_label):
                        prob[y1][y2][t]=prob[y1][y2][t-1]*decay_matrix[y1][y2]
        
        
        relative_conn=torch.zeros((args.num_label, args.num_time, args.num_label, args.num_time))
        
        for t1 in range(args.num_time):
            for y1 in range(args.num_label):
                sum=0
                for y2 in range(args.num_label):                    
                    for t2 in range(args.num_time):
                        sum+=prob[y1][y2][abs(t2-t1)]
                for y2 in range(args.num_label):                    
                    for t2 in range(args.num_time):
                        relative_conn[y1][t1][y2][t2]=prob[y1][y2][abs(t2-t1)]/sum
        

        with Pool(core_num) as p:
            result = p.map(task, range(0,total_node,chunk_size))
        src_dst = np.concatenate(result, axis=1)
        src, dst = src_dst

        newsrc=np.concatenate((src,dst))
        newdst=np.concatenate((dst,src))

        with Pool(core_num) as p:
            result = p.map(mono_task, range(0,len(newsrc),sym_chunk_size))
        src_dst = np.concatenate(result, axis=1)
        monosrc, monodst = src_dst


        with Pool(core_num) as p:
            result = p.map(sym_task, range(0,len(newsrc),sym_chunk_size))
        src_dst = np.concatenate(result, axis=1)
        addsrc, adddst = src_dst
        symsrc=np.concatenate((newsrc,addsrc))
        symdst=np.concatenate((newdst,adddst))

        print(f"\n Num edges : {len(newsrc)}, Num sym edges : {len(symsrc)}")

        new_edges={}
        sym_edges={}
        mono_edges={}

        num_nodes={}
        new_edges[('node','-','node')]=(newsrc,newdst)
        sym_edges[('node','-','node')]=(symsrc,symdst)
        mono_edges[('node','-','node')]=(monosrc,monodst)

        num_nodes['node']=total_node
        new_g = dgl.heterograph(new_edges,num_nodes_dict=num_nodes)
        sym_g = dgl.heterograph(sym_edges,num_nodes_dict=num_nodes)
        mono_g = dgl.heterograph(mono_edges,num_nodes_dict=num_nodes)
        new_g=dgl.add_self_loop(new_g,etype='-')
        sym_g=dgl.add_self_loop(sym_g,etype='-')
        mono_g=dgl.add_self_loop(mono_g,etype='-')

        feat_center=torch.zeros([args.num_label,args.feat_dim])
        for i in range(args.num_label):
            feat_center[i]=torch.randn(args.feat_dim)

        feature=torch.zeros([total_node,args.feat_dim])
        feat_std=[random.uniform(0,2*args.feat_std) for y in range(args.num_label)]
        for i in range(total_node):
            cur_label = label[i]
            feature[i] = torch.randn(args.feat_dim)*feat_std[cur_label]+feat_center[cur_label]
            
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
        mono_g.nodes['node'].data["label"]=label
        mono_g.nodes['node'].data["feat"]=feature
        mono_g.nodes['node'].data["train_mask"]=train_mask
        mono_g.nodes['node'].data["val_mask"]=valid_mask
        mono_g.nodes['node'].data["test_mask"]=test_mask
        
        acc_PMP=train_PMP(sym_g, f_log_PMP)
        acc_JJ=train_JJ(sym_g, f_log_JJ, times)
        acc_PNY=train_PNY(sym_g, f_log_PNY, relative_conn, times)
        #acc_base=train_base(new_g, f_log_base)
        #acc_MMP=train_MMP(mono_g, f_log_MMP)

'''
# 0511

conda activate vv

while sleep 1; do python TSBM_GAT.py --L 0 --decay 0.4 --epoch 200; done
while sleep 1; do python TSBM_GAT.py --L 0 --decay 0.5 --epoch 200; done
while sleep 1; do python TSBM_GAT.py --L 0 --decay 0.6 --epoch 200; done
while sleep 1; do python TSBM_GAT.py --L 0 --decay 0.7 --epoch 200; done
while sleep 1; do python TSBM_GAT.py --L 3 --epoch 200; done
while sleep 1; do python TSBM_GAT.py --L 4 --epoch 200; done



while sleep 1; do python TSBM_GAT.py --L 5 --epoch 200; done
while sleep 1; do python TSBM_GAT.py --L 0 --decay 0.45 --epoch 200; done
while sleep 1; do python TSBM_GAT.py --L 0 --decay 0.55 --epoch 200; done
while sleep 1; do python TSBM_GAT.py --L 0 --decay 0.65 --epoch 200; done
while sleep 1; do python TSBM_GAT.py --L 0 --decay 0.75 --epoch 200; done
while sleep 1; do python TSBM_GAT.py --L 0 --decay 0.8 --epoch 200; done




while sleep 1; do python TSBM_GAT.py --L 0 --decay 0.4 --epoch 400; done
while sleep 1; do python TSBM_GAT.py --L 0 --decay 0.5 --epoch 400; done
while sleep 1; do python TSBM_GAT.py --L 0 --decay 0.6 --epoch 400; done


'''
    
