import argparse

import dgl
import torch
import numpy as np
import random
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs
import os
import time
import torch.nn.functional as F
from tqdm import tqdm 
from multiprocessing import Pool
import multiprocessing
from itertools import repeat
chunk_size=100
sym_chunk_size=1000
core_num=multiprocessing.cpu_count()
len_newsrc=0

t0=time.time()
num_label=10
num_time=100

# Define Sample Functions 
sample_func={}
'''
'''
sample_func['MUL1']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['MUL2']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['MUL3']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]

sample_func['TEST1']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['TEST2']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['TEST3']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['TEST4']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['TEST5']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]

sample_func['TEST15']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['TEST25']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['TEST35']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['TEST45']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]


# Define SBM probability matrices
matrix_func={}
'''
matrix_func['MUL1_tinvar']=lambda delta_t,label_i,label_j:(0.5 if label_i==label_j else 0.2)
matrix_func['MUL2_tinvar']=lambda delta_t,label_i,label_j:(0.5 if label_i==label_j else 0.2)
matrix_func['MUL3_tinvar']=lambda delta_t,label_i,label_j:(0.8 if label_i==label_j else 0.3)
'''
matrix_func['MUL1']=lambda delta_t,label_i,label_j:(0.5 if label_i==label_j else 0.4*(num_time-delta_t)/num_time)
matrix_func['MUL2']=lambda delta_t,label_i,label_j:(0.8 if label_i==label_j else 0.4*(num_time-delta_t)/num_time)
matrix_func['MUL3']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.2*(num_time-delta_t)/num_time)

matrix_func['TEST1']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.1333+0.0*(num_time-delta_t)/num_time)
matrix_func['TEST2']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.1000+0.05*(num_time-delta_t)/num_time)
matrix_func['TEST3']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.0666+0.1*(num_time-delta_t)/num_time)
matrix_func['TEST4']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.0333+0.15*(num_time-delta_t)/num_time)
matrix_func['TEST5']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.0000+0.2*(num_time-delta_t)/num_time)

matrix_func['TEST15']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.11667+0.025*(num_time-delta_t)/num_time)
matrix_func['TEST25']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.08333+0.075*(num_time-delta_t)/num_time)
matrix_func['TEST35']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.05000+0.125*(num_time-delta_t)/num_time)
matrix_func['TEST45']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.01667+0.175*(num_time-delta_t)/num_time)


# TEST5 is exactly same as MUL3

def task(idx):
    src=[]
    dst=[]
    for i in range(idx,min(idx+chunk_size,args.num_node)):
        for j in range(args.num_node):
            delta=abs(times[i]-times[j])
            yi=label[i]
            yj=label[j]
            if (i!=j and random.uniform(0, 1)<matrix(delta,yi,yj)):
                src.append(i)
                dst.append(j)
    return np.array([src, dst])


def mono_task(idx):
    addsrc=[]
    adddst=[]
    for i in range(idx,min(idx+sym_chunk_size,len_newsrc)):
        if times[newsrc[i]]<times[newdst[i]]:
            addsrc.append(newsrc[i])
            adddst.append(newdst[i])
    return np.array([addsrc, adddst])
    

def sym_task(idx):
    addsrc=[]
    adddst=[]
    for i in range(idx,min(idx+sym_chunk_size,len_newsrc)):
        if abs(times[newsrc[i]]-times[newdst[i]])>min(num_time-1-times[newdst[i]],times[newdst[i]]-0):
            addsrc.append(newsrc[i])
            adddst.append(newdst[i])
    return np.array([addsrc, adddst])
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphnum", type=str, default='3')
    parser.add_argument("--time_invar", action='store_true', default=False)
    parser.add_argument("--root",type=str, default='./data/')
    parser.add_argument("--sym", action='store_true', default=False)
    parser.add_argument("--sym2", action='store_true', default=False)
    parser.add_argument("--mono1", action='store_true', default=False)
    parser.add_argument("--mono2", action='store_true', default=False)
    parser.add_argument("--feat_dim", type=int, default=5)
    parser.add_argument("--num_node", type=int, default=3000)
    args = parser.parse_args()

    print(f"Number of cores... {core_num}")

    # Init names
    graph_name=args.graphnum+("_tinvar" if args.time_invar else "")    
    name=graph_name+("_sym" if args.sym else "")
    name=name+("_sym2" if args.sym2 else "")
    name=name+("_mono1" if args.mono1 else "")
    name=name+("_mono2" if args.mono2 else "")
    name=name+"_feat"+str(args.feat_dim)
    name=name+"_node"+str(args.num_node)
    print(f"Your graph's name : {name}")


    # Init nodes
    label_rands=torch.randperm(args.num_node)
    time_rands=torch.randperm(args.num_node)

    label=[0]*args.num_node
    times=[0]*args.num_node
    train_mask=[0]*args.num_node
    valid_mask=[0]*args.num_node
    test_mask=[0]*args.num_node

    for i in range(args.num_node):
        label[i]=label_rands[i]%num_label
        times[i]=time_rands[i]%num_time

    for i in range(args.num_node):
        train_mask[i]=(times[i]<55)
        valid_mask[i]=(times[i]>=55 and times[i]<60)
        test_mask[i]=(times[i]>=60)
        
    label=torch.Tensor(label).type(torch.long)
    times=torch.Tensor(times).type(torch.long)
    train_mask=torch.Tensor(train_mask).type(torch.bool)
    valid_mask=torch.Tensor(valid_mask).type(torch.bool)
    test_mask=torch.Tensor(test_mask).type(torch.bool)


    matrix=matrix_func[graph_name]
    with Pool(core_num) as p:
        result = p.map(task, range(0,args.num_node,chunk_size))
    src_dst = np.concatenate(result, axis=1)
    src, dst = src_dst
    
    newsrc=np.concatenate((src,dst))
    newdst=np.concatenate((dst,src))

    if(args.mono1 or args.mono2):
        len_newsrc=len(newsrc)
        with Pool(core_num) as p:
            result = p.map(mono_task, range(0,len(newsrc),sym_chunk_size))
        src_dst = np.concatenate(result, axis=1)
        addsrc, adddst = src_dst
        newsrc=addsrc
        newdst=adddst
        if(args.mono2):
            newsrc,newdst = newdst,newsrc

    if(args.sym or args.sym2):
        len_newsrc=len(newsrc)
        with Pool(core_num) as p:
            result = p.map(sym_task, range(0,len(newsrc),sym_chunk_size))
        src_dst = np.concatenate(result, axis=1)
        addsrc, adddst = src_dst
        newsrc=np.concatenate((newsrc,addsrc))
        newdst=np.concatenate((newdst,adddst))
        if(args.sym2):
            newsrc,newdst = newdst,newsrc

    print(f"Total Number of edges : {newsrc.shape[0]}")

    new_edges={}
    num_nodes={}
    new_edges[('node','-','node')]=(newsrc,newdst) # undirected
    num_nodes['node']=args.num_node
    new_g = dgl.heterograph(new_edges,num_nodes_dict=num_nodes)
    if(args.mono1 or args.mono2):
        new_g=dgl.add_self_loop(new_g)
    

    feat_center=torch.zeros([num_label,args.feat_dim])
    for i in range(num_label):
        feat_center[i]=torch.randn(args.feat_dim)

    feature=torch.zeros([args.num_node,args.feat_dim])
    for i in range(args.num_node):
        cur_label = label[i]
        feature[i] = sample_func[args.graphnum](cur_label)

    new_g.nodes['node'].data["label"]=label
    new_g.nodes['node'].data["feat"]=feature
    new_g.nodes['node'].data["train_mask"]=train_mask
    new_g.nodes['node'].data["val_mask"]=valid_mask
    new_g.nodes['node'].data["test_mask"]=test_mask

    name=args.root+name
    glabel={'label':label}
    # if not os.path.exists(name):
    #     save_graphs(name,[new_g],glabel)
    save_graphs(name,[new_g],glabel)
    print(f"Successfully saved!... {time.time()-t0}sec")

'''
Node number : 3000 (55 : 5 : 40)
python3 SBM_graph.py --graphnum MUL2 --feat_dim 5
python3 SBM_graph.py --graphnum MUL2 --sym --feat_dim 5

python3 SBM_graph.py --graphnum MUL3 --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum MUL3 --sym --feat_dim 5 --num_node 3000


python3 SBM_graph.py --graphnum TEST1 --feat_dim 5 --num_node 3000
2877980
python3 SBM_graph.py --graphnum TEST2 --feat_dim 5 --num_node 3000
2879362
python3 SBM_graph.py --graphnum TEST3 --feat_dim 5 --num_node 3000
2876580
python3 SBM_graph.py --graphnum TEST4 --feat_dim 5 --num_node 3000
2876032
python3 SBM_graph.py --graphnum TEST5 --feat_dim 5 --num_node 3000
2876820

python3 SBM_graph.py --graphnum TEST1 --sym --feat_dim 5 --num_node 3000
4317487
python3 SBM_graph.py --graphnum TEST2 --sym --feat_dim 5 --num_node 3000 
4246948
python3 SBM_graph.py --graphnum TEST3 --sym --feat_dim 5 --num_node 3000
4185872
python3 SBM_graph.py --graphnum TEST4 --sym --feat_dim 5 --num_node 3000
4108832
python3 SBM_graph.py --graphnum TEST5 --sym --feat_dim 5 --num_node 3000
4053098


python3 SBM_graph.py --graphnum TEST15 --feat_dim 5 --num_node 3000
2878272
python3 SBM_graph.py --graphnum TEST25 --feat_dim 5 --num_node 3000
2877024
python3 SBM_graph.py --graphnum TEST35 --feat_dim 5 --num_node 3000
2877688
python3 SBM_graph.py --graphnum TEST45 --feat_dim 5 --num_node 3000
2880658

python3 SBM_graph.py --graphnum TEST15 --sym --feat_dim 5 --num_node 3000
4283260
python3 SBM_graph.py --graphnum TEST25 --sym --feat_dim 5 --num_node 3000 
4212450
python3 SBM_graph.py --graphnum TEST35 --sym --feat_dim 5 --num_node 3000
4144266
python3 SBM_graph.py --graphnum TEST45 --sym --feat_dim 5 --num_node 3000
4078562
'''
