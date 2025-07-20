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
core_num=multiprocessing.cpu_count()
len_newsrc=0

t0=time.time()

Z={}
Z['T']=4
Z['TM']=4
Z['TN']=4
Z['TC']=16
Z['TL']=4
Z['TEST']=32

P={}
P['T']=(0.04,0.01333)
P['TM']=(0.12,0.04)
P['TN']=(0.12,0.05333)
P['TC']=(0.12,0.04)
P['TL']=(0.06,0.05333)
P['TEST']=(0.4,0.1333)



# Define SBM probability matrices
matrix_func={}

matrix_func['T1']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.01333+0.00*(args.num_time-delta_t)/args.num_time)
matrix_func['T2']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.011667+0.0025*(args.num_time-delta_t)/args.num_time)
matrix_func['T3']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.01000+0.005*(args.num_time-delta_t)/args.num_time)
matrix_func['T4']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.008333+0.0075*(args.num_time-delta_t)/args.num_time)
matrix_func['T5']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.00666+0.01*(args.num_time-delta_t)/args.num_time)
matrix_func['T6']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.005000+0.0125*(args.num_time-delta_t)/args.num_time)
matrix_func['T7']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.00333+0.015*(args.num_time-delta_t)/args.num_time)
matrix_func['T8']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.001667+0.0175*(args.num_time-delta_t)/args.num_time)
matrix_func['T9']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.00000+0.02*(args.num_time-delta_t)/args.num_time)

matrix_func['TM1']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.04+0.00*(args.num_time-delta_t)/args.num_time)
matrix_func['TM2']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.035+0.0075*(args.num_time-delta_t)/args.num_time)
matrix_func['TM3']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.03+0.015*(args.num_time-delta_t)/args.num_time)
matrix_func['TM4']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.025+0.0225*(args.num_time-delta_t)/args.num_time)
matrix_func['TM5']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.02+0.03*(args.num_time-delta_t)/args.num_time)
matrix_func['TM6']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.015+0.0375*(args.num_time-delta_t)/args.num_time)
matrix_func['TM7']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.01+0.045*(args.num_time-delta_t)/args.num_time)
matrix_func['TM8']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.005+0.0525*(args.num_time-delta_t)/args.num_time)
matrix_func['TM9']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.00+0.06*(args.num_time-delta_t)/args.num_time)

matrix_func['TN1']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.05333+0.00*(args.num_time-delta_t)/args.num_time)
matrix_func['TN2']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.046667+0.01*(args.num_time-delta_t)/args.num_time)
matrix_func['TN3']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.04000+0.02*(args.num_time-delta_t)/args.num_time)
matrix_func['TN4']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.033333+0.03*(args.num_time-delta_t)/args.num_time)
matrix_func['TN5']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.02666+0.04*(args.num_time-delta_t)/args.num_time)
matrix_func['TN6']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.020000+0.05*(args.num_time-delta_t)/args.num_time)
matrix_func['TN7']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.01333+0.06*(args.num_time-delta_t)/args.num_time)
matrix_func['TN8']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.006667+0.07*(args.num_time-delta_t)/args.num_time)
matrix_func['TN9']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.00000+0.08*(args.num_time-delta_t)/args.num_time)

matrix_func['TC1']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.04+0.00*(args.num_time-delta_t)/args.num_time)
matrix_func['TC2']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.035+0.0075*(args.num_time-delta_t)/args.num_time)
matrix_func['TC3']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.03+0.015*(args.num_time-delta_t)/args.num_time)
matrix_func['TC4']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.025+0.0225*(args.num_time-delta_t)/args.num_time)
matrix_func['TC5']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.02+0.03*(args.num_time-delta_t)/args.num_time)
matrix_func['TC6']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.015+0.0375*(args.num_time-delta_t)/args.num_time)
matrix_func['TC7']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.01+0.045*(args.num_time-delta_t)/args.num_time)
matrix_func['TC8']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.005+0.0525*(args.num_time-delta_t)/args.num_time)
matrix_func['TC9']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.00+0.06*(args.num_time-delta_t)/args.num_time)

matrix_func['TL1']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.05333+0.00*(args.num_time-delta_t)/args.num_time)
matrix_func['TL2']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.046667+0.01*(args.num_time-delta_t)/args.num_time)
matrix_func['TL3']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.04000+0.02*(args.num_time-delta_t)/args.num_time)
matrix_func['TL4']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.033333+0.03*(args.num_time-delta_t)/args.num_time)
matrix_func['TL5']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.02666+0.04*(args.num_time-delta_t)/args.num_time)
matrix_func['TL6']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.020000+0.05*(args.num_time-delta_t)/args.num_time)
matrix_func['TL7']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.01333+0.06*(args.num_time-delta_t)/args.num_time)
matrix_func['TL8']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.006667+0.07*(args.num_time-delta_t)/args.num_time)
matrix_func['TL9']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.00000+0.08*(args.num_time-delta_t)/args.num_time)


matrix_func['TEST1']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.1333+0.0*(args.num_time-delta_t)/args.num_time)
matrix_func['TEST3']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.1000+0.05*(args.num_time-delta_t)/args.num_time)
matrix_func['TEST5']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.0666+0.1*(args.num_time-delta_t)/args.num_time)
matrix_func['TEST7']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.0333+0.15*(args.num_time-delta_t)/args.num_time)
matrix_func['TEST9']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.0000+0.2*(args.num_time-delta_t)/args.num_time)
matrix_func['TEST2']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.11667+0.025*(args.num_time-delta_t)/args.num_time)
matrix_func['TEST4']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.08333+0.075*(args.num_time-delta_t)/args.num_time)
matrix_func['TEST6']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.05000+0.125*(args.num_time-delta_t)/args.num_time)
matrix_func['TEST8']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.01667+0.175*(args.num_time-delta_t)/args.num_time)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphnum", type=str, default='3')
    parser.add_argument("--root",type=str, default='./data/')
    parser.add_argument("--feat_dim", type=int, default=5)
    parser.add_argument("--num_label", type=int, default=10)
    parser.add_argument("--num_node", type=int, default=3000)
    parser.add_argument("--num_time", type=int, default=100)
    parser.add_argument("--num_trial", type=int, default=1)
    args = parser.parse_args()

    print(f"Number of cores... {core_num}")

    # Init names
    graph_name=args.graphnum 
    name=graph_name
    if(args.feat_dim!=5):
        name=name+"_feat"+str(args.feat_dim)
    if(args.num_node!=3000):
        name=name+"_node"+str(args.num_node)
    if(args.num_label!=10):
        name=name+"_label"+str(args.num_label)
    assert(args.num_time==100)

    print(f"Your graph's name : {name}")
    f_log=open('./etalog/0316'+'.txt','a')
    e_log=open('./etalog/onlyeta'+'.txt','a')
    m_log=open('./etalog/onlymu'+'.txt','a')

    matrix=matrix_func[graph_name]
    
    # calculate eta [Theoretical]
    # Assume uniform t1 distribution in both train and test set.
    # Assume uniform (t2,y2) distribution among total nodes
    M_train_density=np.zeros((args.num_label,args.num_label),dtype=np.float64)
    M_test_density=np.zeros((args.num_label,args.num_label),dtype=np.float64)
    for y1 in tqdm(range(args.num_label)):
        train_density_sum=0
        test_density_sum=0
        for y2 in range(args.num_label):
            for t1 in range(0,55):
                for t2 in range(100):
                    M_train_density[y1][y2]+=matrix(abs(t1-t2),y1,y2)
            train_density_sum+=M_train_density[y1][y2]
            for t1 in range(60,100):
                for t2 in range(100):
                    M_test_density[y1][y2]+=matrix(abs(t1-t2),y1,y2)
            test_density_sum+=M_test_density[y1][y2]
        for y2 in range(args.num_label):
            M_train_density[y1][y2]/=train_density_sum
            M_test_density[y1][y2]/=test_density_sum
    

    print("Train density")
    for y1 in range(args.num_label):
        for y2 in range(args.num_label):
            print(f"{M_train_density[y1][y2]:.3f}",end=' ')
        print()
    print("=================")
    print("Test density")
    for y1 in range(args.num_label):
        for y2 in range(args.num_label):
            print(f"{M_test_density[y1][y2]:.3f}",end=' ')
        print()
    print("=================")

    feat_center=np.zeros([args.num_label,args.feat_dim])
    sum_mu=0
    for tt in tqdm(range(args.num_trial)):
        M_train=np.zeros((args.num_label,args.feat_dim),dtype=np.float64)
        M_test=np.zeros((args.num_label,args.feat_dim),dtype=np.float64)    
        for i in range(args.num_label):
            feat_center[i]=np.random.randn(args.feat_dim)

        for y1 in range(args.num_label):
            for y2 in range(args.num_label):
                M_train[y1]=M_train[y1]+M_train_density[y1][y2]*feat_center[y2]
                M_test[y1]=M_test[y1]+M_test_density[y1][y2]*feat_center[y2]
            
        # Assume uniform test label distribution
        for y1 in range(args.num_label):
            sum_mu+=np.linalg.norm(M_train[y1]-M_test[y1])/args.num_label
    delta_mu=sum_mu/args.num_trial

    print(f"Theoretical delta_mu for {name}, {args.num_trial} repeat : {delta_mu:.8f}\n")
    f_log.write(f"Theoretical delta_mu {name}, {args.num_trial} repeat : {delta_mu:.8f}\n")
    f_log.flush()
    m_log.write(f"{delta_mu:.8f}\n")
    m_log.flush()

    print(f"graph_name[:-1] : {graph_name[:-1]}")
    std=Z[graph_name[:-1]]*np.sqrt(args.feat_dim)
    avg_deg=args.num_node * (P[graph_name[:-1]][0]*1/args.num_label+P[graph_name[:-1]][1]*(args.num_label-1)/args.num_label)
    M_std=std/np.sqrt(avg_deg)


    eta=delta_mu/M_std

    print(f"Theoretical eta for {name} : {eta:.8f}\n")
    f_log.write(f"Theoretical eta {name} : {eta:.8f}\n")
    f_log.flush()
    e_log.write(f"{eta:.8f}\n")
    e_log.flush()

# python SBM_graph_eta.py --graphnum TM1 
