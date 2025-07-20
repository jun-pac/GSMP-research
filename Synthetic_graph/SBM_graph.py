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


# Define Sample Functions 
sample_func={}

sample_func['T1']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['T2']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['T3']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['T4']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['T5']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['T6']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['T7']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['T8']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['T9']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]

sample_func['TM1']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TM2']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TM3']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TM4']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TM5']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TM6']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TM7']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TM8']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TM9']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]

sample_func['TN1']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TN2']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TN3']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TN4']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TN5']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TN6']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TN7']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TN8']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TN9']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]

sample_func['TC1']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]
sample_func['TC2']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]
sample_func['TC3']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]
sample_func['TC4']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]
sample_func['TC5']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]
sample_func['TC6']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]
sample_func['TC7']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]
sample_func['TC8']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]
sample_func['TC9']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]

sample_func['TL1']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TL2']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TL3']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TL4']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TL5']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TL6']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TL7']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TL8']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]
sample_func['TL9']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]

sample_func['TEST1']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['TEST2']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['TEST3']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['TEST4']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['TEST5']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['TEST6']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['TEST7']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['TEST8']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]
sample_func['TEST9']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]

sample_func['TD1']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TD2']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TD3']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TD4']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TD5']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TD6']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TD7']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TD8']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TD9']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2

sample_func['TDM1']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TDM2']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TDM3']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TDM4']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TDM5']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TDM6']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TDM7']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TDM8']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TDM9']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2

sample_func['TDN1']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TDN2']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TDN3']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TDN4']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TDN5']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TDN6']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TDN7']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TDN8']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
sample_func['TDN9']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2

sample_func['TDC1']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]*2
sample_func['TDC2']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]*2
sample_func['TDC3']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]*2
sample_func['TDC4']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]*2
sample_func['TDC5']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]*2
sample_func['TDC6']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]*2
sample_func['TDC7']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]*2
sample_func['TDC8']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]*2
sample_func['TDC9']=lambda cur_label:torch.randn(args.feat_dim)*16+feat_center[cur_label]*2

# sample_func['TDL1']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
# sample_func['TDL2']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
# sample_func['TDL3']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
# sample_func['TDL4']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
# sample_func['TDL5']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
# sample_func['TDL6']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
# sample_func['TDL7']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
# sample_func['TDL8']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2
# sample_func['TDL9']=lambda cur_label:torch.randn(args.feat_dim)*4+feat_center[cur_label]*2

sample_func['TESTD1']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]*2
sample_func['TESTD2']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]*2
sample_func['TESTD3']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]*2
sample_func['TESTD4']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]*2
sample_func['TESTD5']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]*2
sample_func['TESTD6']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]*2
sample_func['TESTD7']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]*2
sample_func['TESTD8']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]*2
sample_func['TESTD9']=lambda cur_label:torch.randn(args.feat_dim)*32+feat_center[cur_label]*2



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
matrix_func['TEST2']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.11667+0.025*(args.num_time-delta_t)/args.num_time)
matrix_func['TEST3']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.1000+0.05*(args.num_time-delta_t)/args.num_time)
matrix_func['TEST4']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.08333+0.075*(args.num_time-delta_t)/args.num_time)
matrix_func['TEST5']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.0666+0.1*(args.num_time-delta_t)/args.num_time)
matrix_func['TEST6']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.05000+0.125*(args.num_time-delta_t)/args.num_time)
matrix_func['TEST7']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.0333+0.15*(args.num_time-delta_t)/args.num_time)
matrix_func['TEST8']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.01667+0.175*(args.num_time-delta_t)/args.num_time)
matrix_func['TEST9']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.0000+0.2*(args.num_time-delta_t)/args.num_time)

matrix_func['TD1']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.01333+0.00*(args.num_time-delta_t)/args.num_time)
matrix_func['TD2']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.011667+0.0025*(args.num_time-delta_t)/args.num_time)
matrix_func['TD3']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.01000+0.005*(args.num_time-delta_t)/args.num_time)
matrix_func['TD4']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.008333+0.0075*(args.num_time-delta_t)/args.num_time)
matrix_func['TD5']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.00666+0.01*(args.num_time-delta_t)/args.num_time)
matrix_func['TD6']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.005000+0.0125*(args.num_time-delta_t)/args.num_time)
matrix_func['TD7']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.00333+0.015*(args.num_time-delta_t)/args.num_time)
matrix_func['TD8']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.001667+0.0175*(args.num_time-delta_t)/args.num_time)
matrix_func['TD9']=lambda delta_t,label_i,label_j:(0.04 if label_i==label_j else 0.00000+0.02*(args.num_time-delta_t)/args.num_time)

matrix_func['TDM1']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.04+0.00*(args.num_time-delta_t)/args.num_time)
matrix_func['TDM2']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.035+0.0075*(args.num_time-delta_t)/args.num_time)
matrix_func['TDM3']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.03+0.015*(args.num_time-delta_t)/args.num_time)
matrix_func['TDM4']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.025+0.0225*(args.num_time-delta_t)/args.num_time)
matrix_func['TDM5']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.02+0.03*(args.num_time-delta_t)/args.num_time)
matrix_func['TDM6']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.015+0.0375*(args.num_time-delta_t)/args.num_time)
matrix_func['TDM7']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.01+0.045*(args.num_time-delta_t)/args.num_time)
matrix_func['TDM8']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.005+0.0525*(args.num_time-delta_t)/args.num_time)
matrix_func['TDM9']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.00+0.06*(args.num_time-delta_t)/args.num_time)

matrix_func['TDN1']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.05333+0.00*(args.num_time-delta_t)/args.num_time)
matrix_func['TDN2']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.046667+0.01*(args.num_time-delta_t)/args.num_time)
matrix_func['TDN3']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.04000+0.02*(args.num_time-delta_t)/args.num_time)
matrix_func['TDN4']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.033333+0.03*(args.num_time-delta_t)/args.num_time)
matrix_func['TDN5']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.02666+0.04*(args.num_time-delta_t)/args.num_time)
matrix_func['TDN6']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.020000+0.05*(args.num_time-delta_t)/args.num_time)
matrix_func['TDN7']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.01333+0.06*(args.num_time-delta_t)/args.num_time)
matrix_func['TDN8']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.006667+0.07*(args.num_time-delta_t)/args.num_time)
matrix_func['TDN9']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.00000+0.08*(args.num_time-delta_t)/args.num_time)

matrix_func['TDC1']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.04+0.00*(args.num_time-delta_t)/args.num_time)
matrix_func['TDC2']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.035+0.0075*(args.num_time-delta_t)/args.num_time)
matrix_func['TDC3']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.03+0.015*(args.num_time-delta_t)/args.num_time)
matrix_func['TDC4']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.025+0.0225*(args.num_time-delta_t)/args.num_time)
matrix_func['TDC5']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.02+0.03*(args.num_time-delta_t)/args.num_time)
matrix_func['TDC6']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.015+0.0375*(args.num_time-delta_t)/args.num_time)
matrix_func['TDC7']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.01+0.045*(args.num_time-delta_t)/args.num_time)
matrix_func['TDC8']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.005+0.0525*(args.num_time-delta_t)/args.num_time)
matrix_func['TDC9']=lambda delta_t,label_i,label_j:(0.12 if label_i==label_j else 0.00+0.06*(args.num_time-delta_t)/args.num_time)

# matrix_func['TDL1']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.05333+0.00*(args.num_time-delta_t)/args.num_time)
# matrix_func['TDL2']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.046667+0.01*(args.num_time-delta_t)/args.num_time)
# matrix_func['TDL3']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.04000+0.02*(args.num_time-delta_t)/args.num_time)
# matrix_func['TDL4']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.033333+0.03*(args.num_time-delta_t)/args.num_time)
# matrix_func['TDL5']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.02666+0.04*(args.num_time-delta_t)/args.num_time)
# matrix_func['TDL6']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.020000+0.05*(args.num_time-delta_t)/args.num_time)
# matrix_func['TDL7']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.01333+0.06*(args.num_time-delta_t)/args.num_time)
# matrix_func['TDL8']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.006667+0.07*(args.num_time-delta_t)/args.num_time)
# matrix_func['TDL9']=lambda delta_t,label_i,label_j:(0.06 if label_i==label_j else 0.00000+0.08*(args.num_time-delta_t)/args.num_time)

matrix_func['TESTD1']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.1333+0.0*(args.num_time-delta_t)/args.num_time)
matrix_func['TESTD2']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.11667+0.025*(args.num_time-delta_t)/args.num_time)
matrix_func['TESTD3']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.1000+0.05*(args.num_time-delta_t)/args.num_time)
matrix_func['TESTD4']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.08333+0.075*(args.num_time-delta_t)/args.num_time)
matrix_func['TESTD5']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.0666+0.1*(args.num_time-delta_t)/args.num_time)
matrix_func['TESTD6']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.05000+0.125*(args.num_time-delta_t)/args.num_time)
matrix_func['TESTD7']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.0333+0.15*(args.num_time-delta_t)/args.num_time)
matrix_func['TESTD8']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.01667+0.175*(args.num_time-delta_t)/args.num_time)
matrix_func['TESTD9']=lambda delta_t,label_i,label_j:(0.4 if label_i==label_j else 0.0000+0.2*(args.num_time-delta_t)/args.num_time)



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
        if abs(times[newsrc[i]]-times[newdst[i]])>min(args.num_time-1-times[newdst[i]],times[newdst[i]]-0):
            addsrc.append(newsrc[i])
            adddst.append(newdst[i])
    return np.array([addsrc, adddst])
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphnum", type=str, default='3')
    parser.add_argument("--root",type=str, default='./data/')
    parser.add_argument("--sym", action='store_true', default=False)
    parser.add_argument("--sym2", action='store_true', default=False)
    parser.add_argument("--mono1", action='store_true', default=False)
    parser.add_argument("--mono2", action='store_true', default=False)
    parser.add_argument("--feat_dim", type=int, default=5)
    parser.add_argument("--num_label", type=int, default=10)
    parser.add_argument("--num_node", type=int, default=3000)
    parser.add_argument("--num_time", type=int, default=100)
    args = parser.parse_args()

    print(f"Number of cores... {core_num}")

    # Init names
    graph_name=args.graphnum 
    name=graph_name+("_sym" if args.sym else "")
    name=name+("_sym2" if args.sym2 else "")
    name=name+("_mono1" if args.mono1 else "")
    name=name+("_mono2" if args.mono2 else "")
    if(args.feat_dim!=5):
        name=name+"_feat"+str(args.feat_dim)
    if(args.num_node!=3000):
        name=name+"_node"+str(args.num_node)
    if(args.num_label!=10):
        name=name+"_label"+str(args.num_label)
    if(args.num_time!=100):
        name=name+"_time"+str(args.num_time)
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
        label[i]=label_rands[i]%args.num_label
        times[i]=time_rands[i]%args.num_time

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

    # if(args.mono1 or args.mono2):
    #     len_newsrc=len(newsrc)
    #     with Pool(core_num) as p:
    #         result = p.map(mono_task, range(0,len(newsrc),sym_chunk_size))
    #     src_dst = np.concatenate(result, axis=1)
    #     addsrc, adddst = src_dst
    #     newsrc=addsrc
    #     newdst=adddst
    #     if(args.mono2):
    #         newsrc,newdst = newdst,newsrc

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
    

    feat_center=torch.zeros([args.num_label,args.feat_dim])
    for i in range(args.num_label):
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

    save_graphs(name,[new_g],glabel)
    print(f"Successfully saved!... {time.time()-t0}sec")

