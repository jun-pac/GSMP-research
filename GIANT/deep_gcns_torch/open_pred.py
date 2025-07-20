#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import os
import random
import time
import logging
import uuid
import sys
import gc
from functools import reduce
import operator as op

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from examples.ogb_eff.ogbn_arxiv_dgl.loss import loss_kd_only
from examples.ogb_eff.ogbn_arxiv_dgl.model_rev import RevGAT


dataset = "ogbn-arxiv"
name = "./output_ori/test1_o/1.pt"
evaluator = Evaluator(name=dataset)
evaluator_wrapper = lambda pred, labels: evaluator.eval(
    {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
)["acc"]



labels=torch.load("./saved_labels")
train_idx=torch.load("./saved_train_idx")
val_idx=torch.load("./saved_val_idx")
test_idx=torch.load("./saved_test_idx")
print(f"labels.shape : {labels.shape}")
print(f"train_idx.shape : {train_idx.shape}")
print(f"val_idx.shape : {val_idx.shape}")
print(f"test_idx.shape : {test_idx.shape}")


logit=torch.load(name, map_location=torch.device('cpu'))
pred=F.softmax(logit, dim=1)
pred=logit
print(f"type(pred) ; {type(pred)}")
print(f"pred.shape ; {pred.shape}")
print(f"pred[0] ; {pred[0]}")
print(torch.sum(pred[0]))
print(torch.sum(pred[1]))

print(f"train acc: {evaluator_wrapper(pred[train_idx], labels[train_idx])}")
print(f"valid acc: {evaluator_wrapper(pred[val_idx], labels[val_idx])}")
print(f"test acc: {evaluator_wrapper(pred[test_idx], labels[test_idx])}")
# conda activate giant

print("okay next")
print()
#============================================================================

dname = "./output_ori/test1_o/" # directory
directory = os.fsencode(dname)
    

sumpred=torch.zeros(pred.shape)
cnt=0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".pt"): 
        print(dname, filename)
        name=dname+'/'+filename
        logit=torch.load(name, map_location=torch.device('cpu'))
        pred=F.softmax(logit, dim=1)
        pred=logit
        cnt+=1
        print(f"{cnt} th model")
        print(f"train acc: {evaluator_wrapper(pred[train_idx], labels[train_idx])}")
        print(f"valid acc: {evaluator_wrapper(pred[val_idx], labels[val_idx])}")
        print(f"test acc: {evaluator_wrapper(pred[test_idx], labels[test_idx])}")
        sumpred=pred+sumpred
    else:
        continue

print()
print(f"======== Ensembled result ========")
print(f"train acc: {evaluator_wrapper(sumpred[train_idx], labels[train_idx])}")
print(f"valid acc: {evaluator_wrapper(sumpred[val_idx], labels[val_idx])}")
print(f"test acc: {evaluator_wrapper(sumpred[test_idx], labels[test_idx])}")