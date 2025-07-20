"""Torch Module for Simplifying Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
import torch
from torch.nn import init

from torch import nn
import dgl
from dgl import function as fn
from dgl import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.conv.graphconv import EdgeWeightNorm
import torch.nn.functional as F
import dgl.nn as dglnn
from JJ_Norm import JJ_Norm
from PNY import PNY


class APPNPConv_PNY(nn.Module):
    def __init__(self, labels, times, num_time, num_label, split, relative_connectivity, k, alpha, edge_drop=0.0):
        super().__init__()
        self._k = k
        self._alpha = alpha
        self.edge_drop = nn.Dropout(edge_drop)
        self.PNY_Norm=PNY(labels, times, num_time, num_label, split, relative_connectivity)

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if edge_weight is None:
                src_norm = th.pow(
                    graph.out_degrees().to(feat).clamp(min=1), -0.5
                )
                shp = src_norm.shape + (1,) * (feat.dim() - 1)
                src_norm = th.reshape(src_norm, shp).to(feat.device)
                dst_norm = th.pow(
                    graph.in_degrees().to(feat).clamp(min=1), -0.5
                )
                shp = dst_norm.shape + (1,) * (feat.dim() - 1)
                dst_norm = th.reshape(dst_norm, shp).to(feat.device)
            else:
                edge_weight = EdgeWeightNorm("both")(graph, edge_weight)
            feat_0 = feat
            for _ in range(self._k):
                # normalization by src node
                if edge_weight is None:
                    feat = feat * src_norm
                graph.ndata["h"] = feat
                prev_x=torch.clone(feat)
                w = (
                    th.ones(graph.num_edges(), 1)
                    if edge_weight is None
                    else edge_weight
                )
                # print(f"prev_x.shape : {prev_x.shape}")
                graph.edata["w"] = self.edge_drop(w).to(feat.device)
                graph.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "h"))
                feat = graph.ndata.pop("h")
                feat=self.PNY_Norm(prev_x,feat)
                # normalization by dst node
                if edge_weight is None:
                    feat = feat * dst_norm
                feat = (1 - self._alpha) * feat + self._alpha * feat_0
            return feat




class APPNP(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layer = dglnn.APPNPConv(2,0.5)
        self.fc1 = nn.Linear(in_size, hid_size, bias=True)
        self.activate=nn.ReLU()
        self.fc2 = nn.Linear(hid_size, out_size, bias=True)

    def forward(self, g, features):
        h = features
        h = self.layer(g, h)
        h = self.fc1(h)
        h = self.activate(h)
        h = self.fc2(h)
        h=torch.sigmoid(h)
        return h



class APPNP_JJ(nn.Module):
    def __init__(self, labels, times, num_time, num_label, split, in_size, hid_size, out_size):
        super().__init__()
        self.layer = dglnn.APPNPConv(2,0.5)
        self.JJ_Norm=JJ_Norm(labels, times, num_time, num_label, split)
        self.fc1 = nn.Linear(in_size, hid_size, bias=True)
        self.activate=nn.ReLU()
        self.fc2 = nn.Linear(hid_size, out_size, bias=True)
        

    def forward(self, g, features):
        h = features
        h = self.layer(g, h)
        h = self.JJ_Norm(h)
        h = self.fc1(h)
        h = self.activate(h)
        h = self.fc2(h)
        h=torch.sigmoid(h)
        return h


class APPNP_PNY(nn.Module):
    def __init__(self, labels, times, num_time, num_label, split, relative_connectivity, in_size, hid_size, out_size):
        super().__init__()
        self.layer = APPNPConv_PNY(labels, times, num_time, num_label, split, relative_connectivity,2,0.5)
        self.fc1 = nn.Linear(in_size, hid_size, bias=True)
        self.activate=nn.ReLU()
        self.fc2 = nn.Linear(hid_size, out_size, bias=True)

    def forward(self, g, features):
        h = features
        h = self.layer(g, h)
        h = self.fc1(h)
        h = self.activate(h)
        h = self.fc2(h)
        h=torch.sigmoid(h)
        return h
