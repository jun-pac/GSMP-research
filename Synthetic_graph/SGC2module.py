"""Torch Module for Simplifying Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
import torch

from torch import nn
import dgl
from dgl import function as fn
from dgl import DGLError
from dgl.nn.pytorch.conv.graphconv import EdgeWeightNorm

from JJ_Norm import JJ_Norm
from PNY import PNY


class SGC2onv(nn.Module):
    def __init__(
        self,
        in_feats,
        hid_feats,
        out_feats,
        k=1,
        cached=False,
        bias=True,
        norm=None,
        allow_zero_in_degree=False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, hid_feats, bias=True)
        self.activate=nn.ReLU()
        self.fc2 = nn.Linear(hid_feats, out_feats, bias=True)
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            msg_func = fn.copy_u("h", "m")
            if edge_weight is not None:
                graph.edata["_edge_weight"] = EdgeWeightNorm("both")(
                    graph, edge_weight
                )
                msg_func = fn.u_mul_e("h", "_edge_weight", "m")

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                if edge_weight is None:
                    # compute normalization
                    degs = graph.in_degrees().to(feat).clamp(min=1)
                    norm = th.pow(degs, -0.5)
                    norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    if edge_weight is None:
                        feat = feat * norm
                    graph.ndata["h"] = feat
                    graph.update_all(msg_func, fn.sum("m", "h"))
                    feat = graph.ndata.pop("h")
                    if edge_weight is None:
                        feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            h=feat
            h = self.fc1(h)
            h = self.activate(h)
            h = self.fc2(h)
            h=torch.sigmoid(h)
            return h



class SGC2onv_JJ(nn.Module):
    def __init__(
        self,
        labels, 
        times, 
        num_time, 
        num_label, 
        split,
        in_feats,
        hid_feats,
        out_feats,
        k=1,
        cached=False,
        bias=True,
        norm=None,
        allow_zero_in_degree=False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, hid_feats, bias=True)
        self.activate=nn.ReLU()
        self.fc2 = nn.Linear(hid_feats, out_feats, bias=True)
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.reset_parameters()
        self.JJ_Norm=JJ_Norm(labels, times, num_time, num_label, split)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            msg_func = fn.copy_u("h", "m")
            if edge_weight is not None:
                graph.edata["_edge_weight"] = EdgeWeightNorm("both")(
                    graph, edge_weight
                )
                msg_func = fn.u_mul_e("h", "_edge_weight", "m")

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                if edge_weight is None:
                    # compute normalization
                    degs = graph.in_degrees().to(feat).clamp(min=1)
                    norm = th.pow(degs, -0.5)
                    norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    if edge_weight is None:
                        feat = feat * norm
                    graph.ndata["h"] = feat
                    graph.update_all(msg_func, fn.sum("m", "h"))
                    feat = graph.ndata.pop("h")
                    if edge_weight is None:
                        feat = feat * norm
                feat=self.JJ_Norm(feat)
                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            
            h=feat
            h = self.fc1(h)
            h = self.activate(h)
            h = self.fc2(h)
            h=torch.sigmoid(h)
            return h



class SGC2onv_PNY(nn.Module):
    def __init__(
        self,
        labels, 
        times, 
        num_time, 
        num_label, 
        split,
        relative_connectivity,
        in_feats,
        hid_feats,
        out_feats,
        k=1,
        cached=False,
        bias=True,
        norm=None,
        allow_zero_in_degree=False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, hid_feats, bias=True)
        self.activate=nn.ReLU()
        self.fc2 = nn.Linear(hid_feats, out_feats, bias=True)
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.reset_parameters()
        self.PNY_Norm=PNY(labels, times, num_time, num_label, split, relative_connectivity)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            msg_func = fn.copy_u("h", "m")
            if edge_weight is not None:
                graph.edata["_edge_weight"] = EdgeWeightNorm("both")(
                    graph, edge_weight
                )
                msg_func = fn.u_mul_e("h", "_edge_weight", "m")

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                if edge_weight is None:
                    # compute normalization
                    degs = graph.in_degrees().to(feat).clamp(min=1)
                    norm = th.pow(degs, -0.5)
                    norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    if edge_weight is None:
                        feat = feat * norm
                    graph.ndata["h"] = feat
                    prev_x=torch.clone(feat)
                    graph.update_all(msg_func, fn.sum("m", "h"))
                    feat = graph.ndata.pop("h")
                    feat = self.PNY_Norm(prev_x,feat)

                    if edge_weight is None:
                        feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            
            h=feat
            h = self.fc1(h)
            h = self.activate(h)
            h = self.fc2(h)
            h=torch.sigmoid(h)
            return h
