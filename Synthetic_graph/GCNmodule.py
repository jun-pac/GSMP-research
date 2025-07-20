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


class GraphConv_PNY(nn.Module):
    def __init__(
        self,
        labels, 
        times, 
        num_time, 
        num_label, 
        split, 
        relative_connectivity,
        in_feats,
        out_feats,
        norm="right",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
    ):
        super().__init__()
        if norm not in ("none", "both", "right", "left"):
            raise DGLError(
                'Invalid norm value. Must be either "none", "both", "right" or "left".'
                ' But got "{}".'.format(norm)
            )
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.PNY_Norm=PNY(labels, times, num_time, num_label, split, relative_connectivity)

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None):
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
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == "both":
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight = self.weight

            prev_x=torch.clone(feat_src)
            graph.srcdata["h"] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
            rst = graph.dstdata["h"]

            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == "both":
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm
            rst = self.PNY_Norm(prev_x,rst)

            if weight is not None:
                rst = th.matmul(rst, weight)
            
            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = "in={_in_feats}, out={_out_feats}"
        summary += ", normalization={_norm}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)



class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu, norm="right")
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size, norm="right")) # Self connection?
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h



class GCN_JJ(nn.Module):
    def __init__(self, labels, times, num_time, num_label, split, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu, norm="right")
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size, norm="right")) # Self connection?
        self.dropout = nn.Dropout(0.5)
        self.JJ_Norm=JJ_Norm(labels, times, num_time, num_label, split)


    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            h = self.JJ_Norm(h)
        return h


class GCN_PNY(nn.Module):
    def __init__(self, labels, times, num_time, num_label, split, relative_connectivity, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            GraphConv_PNY(labels, times, num_time, num_label, split, relative_connectivity, in_size, hid_size, activation=F.relu, norm="left")
        )
        self.layers.append(GraphConv_PNY(labels, times, num_time, num_label, split, relative_connectivity, hid_size, out_size, norm="left")) # Self connection?
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

