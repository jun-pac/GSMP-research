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
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.nn.pytorch.conv.graphconv import EdgeWeightNorm
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv


def normalize(h):
    return (h - h.mean(0)) / h.std(0)

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


class PNY(torch.nn.Module):
    def __init__(self, labels, times, num_time, num_label, split, relative_connectivity):
        super().__init__()
        self.labels = labels
        self.times = times
        self.num_time = num_time
        self.num_label = num_label
        self.split = split
        self.P=relative_connectivity
        self.train_idx = (self.times<self.split).nonzero().squeeze()
        self.test_idx = (self.times>=self.split).nonzero().squeeze()

    def forward(self, prev_x, x):
        clone_x=torch.clone(x)
        previous_cov_matrix=[torch.cov(prev_x[torch.squeeze((self.labels==y).nonzero())].T) for y in range(self.num_label)]

        # Calculate covariances
        current_cov_matrix=torch.zeros((self.num_label, self.num_time, x.shape[1], x.shape[1]))
        for y1 in range(self.num_label):
            for t1 in range(self.num_time):
                denom=0
                for y2 in range(self.num_label):
                    for t2 in range(self.num_time):
                        denom+=self.P[y1][t1][y2][t2]*(2 if abs(t2-t1)>min(self.num_time-1-t1,t1) else 1)
                denom=denom*denom
                for y2 in range(self.num_label):
                    temp=0
                    for t2 in range(self.num_time):
                        temp+=self.P[y1][t1][y2][t2]*(4 if abs(t2-t1)>min(self.num_time-1-t1,t1) else 1)
                    temp=temp/denom
                    current_cov_matrix[y1][t1]+=temp*previous_cov_matrix[y2]
        #print(f"cur cov matrix : {current_cov_matrix[0][0]}")
        #print(f"some Ps {self.P[0][0][0][0]} {self.P[0][4][1][3]} {self.P[1][4][3][2]}")

        # Calculate feature transform matrix
        cov_transform_matrix=torch.zeros((self.num_label, self.num_time, x.shape[1], x.shape[1]))
        for y1 in range(self.num_label):
            L_max, Q_max=torch.linalg.eigh(current_cov_matrix[y1][self.num_time - 1])

            L_max=L_max.real
            Q_max=Q_max.real
            L_max_sq=torch.sqrt(L_max)
            L_max_sq=torch.diag(L_max_sq)
            for t1 in range(self.split):
                L, Q=torch.linalg.eigh(current_cov_matrix[y1][t1])
                L=L.real
                Q=Q.real
                L_sq=torch.sqrt(L)
                Inv_L_sq=1/L_sq
                Inv_L_sq=torch.diag(Inv_L_sq)
                cov_transform_matrix[y1][t1]=Q_max@L_max_sq@Inv_L_sq@Q.T

        # Calculate feature means
        train_mean = torch.zeros((self.num_time, self.num_label, x.shape[1]), requires_grad=False)
        train_cnt = torch.zeros((self.num_time, self.num_label), requires_grad=False)
        for u in range(x.shape[0]):
            t=self.times[u]
            train_cnt[t][self.labels[u]] += 1
            train_mean[t][self.labels[u]] += x[u]                 
        for t in range(self.num_time):
            for l in range(self.num_label):
                train_mean[t][l]=train_mean[t][l]/max(1,train_cnt[t][l])

        # Update modified vals
        for u in self.train_idx:
            t = self.times[u]
            y = self.labels[u]
            clone_x[u] = cov_transform_matrix[y][t]@(x[u]-train_mean[t][y])+train_mean[t][y]
        return clone_x

class GATConv_JJ(nn.Module):
    def __init__(
        self,
        labels, 
        times, 
        num_time, 
        num_label, 
        split, 
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.JJ_Norm=JJ_Norm(labels, times, num_time, num_label, split)

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False
            )
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_r = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias
                )
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(
                th.FloatTensor(size=(num_heads * out_feats,))
            )
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):

        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
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

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(),
                    ) + dst_prefix_shape[1:]
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
                    1, self._num_heads, 1
                ).transpose(0, 2)
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            rst = self.JJ_Norm(rst)

            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                rst = rst + resval
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)),
                    self._num_heads,
                    self._out_feats
                )
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst
            
class GATConv_JJ(nn.Module):
    def __init__(
        self,
        labels, 
        times, 
        num_time, 
        num_label, 
        split, 
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.JJ_Norm=JJ_Norm(labels, times, num_time, num_label, split)

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False
            )
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_r = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias
                )
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(
                th.FloatTensor(size=(num_heads * out_feats,))
            )
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):

        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
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

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(),
                    ) + dst_prefix_shape[1:]
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
                    1, self._num_heads, 1
                ).transpose(0, 2)
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            rst = self.JJ_Norm(rst)

            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                rst = rst + resval
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)),
                    self._num_heads,
                    self._out_feats
                )
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst
            

class GATConv_PNY(nn.Module):
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
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.PNY_Norm=PNY(labels, times, num_time, num_label, split, relative_connectivity)

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False
            )
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_r = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias
                )
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(
                th.FloatTensor(size=(num_heads * out_feats,))
            )
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):

        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        print(f"Num feat of init feat : {feat} {feat.shape}")
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

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(),
                    ) + dst_prefix_shape[1:]
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            print(f"Num feat of rst : {feat_src} {feat_src.shape}")
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
                    1, self._num_heads, 1
                ).transpose(0, 2)
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            print(f"Num feat of rst : {rst} {rst.shape}")
            # rst = self.PNY_Norm(prev_x,rst)

            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                rst = rst + resval
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)),
                    self._num_heads,
                    self._out_feats
                )
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst



class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            GATConv(in_size, hid_size, 1, activation=F.relu)
        )
        self.layers.append(GATConv(hid_size, out_size,1 ))

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        return h



class GAT_JJ(nn.Module):
    def __init__(self, labels, times, num_time, num_label, split, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            GATConv_JJ(labels, times, num_time, num_label, split, in_size, hid_size, 1, activation=F.relu)
        )
        self.layers.append(GATConv_JJ(labels, times, num_time, num_label, split ,hid_size, out_size, 1))
        
    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        return h



class GAT_PNY(nn.Module):
    def __init__(self, labels, times, num_time, num_label, split, relative_connectivity, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            GATConv_PNY(labels, times, num_time, num_label, split, relative_connectivity, in_size, hid_size, 1, activation=F.relu)
        )
        self.layers.append(GATConv_PNY(labels, times, num_time, num_label, split, relative_connectivity, hid_size, out_size, 1))
        
    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        return h
