"""Torch Module for Simplifying Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
import torch

from torch import nn
import dgl
from dgl import function as fn
from dgl import DGLError
from dgl.nn.pytorch.conv.graphconv import EdgeWeightNorm

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

        # Calculate feature transform matrix
        cov_transform_matrix=torch.zeros((self.num_label, self.num_time, x.shape[1], x.shape[1]))
        for y1 in range(self.num_label):
            L_max, Q_max=torch.linalg.eigh(current_cov_matrix[y1][self.num_time - 1])
            L_max=L_max.real
            Q_max=Q_max.real
            L_max_sq=torch.sqrt(L_max)
            L_max_sq=torch.diag(L_max_sq)
            for t1 in range(self.num_time):
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



class SGConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        k=1,
        cached=False,
        bias=True,
        norm=normalize,
        allow_zero_in_degree=False,
    ):
        super(SGConv, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

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
            return self.fc(feat)



class SGConv_JJ(nn.Module):
    def __init__(
        self,
        labels, 
        times, 
        num_time, 
        num_label, 
        split,
        in_feats,
        out_feats,
        k=1,
        cached=False,
        bias=True,
        norm=normalize,
        allow_zero_in_degree=False,
    ):
        super().__init__()
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.reset_parameters()
        self.JJ_Norm=JJ_Norm(labels, times, num_time, num_label, split)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

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
            
            return self.fc(feat)



class SGConv_PNY(nn.Module):
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
        k=1,
        cached=False,
        bias=True,
        norm=normalize,
        allow_zero_in_degree=False,
    ):
        super().__init__()
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.reset_parameters()
        self.PNY_Norm=PNY(labels, times, num_time, num_label, split, relative_connectivity)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

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
            
            return self.fc(feat)
