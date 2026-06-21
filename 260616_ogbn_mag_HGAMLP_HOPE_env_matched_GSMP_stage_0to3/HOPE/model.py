import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MLP(nn.Module):
    def __init__(self, in_feats, hidden, nclass, num_layers, dropout=0.5, bns=True, norm='batch'):
        super(MLP, self).__init__()
        
        layers = []
        if num_layers == 1:
            # 如果只有一层，就是一个简单的线性层
            layers.append(nn.Linear(in_feats, nclass, bias=True))
        else:
            # 输入层
            layers.append(nn.Linear(in_feats, hidden, bias=True))
            if bns:
                layers.append(nn.LayerNorm(hidden) if norm == 'layer' else nn.BatchNorm1d(hidden))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(dropout))
            
            # 中间隐藏层
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden, hidden, bias=True))
                if bns:
                    layers.append(nn.LayerNorm(hidden) if norm == 'layer' else nn.BatchNorm1d(hidden))
                layers.append(nn.PReLU())
                layers.append(nn.Dropout(dropout))
            
            # 输出层
            layers.append(nn.Linear(hidden, nclass, bias=True))
            
        self.layers = nn.Sequential(*layers)

    def reset_parameters(self):
        # 这个函数保持不变，是正确的
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)

class Transformer(nn.Module):
    def __init__(self, n_channels, att_drop=0., act='none', num_heads=1):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.query.weight, gain=gain)
        xavier_uniform_(self.key.weight, gain=gain)
        xavier_uniform_(self.value.weight, gain=gain)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, x, mask=None):
        B, M, C = x.size() # batchsize, num_metapaths, channels
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))

        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1) # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h) # [B, H, M, -1]
        return o.permute(0,2,1,3).reshape((B, M, C)) + x

class Conv1d1x1(nn.Module):
    def __init__(self, cin, cout, groups, bias=True, cformat='channel-first'):
        super(Conv1d1x1, self).__init__()
        self.cin = cin
        self.cout = cout
        self.groups = groups
        self.cformat = cformat
        if not bias:
            self.bias = None
        if self.groups == 1: # different keypoints share same kernel
            self.W = nn.Parameter(torch.randn(self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(1, self.cout))
        else:
            self.W = nn.Parameter(torch.randn(self.groups, self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.groups, self.cout))

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.groups == 1:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,mn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,mn->bnc', x, self.W) + self.bias.T
            else:
                assert False
        else:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,cmn->bnc', x, self.W) + self.bias.T
            else:
                assert False

class L2Norm(nn.Module):

    def __init__(self, dim):
        super(L2Norm, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class MOE(nn.Module):
    def __init__(self, dataset, data_size, nfeat, hidden, nclass,
                 num_feats, num_label_feats, tgt_key,
                 dropout, input_drop, att_drop, label_drop,
                 n_layers_1, n_layers_2, n_layers_3,
                 act, residual=False, bns=False, label_bns=False,
                 label_residual=True, num_experts=10, aggregation="SeHGNN", similarity_threshold = 0.2, 
                 lower_bound: float = 0.5, 
                 upper_bound: float = 3):
        super(MOE, self).__init__()
        self.dataset = dataset
        self.tgt_key = tgt_key
        self.hidden = hidden
        self.num_experts = num_experts
        self.nclass = nclass
        self.aggregation = aggregation
        self.residual = residual
        self.label_residual = label_residual

        if any([v != nfeat for k, v in data_size.items()]):
            self.embedings = nn.ParameterDict({})
            for k, v in data_size.items():
                if v != nfeat:
                    self.embedings[k] = nn.Parameter(
                        torch.Tensor(v, nfeat).uniform_(-0.5, 0.5))
        else:
            self.embedings = None

        self.feat_project_layers = nn.Sequential(
            Conv1d1x1(nfeat, hidden, num_feats, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_feats, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
            Conv1d1x1(hidden, hidden, num_feats, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_feats, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        if num_label_feats > 0:
            self.label_feat_project_layers = nn.Sequential(
                Conv1d1x1(nclass, hidden, num_label_feats, bias=True, cformat='channel-first'),
                nn.LayerNorm([num_label_feats, hidden]),
                nn.PReLU(),
                nn.Dropout(dropout),
                Conv1d1x1(hidden, hidden, num_label_feats, bias=True, cformat='channel-first'),
                nn.LayerNorm([num_label_feats, hidden]),
                nn.PReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.label_feat_project_layers = None
        
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.res_fc = nn.Linear(nfeat, hidden, bias=False)
        self.label_fc = MLP(nclass, hidden, nclass, n_layers_3, bns=label_bns, norm="batch", dropout=dropout)
        self.label_drop = nn.Dropout(label_drop)

        if "SeHGNN" in self.aggregation:
            self.semantic_aggr_layers = Transformer(hidden, att_drop, act)
        elif "HGAMLP" in self.aggregation:
            self.att_drop = nn.Dropout(att_drop)
            self.weight = nn.Parameter(torch.Tensor(hidden, 1))
            gain = nn.init.calculate_gain("sigmoid")
            nn.init.xavier_uniform_(self.weight, gain=gain)
        
        if "HOPE" in self.aggregation:
            self.mohe = HOPE(in_feats=hidden,
                             hidden_feats=hidden,
                             out_feats=nclass,
                             dropout=dropout,
                             num_layers=n_layers_1,
                             num_experts=(num_feats + num_label_feats),
                             similarity_threshold=similarity_threshold,
                             lower_bound=lower_bound,
                             upper_bound=upper_bound)
        else:
            self.concat_project_layer = nn.Linear((num_feats + num_label_feats) * hidden, hidden)
            self.lr_output = MLP(hidden, hidden, nclass, n_layers_2, bns=bns, norm="batch", dropout=dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        for layer in self.feat_project_layers:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()
        if self.label_feat_project_layers is not None:
            for layer in self.label_feat_project_layers:
                if isinstance(layer, Conv1d1x1):
                    layer.reset_parameters()
        if "SeHGNN" in self.aggregation:
            self.semantic_aggr_layers.reset_parameters()
        elif "HGAMLP" in self.aggregation:
            pass

        if "HOPE" in self.aggregation:
            self.mohe.reset_parameters()
        else:
            nn.init.xavier_uniform_(self.concat_project_layer.weight, gain=gain)
            nn.init.zeros_(self.concat_project_layer.bias)
            self.lr_output.reset_parameters()

        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        if isinstance(self.label_fc, nn.Linear):
            nn.init.xavier_uniform_(self.label_fc.weight, gain=gain)
            if self.label_fc.bias is not None:
                nn.init.zeros_(self.label_fc.bias)
        

    def forward(self, feats_dict, layer_feats_dict, label_emb, label=None, priori_expert_id=None):
        if self.embedings is not None:
            for k, v in feats_dict.items():
                if k in self.embedings:
                    feats_dict[k] = v @ self.embedings[k]

        tgt_feat = self.input_drop(feats_dict[self.tgt_key])
        B = num_node = tgt_feat.size(0)
        x = self.input_drop(torch.stack(list(feats_dict.values()), dim=1))
        x = self.feat_project_layers(x)
       
        if self.label_feat_project_layers is not None:
            label_feats = self.input_drop(torch.stack(list(layer_feats_dict.values()), dim=1))
            label_feats = self.label_feat_project_layers(label_feats)
            x = torch.cat((x, label_feats), dim=1)
        
        if "SeHGNN" in self.aggregation:
            x = self.semantic_aggr_layers(x)
        elif "HGAMLP" in self.aggregation:
            global_vector = torch.softmax(torch.sigmoid(torch.matmul(x, self.weight)).squeeze(2), dim=-1)
        
            output_r = []
            for i in range(x.shape[1]):
                output_r.append(x[:,i,:].mul(self.att_drop(global_vector[:, i].unsqueeze(1))))
            x = torch.stack(output_r, dim=1)
        if "HOPE" in self.aggregation:
            x = self.mohe(x)
        else:
            x = self.concat_project_layer(x.reshape(B, -1))
            x = self.dropout(self.prelu(x))
            x = self.lr_output(x)

        if self.label_residual:
            x = x + self.label_fc(self.label_drop(label_emb))

        return x

class HOPE(torch.nn.Module):
    def __init__(self, 
                 in_feats: int, 
                 hidden_feats: int, 
                 out_feats: int, 
                 dropout: float, 
                 num_layers: int, 
                 num_experts: int, 
                 similarity_threshold = 0.2, 
                 lower_bound: float = 0.5, 
                 upper_bound: float = 3):         # 新增：最低保障样本数 (原K值概念)
        super(HOPE, self).__init__()
        
        self.num_experts = num_experts
        self.similarity_threshold = similarity_threshold
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        self.in_feats = in_feats
        self.hidden = hidden_feats
        self.out_feats = out_feats
        
        # 专家原型向量
        self.expert_prototypes = torch.nn.Parameter(
            torch.empty(num_experts, in_feats)
        )
 
        self.expert_weights = nn.Parameter(torch.empty(num_experts, in_feats, hidden_feats))
        self.expert_bias = nn.Parameter(torch.empty(num_experts, hidden_feats))

        self.gating_logit_temp = nn.Parameter(torch.zeros(1)) 
        
        # 专家激活函数和Dropout
        self.expert_act = nn.PReLU()
        self.expert_dropout = nn.Dropout(dropout)
        self.expert_ln = VectorizedLayerNorm(num_experts, hidden_feats)

        # 共享专家
        shared_input_dim = in_feats * num_experts
        self.shared_expert = nn.Sequential(
            nn.Linear(shared_input_dim, hidden_feats),
            nn.PReLU(),
            nn.Dropout(dropout)
        )
        
        self.merge = MLP(hidden_feats, hidden_feats, out_feats, num_layers, dropout=dropout, norm="batch")
        # 参数初始化
        self.init_coverage_ratio()
        self.reset_parameters()
    
    def reset_parameters(self):
        """参数初始化"""
        nn.init.xavier_uniform_(self.expert_prototypes)
        for i in range(self.num_experts):
            nn.init.xavier_uniform_(self.expert_weights[i])
        nn.init.zeros_(self.expert_bias)

        for layer in self.shared_expert:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.merge.reset_parameters()
    
    def forward(self, feat_neighbor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_neighbor: [B, N, D]
        """
        # feat_neighbor: [B, N, D]
        B, N, D = feat_neighbor.shape
        device = feat_neighbor.device

        # ---- 1) 相似度 [B, N] ----
        feat_norm = F.normalize(feat_neighbor, p=2, dim=-1)          # [B, N, D]
        proto_norm = F.normalize(self.expert_prototypes, p=2, dim=-1) # [N, D]
        raw_sim = torch.einsum("bnd,nd->bn", feat_norm, proto_norm)   # [B, N]
        sim_expert_view = raw_sim.transpose(0, 1).contiguous()        # [N, B]

        # ---- 2) 设置 K 与 cap（容量上限）----
        K = int(self.lower_bound * B / self.num_experts)  # 每个专家至少 K 个
        K = max(1, min(K, B))

        cap = int((B / self.num_experts) * self.upper_bound)
        cap = max(K, min(cap, B))  # cap 至少 >= K，且不超过 B

        # ---- 3) 构造“阈值通过”的候选集，并在每个专家内做 capped topk ----
        threshold = float(self.similarity_threshold)
        NEG_INF = torch.finfo(sim_expert_view.dtype).min

        cand_scores = torch.where(
            sim_expert_view > threshold,
            sim_expert_view,
            torch.full_like(sim_expert_view, NEG_INF)
        )  # [N, B]

        cand_top_vals, cand_top_idx = torch.topk(cand_scores, k=cap, dim=1)  # [N, cap]
        cand_valid = cand_top_vals > NEG_INF / 2  # [N, cap] 近似判断是否为有效阈值样本

        # ---- 4) 保底：每个专家再取 topK（不看阈值）----
        topk_vals, topk_idx = torch.topk(sim_expert_view, k=K, dim=1)  # [N, K]

        dup = (cand_top_idx.unsqueeze(-1) == topk_idx.unsqueeze(1)).any(dim=-1)  # [N, cap]
        cand_keep = cand_valid & (~dup)  # [N, cap] 可用于补齐的位置

        fill = cap - K
        if fill > 0:
            fill_scores = torch.where(
                cand_keep,
                cand_top_vals,
                torch.full_like(cand_top_vals, NEG_INF)
            )  # [N, cap]

            fill_pos_vals, fill_pos = torch.topk(fill_scores, k=fill, dim=1)  # [N, fill]
            fill_idx = cand_top_idx.gather(1, fill_pos)                       # [N, fill]
            fill_valid = fill_pos_vals > NEG_INF / 2                          # [N, fill]
        else:
            fill_idx = torch.empty(N, 0, dtype=cand_top_idx.dtype, device=device)
            fill_valid = torch.empty(N, 0, dtype=torch.bool, device=device)

        # 拼接最终选择： [N, cap]
        selected_idx = torch.cat([topk_idx, fill_idx], dim=1)          # [N, cap]
        selected_valid = torch.cat(
            [torch.ones(N, K, dtype=torch.bool, device=device), fill_valid],
            dim=1
        )

        # ---- 6) 计算 gating 权重（只对选中位置）----
        temperature = F.softplus(self.gating_logit_temp).clamp(min=0.01, max=10.0)

        # 取出选中位置的相似度： [N, cap]
        selected_sim = sim_expert_view.gather(1, selected_idx)

        gating_weights = torch.sigmoid(selected_sim / temperature)  # [N, cap]
        # 将无效补齐位权重置 0（当阈值候选不足 cap-K 时）
        gating_weights = gating_weights * selected_valid.float()

        # ---- 7) 向量化专家并行计算（保持你原始风格）----
        # 展平索引 [N*cap]
        flat_indices = selected_idx.reshape(-1)  # [N*cap]

        # 专家编号展开 [N*cap]
        expert_idx_range = torch.arange(N, device=device).repeat_interleave(cap)  # [N*cap]

        # 取输入特征：对于专家 n，取 feat_neighbor[batch_idx, n, :]
        expert_inputs_flat = feat_neighbor[flat_indices, expert_idx_range, :]  # [N*cap, D]
        expert_inputs_grouped = expert_inputs_flat.view(N, cap, D)             # [N, cap, D]

        # 并行线性： [N, cap, D] x [N, D, H] -> [N, cap, H]
        expert_outputs_grouped = torch.einsum("ncd,ndh->nch", expert_inputs_grouped, self.expert_weights)
        expert_outputs_grouped = expert_outputs_grouped + self.expert_bias.unsqueeze(1)

        # LN/act/dropout（如你的 VectorizedLayerNorm 支持 [N, cap, H] 就继续用它）
        expert_outputs_grouped = self.expert_ln(expert_outputs_grouped)
        expert_outputs_grouped = self.expert_act(expert_outputs_grouped)
        expert_outputs_grouped = self.expert_dropout(expert_outputs_grouped)

        # 加权
        weighted_outputs = expert_outputs_grouped * gating_weights.unsqueeze(-1)  # [N, cap, H]

        # 聚合回 [B, H]
        flat_weighted_outputs = weighted_outputs.reshape(-1, self.hidden)  # [N*cap, H]
        moe_feat = torch.zeros(B, self.hidden, device=device, dtype=feat_neighbor.dtype)
        moe_feat = moe_feat.index_add(0, flat_indices, flat_weighted_outputs.to(moe_feat.dtype))

        # ---- 8) 覆盖率（按“被任何专家选中过至少一次”统计）----
        unique_nodes = torch.unique(flat_indices)
        self.set_coverage_ratio(num_covered=unique_nodes.numel(), num_nodes=B)

        # 5. 共享专家处理 (保持不变)
        shared_input = feat_neighbor.reshape(B, -1)
        shared_feat = self.shared_expert(shared_input)

        # 6. 特征融合
        combined_feat = shared_feat + moe_feat
        predictions = self.merge(combined_feat)

        # Loss 计算
        self.expert_prototypes_loss = self.prototype_separation_loss()
        
        return predictions
    
    def prototype_separation_loss(self):
        num_experts = self.expert_prototypes.shape[0]
        prototypes_norm = F.normalize(self.expert_prototypes, p=2, dim=1)
        sim_matrix = torch.matmul(prototypes_norm, prototypes_norm.T)
        identity = torch.eye(num_experts, device=self.expert_prototypes.device)
        loss = torch.norm(sim_matrix - identity, p='fro') ** 2
        normalization = num_experts * (num_experts - 1)
        return loss / (normalization + 1e-9)
    
    def set_coverage_ratio(self, num_covered, num_nodes):
        self.num_covered += num_covered
        self.num_nodes += num_nodes

    def init_coverage_ratio(self):
        self.num_covered = 0
        self.num_nodes = 0

class VectorizedLayerNorm(nn.Module):
    """
    支持每个专家拥有自己独立的参数 γ 和 β，
    同时可以一次性对多个专家的输入向量化计算。
    """
    def __init__(self, num_experts, hidden_feats, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_experts, hidden_feats))
        self.beta = nn.Parameter(torch.zeros(num_experts, hidden_feats))

    def forward(self, x):
        """
        x: [num_experts, capacity, hidden_feats]
        返回: 同形状的张量
        """
        # 1. 按最后一维计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        # 2. 标准化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # 3. 应用每个专家的参数（通过广播）
        return x_norm * self.gamma.unsqueeze(1) + self.beta.unsqueeze(1)