# models.py

import torch
import math
import torch.nn.functional as F
from torch.nn import Parameter, Module, Linear, BatchNorm1d
from einops import repeat, einsum
from mamba_ssm import Mamba as MambaOfficial

# Wrapper برای سازگاری با نسخه اصلی Mamba
class Mamba(Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        effective_d_state = max(d_state, 8)
        self.mamba = MambaOfficial(
            d_model=d_model,
            d_state=effective_d_state,
            d_conv=d_conv,
            expand=expand,
        )
    
    def forward(self, x):
        return self.mamba(x)

class GCN_mamba_liner(Module):
    def __init__(self, in_features, out_features, with_bias=False):
        super(GCN_mamba_liner, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = input @ self.weight
        return output + self.bias if self.bias is not None else output

class GCN_mamba_block(Module):
    def __init__(self, args, dataset):
        super(GCN_mamba_block, self).__init__()
        self.args = args
        self.x_proj = GCN_mamba_liner(args.d_model, args.dt_rank + args.d_state * 2, with_bias=args.bias)
        self.dropout = args.mamba_dropout
        self.dt_proj = GCN_mamba_liner(args.dt_rank, args.d_model, with_bias=args.bias)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_model)
        self.A_log = Parameter(torch.log(A))
        self.D = Parameter(torch.ones(args.d_model))
        self.out_proj = GCN_mamba_liner(args.d_model, args.d_model, with_bias=args.bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.x_proj.reset_parameters()
        self.dt_proj.reset_parameters()
        self.out_proj.reset_parameters()

    def forward(self, x, adj, layer_num):
        alpha = 0.05
        features = [x]
        xi = x
        for i in range(layer_num - 1):
            xi = adj @ xi
            xi = (1-alpha)*xi + alpha*x
            features.append(xi)

        x_stacked = torch.stack(features, dim=0).transpose(0, 1)
        y = self.ssm(x_stacked, self.args, adj)
        return self.out_proj(y)

    def ssm(self, x, args, adj):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        
        x_dbl = self.x_proj(x)
        x_dbl = F.relu(x_dbl)
        x_dbl = F.dropout(x_dbl, p=self.dropout, training=self.training)

        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, A, B, C, D, args, adj)

    def selective_scan(self, u, delta, A, B, C, D, args, adj):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        x_state = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x_state = deltaA[:, i] * x_state + deltaB_u[:, i]
            x_state = F.dropout(x_state, p=self.dropout, training=self.training)
            y = einsum(x_state, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)

        y_out = torch.stack(ys, dim=1)
        return y_out + u * D

class GCN_mamba_Net(Module):
    def __init__(self, dataset, args):
        super(GCN_mamba_Net, self).__init__()
        self.args = args
        self.dropout = args.mamba_dropout
        self.a = args.alpha
        self.b = args.graph_weight
        self.lin1 = GCN_mamba_liner(dataset.num_features, args.d_model, with_bias=args.bias)
        self.mamba_global_attention = Mamba(d_model=args.d_model, d_state=8, expand=1)
        self.bn_1 = BatchNorm1d(args.d_model)
        self.mamba = GCN_mamba_block(args, dataset)
        self.bn_2 = BatchNorm1d(args.d_model)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.mamba.reset_parameters()

    def forward(self, x, adj):
        # 1. Embedding اولیه
        x_input = self.lin1(x)

        # 2. بخش Mamba جهانی (Global)
        ga_in = x_input.reshape(1, x_input.shape[0], x_input.shape[1])
        ga_in = F.dropout(ga_in, p=self.dropout, training=self.training)
        ga_out = self.mamba_global_attention(ga_in)
        ga_out_flip = self.mamba_global_attention(torch.flip(ga_in, dims=[1]))
        ga_final = ga_out + torch.flip(ga_out_flip, dims=[1])
        ga_final = F.relu(F.dropout(ga_final, p=self.dropout, training=self.training))
        ga_final = torch.squeeze(ga_final) * (1 - self.a) + self.a * x_input

        # 3. بخش Mamba محلی (Local)
        x_local_in = F.dropout(F.relu(self.bn_1(x_input)), p=self.dropout, training=self.training)
        all_layers_output = self.mamba(x_local_in, adj, self.args.layer_num)
        local_output = all_layers_output[:, -1, :] + x_input
        
        # 4. ترکیب نهایی
        output = local_output * self.b + ga_final * (1 - self.b)
        output = F.dropout(F.relu(self.bn_2(output)), p=self.dropout, training=self.training)
        
        # <--- CHANGE: فقط خروجی نهایی و ورودی اولیه برگردانده می‌شود
        return output, x_input
