import torch 
from torch import nn
import torch.nn.functional as F 
import math
from typing import Optional, Tuple

from .args import *

class IF(nn.Module):
    def __init__(self, T=4, step=1, is_first=False):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(8.0))
        self.T = T
        self.step = step
        self.is_first = is_first

    def forward(self, x):
        # x [T, B, S, D]
        if self.is_first:
            x.unsqueeze_(0)
            x = x.repeat(self.T, 1, 1, 1)

        threshold = self.alpha
        membrane = 0.5 * threshold
        spikes = torch.zeros(x.shape)

        for i in range(self.step):
            membrane = membrane + x[i]
        for i in range(0, self.T):
            if i + self.step < self.T:
                membrane = membrane + x[i + self.step]
            spike = membrane > threshold
            membrane[spike] = membrane[spike] - threshold
            spikes[i+j] = spike.float()

        return threshold * spikes

class SpikeInnerProduct(nn.Module):
    def __init__(self, T=4, step=1):
        super().__init__()
        self.T = 4
        self.step = 1

    def forward(self, x, y):
        T, B, n_heads, S, head_dim = x.shape
        out_weight = torch.zeros([T, B, n_heads, S, S])
        
        for i in range(0, self.T, self.step):
            x_add = 0
            y_add = 0
            for j in range(self.step):
                x_add = x_add + x[i+j]
                y_add = y_add + y[i+j]
            weight = x_add @ y_add
            for j in range(self.step):
                out_weight[i+j] = weight
        return out_weight

class MySpikeGPT(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args 

        self.encode_layer = nn.Embedding(self.args.vocab_size, self.args.embed)
        self.transformer = [TransformerBlock(i, self.args) for i in range(self.args.n_layers)]
        self.out = OutputLayer(self.args)

        self.pos = nn.Parameter(torch.zeros([self.args.ctx_len, self.args.embed]))
        self.register_parameter('pos', self.pos)

        self.init_spike = IF(step=4, is_first=True)

        for i in range(self.args.n_layers):
            self.register_module('transformer_block'+str(i), self.transformer[i])
        self.register_module('out', self.out)

    def forward(self, x, y=None):
        # x: [B, S], out: [B, S, vocab]
        assert x.shape[1] == self.args.ctx_len, "input sequence length is not equal to ctx_len!"
        out = self.encode_layer(x)
        out = out + self.pos
        out = self.init_spike(out)
        for i in range(self.args.n_layers):
            out = self.transformer[i](out)
        out = self.out(out)

        if y is not None and y.shape[1] != 2:
            return F.cross_entropy(out.view(-1, self.args.vocab_size), y.view(-1))
        elif y is not None and y.shape[1] == 2:
            return out
        else:
            return out[:, -1, :]

class TransformerBlock(nn.Module):
    def __init__(self, i, model_args=args):
        super().__init__()
        self.args = model_args
        self.ffn = FFN(self.args)
        self.sdsa = SDSA(i, self.args)
        #self.sdsa_norm = nn.LayerNorm(self.args.embed)
        #self.ffn_norm = nn.LayerNorm(self.args.embed)

        self.register_module('ffn', self.ffn)
        self.register_module('sdsa', self.sdsa)
        #self.register_module('sdsa_norm', self.sdsa_norm)
        #self.register_module('ffn_norm', self.ffn_norm)

    def forward(self, x):
        # x: [B, S, D], out: [B, S, D]
        h = x + self.sdsa(x)
        out = h + self.ffn(h)
        return out 

class SDSA(nn.Module):
    def __init__(self, i, model_args=args):
        super().__init__()
        self.args = model_args
        self.n_heads = self.args.n_heads
        self.head_dim = self.args.head_dim
        self.dim = self.args.embed

        self.wk = nn.Linear(self.dim, self.dim, bias=False)
        self.wv = nn.Linear(self.dim, self.dim, bias=False)
        self.wq = nn.Linear(self.dim, self.dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

        self.q_if = IF(step=4)
        self.k_if = IF(step=4)
        self.v_if = IF(step=4)
        self.o_if = IF(step=4)
        self.softmax_if = IF(step=4)
        self.weight_if = IF(step=4)
        self.inner_prodcut = SpikeInnerProduct(step=4)

        #self.freq_cis = precompute_freqs_cis(self.head_dim, self.args.ctx_len)

    def forward(self, x):
        T, B, S, D = x.shape

        Q = self.wq(x)
        V = self.wv(x)
        K = self.wk(x)

        Q = Q.reshape(T, B, -1, self.n_heads, self.head_dim)
        V = V.reshape(T, B, -1, self.n_heads, self.head_dim)
        K = K.reshape(T, B, -1, self.n_heads, self.head_dim)

        Q = Q.transpose(2, 3)
        V = V.transpose(2, 3)
        K = K.transpose(2, 3)  # [T, B, n_heads, S, head_dim]
        K = self.k_if(K)
        Q = self.q_if(Q)
        V = self.v_if(V)

        QK = self.inner_product(Q, K) / math.sqrt(self.dim) # [T, B, n_heads, S, S]
        mask = torch.full(
                (1, 1, S, S), float("-inf"), device=self.args.device
            )
        mask = torch.triu(mask, diagonal=1).type_as(x)
        QK = QK + mask
        QK = torch.softmax(QK, dim=-1) # [B, n_heads, S, S]
        QK = self.softmax_if(QK)
        QKV = QK @ V # [B, n_heads, S, head_dim]

        QKV = QKV.transpose(2, 1)
        QKV = QKV.reshape(B, S, -1)
        QKV = self.weight_if(QKV)
        QKV = self.wo(QKV)
        QKV = self.o_if(QKV)
        return QKV

class FFN(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.hidden = model_args.ffn_hidden_layer
        self.dim = model_args.embed 
        #self.spike1 = nn.SiLU()
        self.spike1 = QuantReLU()
        #self.init_spike = nn.SiLU()
        self.init_spike = QuantReLU()
        self.linear1 = nn.Linear(self.dim, self.hidden, bias=False)
        self.linear2 = nn.Linear(self.hidden, self.dim, bias=False)
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.init_spike(out)
        out = self.linear2(out)
        out = self.spike1(out)
        return out 

class OutputLayer(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args 
        self.output = nn.Linear(self.args.embed, self.args.vocab_size, bias=False)
        #self.norm = nn.LayerNorm(self.args.embed)
        #self.out_if = QuantReLU()
        #self.register_module("norm", self.norm)

    def forward(self, x):
        #out = self.norm(x)
        #out = self.out_if(out)
        out = self.output(x)
        return out
