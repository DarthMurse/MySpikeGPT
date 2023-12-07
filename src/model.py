import torch 
from torch import nn
import torch.nn.functional as F 
import math
from spikingjelly.activation_based import neuron, functional, surrogate, layer

from .args import *

class MySpikeGPT(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args 

        self.encode_layer = EncodingLayer(self.args)
        self.spiking_transformer = [SpikeTransformerBlock(i, self.args) for i in range(self.args.n_layers)]
        self.out = OutputLayer(self.args)

        self.register_module('encode_layer', self.encode_layer)
        for i in range(self.args.n_layers):
            self.register_module('block'+str(i), self.spiking_transformer[i])
        self.register_module('output_layer', self.out)

    def forward(self, x):
        # x: [B, S], out: [B, S, vocab]
        assert x.shape[1] == self.args.ctx_len, "input sequence length is not equal to ctx_len!"
        out = self.encode_layer(x)
        for i in range(self.args.n_layers):
            out = self.spiking_transformer[i](out)
        out = self.out(out)
        return out

    def reset(self):
        self.encode_layer.reset()
        for i in range(self.args.n_layers):
            self.spiking_transformer[i].reset()
        self.out.reset()

class EncodingLayer(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.emb = nn.Embedding(self.args.vocab_size, self.args.embed)
        self.ln = nn.LayerNorm(self.args.embed)
        
        self.poe = nn.Parameter(torch.zeros(self.args.ctx_len, self.args.embed))

    def forward(self, x):
        # x: [B, S], out: [T, B, S, D]
        S = x.shape[1]
        out = self.emb(x)
        out = out + self.poe
        out = out.unsqueeze(0).repeat(self.args.T, 1, 1, 1)
        out = self.ln(out)
        return out

    def reset(self):
        pass

class SpikeTransformerBlock(nn.Module):
    def __init__(self, i, model_args=args):
        super().__init__()
        self.args = model_args
        self.ffn = FFN(self.args)
        self.sdsa = SDSA(i, self.args)

        self.register_module('ffn', self.ffn)
        self.register_module('sdsa', self.sdsa)

    def forward(self, x):
        # x: [T, B, S, D], out: [T, B, S, D]
        h = x + self.sdsa(x)
        out = h + self.ffn(h)
        return out 

    def reset(self):
        self.ffn.reset()
        self.sdsa.reset()

class SDSA(nn.Module):
    def __init__(self, i, model_args=args):
        super().__init__()
        self.args = model_args
        self.n_heads = self.args.n_heads
        self.head_dim = self.args.head_dim
        self.dim = self.args.embed

        self.wk = nn.Parameter(torch.zeros([self.dim, self.dim]))
        self.wv = nn.Parameter(torch.zeros([self.dim, self.dim]))
        self.wq = nn.Parameter(torch.zeros([self.dim, self.dim]))
        self.wo = nn.Parameter(torch.zeros([self.dim, self.dim]))

        self.spike_q = neuron.IFNode(step_mode='m', surrogate_function=surrogate.ATan(2.0))
        self.spike_k = neuron.IFNode(step_mode='m', surrogate_function=surrogate.ATan(2.0))
        self.spike_v = neuron.IFNode(step_mode='m', surrogate_function=surrogate.ATan(2.0))
        self.init_spike = neuron.IFNode(step_mode='m', surrogate_function=surrogate.ATan(2.0))
        self.talking_heads = neuron.IFNode(step_mode='m', surrogate_function=surrogate.ATan(2.0))

        self.lnq = nn.LayerNorm(self.args.embed)
        self.lnk = nn.LayerNorm(self.args.embed)
        self.lnv = nn.LayerNorm(self.args.embed)
        self.lno = nn.LayerNorm(self.args.embed)

    def forward(self, x):
        T, B, S, D = x.shape

        tmp = self.init_spike(x)
        # Needs optimization
        Q = tmp @ self.wq 
        V = tmp @ self.wv 
        K = tmp @ self.wk
        Q = self.spike_q(self.lnq(Q)) # [T, B, S, D]
        V = self.spike_v(self.lnv(V))
        K = self.spike_k(self.lnk(K))

        Q = Q.reshape(T, B, -1, self.n_heads, self.head_dim)
        V = V.reshape(T, B, -1, self.n_heads, self.head_dim)
        K = K.reshape(T, B, -1, self.n_heads, self.head_dim)
        Q = Q.transpose(2, 3)
        V = V.transpose(2, 3)
        K = K.transpose(2, 3)  # [T, B, n_heads, S, head_dim]

        QK = Q.mul(K).sum(dim=-2, keepdim=True) # [T, B, n_heads, 1, head_dim]
        QK = self.talking_heads(QK)
        QKV = V.mul(QK) # [T, B, n_heads, S, head_dim]

        QKV = QKV.transpose(2, 3)
        QKV = QKV.reshape(T, B, -1, self.dim)
        QKV = self.lno(QKV @ self.wo)
        return QKV

    def reset(self):
        self.spike_q.reset()
        self.spike_v.reset()
        self.spike_k.reset()
        self.init_spike.reset()
        self.talking_heads.reset()

class FFN(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.hidden = model_args.ffn_hidden_layer
        self.dim = model_args.embed 
        self.spike1 = neuron.IFNode(step_mode='m', surrogate_function=surrogate.ATan(2.0))
        self.init_spike = neuron.IFNode(step_mode='m', surrogate_function=surrogate.ATan(2.0))
        self.linear1 = nn.Linear(self.dim, self.hidden, bias=False)
        self.linear2 = nn.Linear(self.hidden, self.dim, bias=False)
        self.ln1 = nn.LayerNorm(self.hidden)
        self.ln2 = nn.LayerNorm(self.args.embed)
        
    def forward(self, x):
        out = self.init_spike(x)
        out = self.linear1(out)
        out = self.ln1(out)
        out = self.spike1(out)
        out = self.linear2(out)
        out = self.ln2(out)
        return out 

    def reset(self):
        self.spike1.reset()
        self.init_spike.reset()

class OutputLayer(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args 
        self.output = nn.Linear(self.args.embed, self.args.vocab_size, bias=False)
        self.init_spike = neuron.IFNode(step_mode='m', surrogate_function=surrogate.ATan(2.0))
        
    def forward(self, x):
        out = self.init_spike(x) # [T, B, S, D]
        out = out.mean(0)
        out = self.output(out)
        return out

    def reset(self):
        self.init_spike.reset()
