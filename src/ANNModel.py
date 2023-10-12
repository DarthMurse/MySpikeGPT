import torch 
from torch import nn
import torch.nn.functional as F 
import math

from .args import *

# This model replaces all the spiking neurons with ReLU

class MySpikeGPT(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args 

        self.encode_layer = EncodingLayer(self.args)
        self.transformer = [TransformerBlock(self.args) for i in range(self.args.n_layers)]
        self.out = OutputLayer(self.args)

    def forward(self, x):
        # x: [B, S], out: [B, S, vocab]
        assert x.shape[1] == self.args.ctx_len, "input sequence length is not equal to ctx_len!"
        out = self.encode_layer(x)
        for i in range(self.args.n_layers):
            out = self.transformer[i](out)
        out = self.out(out)
        return out

class EncodingLayer(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.emb = nn.Embedding(self.args.vocab_size, self.args.embed)
        self.ln = nn.LayerNorm(self.args.embed).to(self.args.device)

        # Apply sin/cos positional embedding (just because it is easy)
        self.poe = torch.zeros(self.args.ctx_len, self.args.embed, requires_grad=False)
        lamb = 10000
        for i in range(self.args.ctx_len):
            for j in range(self.args.embed):
                if j % 2 == 0:
                    self.poe[i, j] = math.sin(i / lamb ** (j/self.args.embed))
                else:
                    self.poe[i, j] = math.cos(i / lamb ** ((j-1)/self.args.embed))
        self.poe = self.poe.to(self.args.device)

    def forward(self, x):
        # x: [B, S], out: [B, S, D]
        S = x.shape[1]
        out = self.emb(x)
        out = out + self.poe[:S]
        out = self.ln(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.ffn = FFN(self.args)
        self.sdsa = SDSA(self.args)

    def forward(self, x):
        # x: [B, S, D], out: [B, S, D]
        h = x + self.sdsa(x)
        out = h + self.ffn(h)
        return out 

class SDSA(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.n_heads = self.args.n_heads
        self.head_dim = self.args.head_dim
        self.dim = self.args.embed

        self.wk = nn.Parameter(torch.zeros([self.dim, self.dim])).to(self.args.device)
        self.wv = nn.Parameter(torch.zeros([self.dim, self.dim])).to(self.args.device)
        self.wq = nn.Parameter(torch.zeros([self.dim, self.dim])).to(self.args.device)
        self.wo = nn.Parameter(torch.zeros([self.dim, self.dim])).to(self.args.device)

        self.spike_q = nn.ReLU()
        self.spike_k = nn.ReLU()
        self.spike_v = nn.ReLU()
        self.init_spike = nn.ReLU()
        self.talking_heads = nn.ReLU()

        self.lnq = nn.LayerNorm(self.args.embed).to(self.args.device)
        self.lnk = nn.LayerNorm(self.args.embed).to(self.args.device)
        self.lnv = nn.LayerNorm(self.args.embed).to(self.args.device)
        self.lno = nn.LayerNorm(self.args.embed).to(self.args.device)

    def forward(self, x):
        T, B, S, D = x.shape

        tmp = self.init_spike(x)
        # Needs optimization
        Q = tmp @ self.wq 
        V = tmp @ self.wv 
        K = tmp @ self.wk
        Q = self.spike_q(self.lnq(Q)) # [B, S, D]
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

class FFN(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.hidden = model_args.ffn_hidden_layer
        self.dim = model_args.embed 
        self.spike1 = nn.ReLU()
        self.init_spike = nn.ReLU()
        self.linear1 = nn.Linear(self.dim, self.hidden, bias=False).to(self.args.device)
        self.linear2 = nn.Linear(self.hidden, self.dim, bias=False).to(self.args.device)
        self.ln1 = nn.LayerNorm(self.hidden).to(self.args.device)
        self.ln2 = nn.LayerNorm(self.args.embed).to(self.args.device)
        
    def forward(self, x):
        out = self.init_spike(x)
        out = self.linear1(out)
        out = self.ln1(out)
        out = self.spike1(out)
        out = self.linear2(out)
        out = self.ln2(out)
        return out 

class OutputLayer(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args 
        self.output = nn.Linear(self.args.embed, self.args.vocab_size, bias=False).to(self.args.device)
        self.init_spike = nn.ReLU()

    def forward(self, x):
        out = self.init_spike(x) # [T, B, S, D]
        out = out.mean(0)
        out = self.output(out)
        return out