import torch 
from torch import nn
import torch.nn.functional as F 
import math
from typing import Optional, Tuple

from .args import *

class PreNormResidual(nn.Module):
    def __init__(self, dim, layer):
        super().__init__()
        self.dim = dim
        self.layer = layer
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.layer(self.norm(x)) + x

class FFN(nn.Module):
    def __init__(self, dim, factor=4):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(dim, dim*factor),
                nn.ReLU(),
                nn.Linear(dim*factor, dim),
                nn.ReLU()
                )

    def forward(self, x):
        return self.layers(x)

class Mixer(nn.Module):
    def __init__(self, model_args=args, head=12):
        super().__init__()
        self.args = model_args
        self.head = 12
        assert self.args.embed % self.head == 0, "args.embed must be a multiple of head!"
        self.weight = nn.Parameter(torch.zeros([self.head, self.args.ctx_len, self.args.ctx_len]))
        self.register_parameter("weight", self.weight)

    def forward(self, x):
        B, D, S = x.shape
        with torch.no_grad():
            for i in range(self.head):
                self.weight[i] = self.weight[i].triu()
        out = x.reshape(B, self.head, D//self.head, S)
        out = out @ self.weight
        out = out.reshape(B, D, S)
        return out

class MLPBlock(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.block1 = PreNormResidual(self.args.embed, FFN(self.args.embed))
        self.block2 = PreNormResidual(self.args.ctx_len, Mixer(self.args))
        self.register_module('block1', self.block1)
        self.register_module('block2', self.block2)

    def forward(self, x):
        B, S, D = x.shape
        out = self.block1(x)
        out = out.reshape(B, D, S)
        out = self.block2(out)
        out = out.reshape(B, S, D)
        return out

class OutputLayer(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.output = nn.Linear(self.args.embed, self.args.vocab_size, bias=False)
        self.norm = nn.LayerNorm(self.args.embed)
    
    def forward(self, x):
        B, S, D = x.shape
        out = self.norm(x)
        return self.output(out)

class MySpikeGPT(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.pos = nn.Parameter(torch.zeros([self.args.ctx_len, self.args.embed]))
        self.register_parameter('pos', self.pos)
        self.encode = nn.Embedding(self.args.vocab_size, self.args.embed)
        self.mlp = nn.Sequential(
                *[MLPBlock(self.args) for _ in range(self.args.n_layers)],
                OutputLayer(self.args)
                )

    def forward(self, x, y=None):
        out = self.encode(x)
        out = out + self.pos
        out = self.mlp(out)
        if y is not None:
            return F.cross_entropy(out.view(-1, self.args.vocab_size), y.view(-1))
        else:
            return out[:, -1, :]
