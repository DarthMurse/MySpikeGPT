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

class MLPBlock(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.block1 = PreNormResidual(self.args.embed, FFN(self.args.embed))
        self.block2 = PreNormResidual(self.args.ctx_len, FFN(self.args.ctx_len))
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

    def forward(self, x):
        out = self.encode(x)
        out = out + self.pos
        return self.mlp(out)
