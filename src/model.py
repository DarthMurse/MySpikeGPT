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
        self.spiking_transformer = [SpikeTransformerBlock(self.args) for i in range(self.args.n_layers)]
        self.out = OutputLayer(self.args)

    # cur_pos start from 1
    def forward(self, x, cur_pos):
        out = self.encode_layer(x, cur_pos)
        for i in range(self.args.n_layers):
            out = self.spiking_transformer[i](out, cur_pos)
        out = self.out(out, cur_pos)
        return out

    def reset(self):
        self.encode_layer.reset()
        for layer in self.spiking_transformer:
            layer.reset()
        self.out.reset()

class EncodingLayer(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.emb = nn.Embedding(self.args.vocab_size, self.args.embed)
        self.poe = torch.zeros(self.args.ctx_len, self.args.embed)
        lamb = 10000
        for i in range(self.args.ctx_len):
            for j in range(self.args.embed):
                if j % 2 == 0:
                    self.poe[i, j] = math.sin(i / lamb ** (j/self.args.embed))
                else:
                    self.poe[i, j] = math.cos(i / lamb ** ((j-1)/self.args.embed))

    def forward(self, x, cur_pos):
        B, _ = x.shape
        out = self.emb(x)
        out = out + self.poe
        out = out.unsqueeze(0).repeat(self.args.T, 1, 1, 1)
        out[:, :, :cur_pos, :] = nn.BatchNorm2d(B)(out[:, :, :cur_pos, :])
        return out

    def reset(self):
        return

class SpikeTransformerBlock(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.ffn = FFN(self.args)
        self.sdsa = SDSA(self.args)

    def forward(self, x, cur_pos):
        h = x + self.sdsa(x, cur_pos)
        out = h + self.ffn(h, cur_pos)
        return out 

    def reset(self):
        self.ffn.reset()
        self.sdsa.reset()

class SDSA(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.n_heads = self.args.n_heads
        self.head_dim = self.args.head_dim
        self.dim = self.args.embed

        self.wk = nn.Parameter(torch.zeros([self.dim, self.dim]))
        self.wv = nn.Parameter(torch.zeros([self.dim, self.dim]))
        self.wq = nn.Parameter(torch.zeros([self.dim, self.dim]))
        self.wo = nn.Parameter(torch.zeros([self.dim, self.dim]))

        self.spike_q = neuron.IFNode(step_mode='m')
        self.spike_k = neuron.IFNode(step_mode='m')
        self.spike_v = neuron.IFNode(step_mode='m')
        self.init_spike = neuron.IFNode(step_mode='m')
        self.talking_heads = neuron.IFNode(step_mode='m')

    def forward(self, x, cur_pos):
        T, B, S, D = x.shape
        self.bnk = nn.BatchNorm2d(B)
        self.bnq = nn.BatchNorm2d(B)
        self.bnv = nn.BatchNorm2d(B)
        self.bno = nn.BatchNorm2d(B)

        x[:, :, :cur_pos, :] = self.init_spike(x[:, :, :cur_pos, :])
        Q = x[:, :, :cur_pos, :] @ self.wq 
        V = x[:, :, :cur_pos, :] @ self.wv 
        K = x[:, :, :cur_pos, :] @ self.wk
        Q = self.spike_q(self.bnq(Q)) # [T, B, cur_pos+1, dim]
        V = self.spike_q(self.bnq(V))
        K = self.spike_q(self.bnq(K))

        Q = Q.reshape(T, B, -1, self.n_heads, self.head_dim)
        V = V.reshape(T, B, -1, self.n_heads, self.head_dim)
        K = K.reshape(T, B, -1, self.n_heads, self.head_dim)
        Q = Q.transpose(2, 3)
        V = V.transpose(2, 3)
        K = K.transpose(2, 3)  # [T, B, n_heads, cur_pos, head_dim]

        QK = Q.mul(K).sum(dim=-2, keepdim=True) # [T, B, n_heads, 1, head_dim]
        QK = self.talking_heads(QK)
        QKV = V.mul(QK) # [T, B, n_heads, cur_pos, head_dim]

        QKV = QKV.transpose(2, 3)
        QKV = QKV.reshape(T, B, -1, self.dim)
        QKV = self.bno(QKV @ self.wo)
        QKV = torch.cat((QKV, x[:, :, cur_pos:, :]), dim=2)
        return QKV

    def reset(self):
        self.spike_k.reset()
        self.spike_q.reset()
        self.spike_v.reset()
        self.talking_heads.reset()
        self.init_spike.reset()

class FFN(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.hidden = model_args.ffn_hidden_layer
        self.dim = model_args.embed 
        self.spike1 = neuron.IFNode(step_mode='m')
        self.init_spike = neuron.IFNode(step_mode='m')
        self.linear1 = nn.Linear(self.dim, self.hidden)
        self.linear2 = nn.Linear(self.hidden, self.dim)
        
    def forward(self, x, cur_pos):
        T, B, _, _ = x.shape
        self.bn1 = nn.BatchNorm2d(B)
        self.bn2 = nn.BatchNorm2d(B)

        out = self.init_spike(x[:, :, :cur_pos, :])
        out = self.linear1(out)
        out = self.bn1(out)
        out = self.spike1(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = torch.concat((out, x[:, :, cur_pos:, :]), dim=2)
        return out 

    def reset(self):
        self.spike1.reset()
        self.init_spike.reset()

class OutputLayer(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args 
        self.output = nn.Linear(self.args.embed, self.args.vocab_size)
        self.spike = neuron.IFNode(step_mode='m')
        self.init_spike = neuron.IFNode(step_mode='m')
        
    def forward(self, x, cur_pos):
        T, B, _, _ = x.shape
        self.bn = nn.BatchNorm2d(B)

        out = self.init_spike(x[:, :, :cur_pos, :]) # [T, B, S, D]
        out = self.output(out)
        out = self.bn(out)
        out = self.spike(out)
        out = out.mean(0)
        out = out / out.sum(dim=-1, keepdim=True)
        return out[:, cur_pos-1, :]

    def reset(self):
        self.spike.reset()
        self.init_spike.reset()
