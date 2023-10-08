import torch 
from torch import nn
import torch.nn.functional as F 
import math
from spikingjelly.activation_based import neuron, functional, surrogate, layer

from .args import *

# Accept a single sequence as input, without batch
class MySpikeGPT(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args 

        self.encode_layer = EncodingLayer(self.args)
        self.spiking_transformer = [SpikeTransformerBlock(self.args) for i in range(self.args.n_layers)]
        self.out = OutputLayer(self.args)

    # cur_pos start from 1
    def forward(self, x, cur_pos):
        out = x[:cur_pos]
        out = self.encode_layer(out, cur_pos)
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
        self.poe = torch.zeros(self.args.ctx_len, self.args.embed, requires_grad=False)
        lamb = 10000
        for i in range(self.args.ctx_len):
            for j in range(self.args.embed):
                if j % 2 == 0:
                    self.poe[i, j] = math.sin(i / lamb ** (j/self.args.embed))
                else:
                    self.poe[i, j] = math.cos(i / lamb ** ((j-1)/self.args.embed))
        self.poe = self.poe.to(self.args.device)

    def forward(self, x, cur_pos):
        bn = nn.BatchNorm1d(cur_pos).to(self.args.device)
        S = x.shape[0]
        out = self.emb(x)
        out = out + self.poe[:S]
        out = out.unsqueeze(0).repeat(self.args.T, 1, 1)
        out = bn(out)
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

        self.wk = nn.Parameter(torch.zeros([self.dim, self.dim])).to(self.args.device)
        self.wv = nn.Parameter(torch.zeros([self.dim, self.dim])).to(self.args.device)
        self.wq = nn.Parameter(torch.zeros([self.dim, self.dim])).to(self.args.device)
        self.wo = nn.Parameter(torch.zeros([self.dim, self.dim])).to(self.args.device)

        self.spike_q = neuron.IFNode(step_mode='m')
        self.spike_k = neuron.IFNode(step_mode='m')
        self.spike_v = neuron.IFNode(step_mode='m')
        self.init_spike = neuron.IFNode(step_mode='m')
        self.talking_heads = neuron.IFNode(step_mode='m')

    def forward(self, x, cur_pos):
        T, S, D = x.shape
        bnk = nn.BatchNorm1d(cur_pos).to(self.args.device)
        bnq = nn.BatchNorm1d(cur_pos).to(self.args.device)
        bnv = nn.BatchNorm1d(cur_pos).to(self.args.device)
        bno = nn.BatchNorm1d(cur_pos).to(self.args.device)

        tmp = self.init_spike(x)
        Q = tmp @ self.wq 
        V = tmp @ self.wv 
        K = tmp @ self.wk
        Q = self.spike_q(bnq(Q)) # [T, cur_pos+1, dim]
        V = self.spike_v(bnv(V))
        K = self.spike_k(bnk(K))

        Q = Q.reshape(T, -1, self.n_heads, self.head_dim)
        V = V.reshape(T, -1, self.n_heads, self.head_dim)
        K = K.reshape(T, -1, self.n_heads, self.head_dim)
        Q = Q.transpose(2, 1)
        V = V.transpose(2, 1)
        K = K.transpose(2, 1)  # [T, n_heads, cur_pos, head_dim]

        QK = Q.mul(K).sum(dim=-2, keepdim=True) # [T, n_heads, 1, head_dim]
        QK = self.talking_heads(QK)
        QKV = V.mul(QK) # [T, n_heads, cur_pos, head_dim]

        QKV = QKV.transpose(2, 1)
        QKV = QKV.reshape(T, -1, self.dim)
        QKV = bno(QKV @ self.wo)
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
        self.linear1 = nn.Linear(self.dim, self.hidden).to(self.args.device)
        self.linear2 = nn.Linear(self.hidden, self.dim).to(self.args.device)
        
    def forward(self, x, cur_pos):
        T, S, _ = x.shape
        bn1 = nn.BatchNorm1d(cur_pos).to(self.args.device)
        bn2 = nn.BatchNorm1d(cur_pos).to(self.args.device)

        out = self.init_spike(x)
        out = self.linear1(out)
        out = bn1(out)
        out = self.spike1(out)
        out = self.linear2(out)
        out = bn2(out)
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
        T, S, _ = x.shape
        bn = nn.BatchNorm1d(cur_pos).to(self.args.device)

        out = self.init_spike(x) # [T, S, D]
        out = self.output(out)
        out = bn(out)
        out = self.spike(out)
        out = out.mean(0)
        out = out / out.sum(dim=-1, keepdim=True)
        return out[cur_pos-1, :].clone()

    def reset(self):
        self.spike.reset()
        self.init_spike.reset()
