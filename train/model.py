"""A GPT small enough for a microcontroller.

Same math as firmware/lib/tinyai/tinyai.c: learned positions, pre-RMSNorm,
multi-head causal attention, GELU MLP, output head tied to the embedding.
No biases anywhere — fewer tensors to ship and quantize."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 97
    ctx: int = 96
    dim: int = 128
    layers: int = 4
    heads: int = 4
    hidden: int = 384


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.w


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.n1, self.n2 = RMSNorm(c.dim), RMSNorm(c.dim)
        self.wq = nn.Linear(c.dim, c.dim, bias=False)
        self.wk = nn.Linear(c.dim, c.dim, bias=False)
        self.wv = nn.Linear(c.dim, c.dim, bias=False)
        self.wo = nn.Linear(c.dim, c.dim, bias=False)
        self.w1 = nn.Linear(c.dim, c.hidden, bias=False)
        self.w2 = nn.Linear(c.hidden, c.dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h = self.n1(x)
        split = lambda t: t.view(B, T, self.heads, D // self.heads).transpose(1, 2)
        q, k, v = split(self.wq(h)), split(self.wk(h)), split(self.wv(h))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(a.transpose(1, 2).reshape(B, T, D))
        return x + self.w2(F.gelu(self.w1(self.n2(x)), approximate="tanh"))


class TinyGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.dim)
        self.pos = nn.Embedding(c.ctx, c.dim)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = RMSNorm(c.dim)
        for n, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * c.layers) if n.endswith(("wo.weight", "w2.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.norm(x) @ self.tok.weight.T
