"""A GPT small enough for a microcontroller.

Same math as firmware/lib/tinyai/tinyai.c: learned positions, pre-RMSNorm,
grouped-query causal attention, GELU MLP, output head tied to the embedding.
No biases anywhere — fewer tensors to ship and quantize.

Linear weights ship as int4 (groups of 32 with an fp16 scale). With
`model.qat = True` training sees exactly those rounded weights, so the
network learns to be accurate *after* quantization."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP = 32  # int4 weights share one scale per 32 inputs


@dataclass
class Config:
    vocab: int = 97
    ctx: int = 96
    dim: int = 192
    layers: int = 6
    heads: int = 4
    kv_heads: int = 2
    hidden: int = 576


def quantize_int4(w):
    """Round weights exactly like train/export.py and the C engine do."""
    out, cols = w.shape
    g = w.reshape(out, cols // GROUP, GROUP)
    # scales ship as fp16; the floor keeps them out of fp16's subnormal range
    scale = (g.abs().amax(-1, keepdim=True) / 7).clamp(min=6.2e-5).half().float()
    return (torch.clamp(torch.round(g / scale), -7, 7) * scale).reshape(out, cols)


class QLinear(nn.Linear):
    def __init__(self, i, o):
        super().__init__(i, o, bias=False)
        self.qat = False

    def forward(self, x):
        w = self.weight
        if self.qat:
            w = w + (quantize_int4(w) - w).detach()  # straight-through estimator
        return F.linear(x, w)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.w


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.kv_heads, self.hd = c.heads, c.kv_heads, c.dim // c.heads
        kv_dim = c.kv_heads * self.hd
        self.n1, self.n2 = RMSNorm(c.dim), RMSNorm(c.dim)
        self.wq = QLinear(c.dim, c.dim)
        self.wk = QLinear(c.dim, kv_dim)
        self.wv = QLinear(c.dim, kv_dim)
        self.wo = QLinear(c.dim, c.dim)
        self.w1 = QLinear(c.dim, c.hidden)
        self.w2 = QLinear(c.hidden, c.dim)

    def forward(self, x):
        B, T, D = x.shape
        h = self.n1(x)
        q = self.wq(h).view(B, T, self.heads, self.hd).transpose(1, 2)
        k = self.wk(h).view(B, T, self.kv_heads, self.hd).transpose(1, 2)
        v = self.wv(h).view(B, T, self.kv_heads, self.hd).transpose(1, 2)
        rep = self.heads // self.kv_heads  # grouped-query attention: heads share K/V
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
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

    def set_qat(self, on):
        for m in self.modules():
            if isinstance(m, QLinear):
                m.qat = on

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.norm(x) @ self.tok.weight.T
