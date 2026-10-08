"""Latent neural operator: dense self-attention over all latent grid points.

Every latent voxel is a token and attention runs over all of them. Position
enters as Fourier features of the normalized grid coordinate. Blocks are
adaLN-Zero conditioned on (controls, drive phase) with zero-initialized gates;
with persist=True the head predicts a correction to the last history frame.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordEmbed(nn.Module):
    """Fourier features of the normalized grid coordinate -> dim."""

    def __init__(self, dim, bands=6):
        super().__init__()
        self.bands = bands
        self.proj = nn.Linear(3 * (2 * bands + 1), dim)

    def forward(self, shape, device, dtype):
        ax = [torch.linspace(-1.0, 1.0, s, device=device, dtype=dtype) for s in shape]
        g = torch.stack(torch.meshgrid(*ax, indexing="ij"), dim=-1)          # (Z, Y, X, 3)
        f = (2.0 ** torch.arange(self.bands, device=device, dtype=dtype)) * math.pi
        a = g[..., None] * f
        e = torch.cat([g, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], dim=-1)
        return self.proj(e.reshape(-1, e.shape[-1]))                         # (N, dim)


class DenseSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):                                                    # (B, N, D)
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class LNOBlock(nn.Module):
    """adaLN-Zero conditioned attention + MLP block."""

    def __init__(self, dim, heads, cond_dim, mlp_ratio=4, attn=DenseSelfAttention):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = attn(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Linear(dim * mlp_ratio, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        s1, c1, g1, s2, c2, g2 = self.ada(c).chunk(6, dim=-1)
        x = x + g1[:, None, :] * self.attn(modulate(self.n1(x), s1, c1))
        return x + g2[:, None, :] * self.mlp(modulate(self.n2(x), s2, c2))


class LatentNeuralOperator(nn.Module):
    """(k past latents, controls, drive phase) -> h future latents in one application.

    Latents are (B, k, C, Lz, Ly, Lx); ctrl (B, n_ctrl) standardized; phase (B, 2)."""

    def __init__(self, ch, k, h, dim=512, depth=12, heads=8, n_ctrl=3, mlp_ratio=4,
                 bands=6, persist=False):
        super().__init__()
        self.k, self.h, self.ch, self.persist = k, h, ch, persist
        self.lift = nn.Linear(k * ch + 1, dim)                # +1: constant input column
        self.coord = CoordEmbed(dim, bands)
        self.cond = nn.Sequential(nn.Linear(n_ctrl + 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([LNOBlock(dim, heads, dim, mlp_ratio) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.head = nn.Linear(dim, h * ch)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, hist, ctrl, phase):
        B, grid = hist.shape[0], hist.shape[-3:]
        N = grid[0] * grid[1] * grid[2]
        x = hist.reshape(B, self.k * self.ch, N).transpose(1, 2)
        c = self.cond(torch.cat([ctrl, phase], dim=-1))                      # (B, dim)
        y = self.lift(torch.cat([x, hist.new_ones(B, N, 1)], dim=-1))
        y = y + self.coord(grid, y.device, y.dtype)[None] + c[:, None, :]
        for b in self.blocks:
            y = b(y, c)
        out = self.head(self.norm_out(y)).transpose(1, 2).reshape(B, self.h, self.ch, *grid)
        if self.persist:      # predict a correction to the last history frame
            out = out + hist[:, -1:]
        return out


def load_operator(path, device):
    """Rebuild a trained operator from its checkpoint: (model, args, stats, step)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    if a.__dict__.get("time_cond") or a.__dict__.get("cond_add") is False:
        raise SystemExit(f"{path} uses a conditioning option this release does not implement")
    a.ae_sha256_16 = ck.get("ae_sha256_16")
    stats = {key: np.asarray(ck[key], np.float32) for key in ("zmean", "zstd", "cmean", "cstd")}
    net = LatentNeuralOperator(stats["zmean"].shape[0], a.k, a.h, dim=a.width, depth=a.blocks,
                               heads=a.heads, n_ctrl=len(stats["cmean"]), mlp_ratio=a.mlp_ratio,
                               persist=a.persist).to(device).eval()
    net.load_state_dict(ck["model"])
    return net, a, stats, int(ck.get("step", -1))
