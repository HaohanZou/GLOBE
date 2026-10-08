"""Conditional rectified flow on the latent residual.

The frozen operator carries the conditional mean; this network is a velocity
field for the per-channel normalized residual r = (z_true - f_det) / sigma_r.
Each token carries (x_tau, deterministic forecast, history) for its grid
point, and the global conditioning is (controls, Fourier features of the flow
time tau). Same backbone as the operator, with RMS-normalized queries and
keys.

Amplitude normalizer (amp="wintok"): sigma_r is multiplied by a frozen,
pre-fit scale s_win(window) * s_tok(grid point), both predicted from the
forecast and the history. amp="none" uses sigma_r alone.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from operator_learning.model import CoordEmbed, LNOBlock


def time_feats(t, n=8):
    """Fourier features of the flow time tau in [0, 1]: (B, 1) -> (B, 1 + 2n)."""
    f = torch.arange(n, device=t.device, dtype=t.dtype)
    a = t * (2 * math.pi * (2.0 ** f))
    return torch.cat([t, torch.sin(a), torch.cos(a)], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        n = x.float()
        n = n * torch.rsqrt(n.pow(2).mean(-1, keepdim=True) + self.eps)
        return (n * self.weight.float()).to(x.dtype)


class QKNormAttention(nn.Module):
    """Dense self-attention with RMSNorm on queries and keys."""

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.qn = RMSNorm(dim // heads)
        self.kn = RMSNorm(dim // heads)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(self.qn(q), self.kn(k), v)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


class _ScaleHead(nn.Module):
    """3-layer MLP -> log scale, zero-initialized output (scale 1 at init), clamped."""

    def __init__(self, n_in, hidden, log_lo, log_hi):
        super().__init__()
        self.log_lo, self.log_hi = float(log_lo), float(log_hi)
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)


class ScalarAmplitude(_ScaleHead):
    """One log scale per window from the log RMS of every (frame, channel) of forecast + history."""

    def __init__(self, ch, k, h, hidden=128, log_lo=-4.0, log_hi=4.0):
        super().__init__((h + k) * ch, hidden, log_lo, log_hi)
        self.k, self.h, self.ch = k, h, ch

    def forward(self, fdet, hist):
        B = fdet.shape[0]
        x = torch.cat([fdet.reshape(B, self.h * self.ch, -1), hist.reshape(B, self.k * self.ch, -1)], 1)
        feat = 0.5 * x.pow(2).mean(-1).clamp_min(1e-12).log()
        return self.net(feat).squeeze(-1).clamp(self.log_lo, self.log_hi)               # (B,)


class TokenAmplitude(_ScaleHead):
    """One log scale per grid point from that point's forecast + history values."""

    def __init__(self, ch, k, h, hidden=128, log_lo=math.log(0.05), log_hi=math.log(10.0)):
        super().__init__((h + k) * ch, hidden, log_lo, log_hi)
        self.k, self.h, self.ch = k, h, ch

    def forward(self, fdet, hist):
        B, grid = fdet.shape[0], fdet.shape[-3:]
        N = grid[0] * grid[1] * grid[2]
        x = torch.cat([fdet.reshape(B, self.h * self.ch, N), hist.reshape(B, self.k * self.ch, N)], 1)
        return self.net(x.transpose(1, 2)).reshape(B, 1, 1, *grid).clamp(self.log_lo, self.log_hi)


class WinTokAmplitude(nn.Module):
    """log(s_win * s_tok); the token floor is stored so a reloaded model keeps its clamp."""

    def __init__(self, ch, k, h, hidden=128, tok_floor=0.05):
        super().__init__()
        self.win = ScalarAmplitude(ch, k, h, hidden)
        self.tok = TokenAmplitude(ch, k, h, hidden, log_lo=math.log(tok_floor))
        self.register_buffer("tok_log_floor", torch.tensor(math.log(tok_floor)))
        self.register_load_state_dict_post_hook(lambda m, _: setattr(m.tok, "log_lo", float(m.tok_log_floor)))

    def forward(self, fdet, hist):
        return self.win(fdet, hist).view(-1, 1, 1, 1, 1, 1) + self.tok(fdet, hist)   # (B, 1, 1, Z, Y, X)


class LatentResidualFM(nn.Module):
    def __init__(self, ch, k, h, dim=384, depth=10, heads=8, n_ctrl=8, ftau=8, mlp_ratio=4,
                 bands=6, amp="wintok", tok_floor=0.05):
        super().__init__()
        self.k, self.h, self.ch, self.ftau = k, h, ch, ftau
        self.lift = nn.Linear((2 * h + k) * ch, dim)            # x_tau + forecast + history
        self.coord = CoordEmbed(dim, bands)
        self.cond = nn.Sequential(nn.Linear(n_ctrl + 1 + 2 * ftau, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([LNOBlock(dim, heads, dim, mlp_ratio, attn=QKNormAttention)
                                     for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.head = nn.Linear(dim, h * ch)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.amp_mode = amp
        self.amp = WinTokAmplitude(ch, k, h, tok_floor=tok_floor) if amp == "wintok" else None

    def residual_scale(self, fdet, hist):
        """Multiplicative scale on sigma_r, broadcastable against (B, h, C, g); None for amp='none'."""
        return None if self.amp is None else self.amp(fdet, hist).exp()

    def forward(self, x_tau, fdet, hist, ctrl, tau):
        """x_tau, fdet (B, h, C, g); hist (B, k, C, g); ctrl (B, n_ctrl); tau (B,).
        Returns the velocity (B, h, C, g) in normalized-residual units."""
        B, grid = hist.shape[0], hist.shape[-3:]
        N = grid[0] * grid[1] * grid[2]
        x = torch.cat([x_tau.reshape(B, self.h * self.ch, N), fdet.reshape(B, self.h * self.ch, N),
                       hist.reshape(B, self.k * self.ch, N)], dim=1).transpose(1, 2)
        c = self.cond(torch.cat([ctrl, time_feats(tau[:, None], self.ftau)], dim=-1))
        y = self.lift(x)
        y = y + self.coord(grid, y.device, y.dtype)[None] + c[:, None, :]
        for b in self.blocks:
            y = b(y, c)
        return self.head(self.norm_out(y)).transpose(1, 2).reshape(B, self.h, self.ch, *grid)
