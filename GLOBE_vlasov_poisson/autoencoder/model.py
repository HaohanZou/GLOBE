"""3D convolutional autoencoder for charge-density frames.

Encoder: a frozen orthonormal Haar transform (isotropic /2, 1 -> 8 subband
channels) followed by strided convolutions; the decoder mirrors it with
trilinear upsampling and the inverse Haar. Every learned down/up stage carries
a parameter-free space<->channel shortcut. GroupNorm + SiLU residual blocks;
latent (B, latent_ch, Z/16, Y/8, X/8) for the default strides.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# per-stage (z, y, x) strides: Haar stem, then three learned stages
DEFAULT_STRIDES = ((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 1, 1))


def _gn(ch):
    return nn.GroupNorm(min(8, ch), ch)


def _prod(t):
    p = 1
    for v in t:
        p *= v
    return p


def _space_to_channel(x, s):
    B, C, Z, Y, X = x.shape
    sz, sy, sx = s
    x = x.reshape(B, C, Z // sz, sz, Y // sy, sy, X // sx, sx)
    return x.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(B, C * sz * sy * sx, Z // sz, Y // sy, X // sx)


def _channel_to_space(x, s):
    B, C, Z, Y, X = x.shape
    sz, sy, sx = s
    c = C // (sz * sy * sx)
    x = x.reshape(B, c, sz, sy, sx, Z, Y, X)
    return x.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(B, c, Z * sz, Y * sy, X * sx)


class _ChanResample(nn.Module):
    """Parameter-free channel resize: group-average to reduce, repeat to expand."""

    def __init__(self, cin, cout):
        super().__init__()
        self.reduce = cin >= cout
        self.g = cin // cout if self.reduce else cout // cin
        self.cout = cout

    def forward(self, x):
        if self.reduce:
            return x.reshape(x.shape[0], self.cout, self.g, *x.shape[2:]).mean(2)
        return x.repeat_interleave(self.g, dim=1)


class Haar3D(nn.Module):
    """One-level orthonormal 3D Haar transform; inv(fwd(x)) == x."""

    def __init__(self, ch):
        super().__init__()
        lo = torch.tensor([1.0, 1.0]) / 2 ** 0.5
        hi = torch.tensor([1.0, -1.0]) / 2 ** 0.5
        ker = torch.stack([torch.einsum("i,j,k->ijk", a, b, c)
                           for a in (lo, hi) for b in (lo, hi) for c in (lo, hi)])[:, None]
        self.register_buffer("w", ker.repeat(ch, 1, 1, 1, 1))   # (ch*8, 1, 2, 2, 2)
        self.ch = ch

    def fwd(self, x):
        return F.conv3d(x, self.w, stride=2, groups=self.ch)

    def inv(self, y):
        return F.conv_transpose3d(y, self.w, stride=2, groups=self.ch)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.n1, self.c1 = _gn(ch), nn.Conv3d(ch, ch, 3, 1, 1)
        self.n2, self.c2 = _gn(ch), nn.Conv3d(ch, ch, 3, 1, 1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        return x + self.c2(F.silu(self.n2(h)))


class Down(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.op = nn.Conv3d(cin, cout, tuple(4 if s == 2 else 3 for s in stride), stride, 1)
        self.stride = tuple(stride)
        self.short = _ChanResample(cin * _prod(stride), cout)

    def forward(self, x):
        return self.op(x) + self.short(_space_to_channel(x, self.stride))


class Up(nn.Module):
    def __init__(self, cin, cout, scale):
        super().__init__()
        self.scale = tuple(int(s) for s in scale)
        self.op = nn.Conv3d(cin, cout, 3, 1, 1)
        self.short = _ChanResample(cin, cout * _prod(self.scale))

    def forward(self, x):
        h = F.interpolate(x, scale_factor=tuple(float(s) for s in self.scale),
                          mode="trilinear", align_corners=False)
        return self.op(h) + _channel_to_space(self.short(x), self.scale)


class Autoencoder(nn.Module):
    def __init__(self, in_ch=1, base=32, ch_mult=(1, 2, 4, 4), latent_ch=64,
                 strides=DEFAULT_STRIDES):
        super().__init__()
        chs = [base * m for m in ch_mult]
        assert len(strides) == len(chs) and tuple(strides[0]) == (2, 2, 2)
        self.strides = [tuple(s) for s in strides]
        self.latent_ch = latent_ch
        self.haar = Haar3D(in_ch)
        self.stem = nn.Conv3d(in_ch * 8, chs[0], 3, 1, 1)
        self.enc = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.enc.extend([ResBlock(chs[i]), Down(chs[i], chs[i + 1], self.strides[i + 1])])
        self.enc_post = ResBlock(chs[-1])
        self.to_latent = nn.Conv3d(chs[-1], latent_ch, 1)
        self.from_latent = nn.Conv3d(latent_ch, chs[-1], 1)
        self.dec_pre = ResBlock(chs[-1])
        self.dec = nn.ModuleList()
        for i in reversed(range(len(chs) - 1)):
            self.dec.extend([Up(chs[i + 1], chs[i], self.strides[i + 1]), ResBlock(chs[i])])
        self.to_subbands = nn.Conv3d(chs[0], in_ch * 8, 3, 1, 1)

    def axis_factors(self):
        return tuple(_prod(s[a] for s in self.strides) for a in range(3))

    def encode(self, x):
        h = self.stem(self.haar.fwd(x))
        for m in self.enc:
            h = m(h)
        return self.to_latent(self.enc_post(h))

    def decode(self, z):
        h = self.dec_pre(self.from_latent(z))
        for m in self.dec:
            h = m(h)
        return self.haar.inv(self.to_subbands(h))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def build_model(args):
    """Model from CLI args or from the args dict stored in a checkpoint."""
    return Autoencoder(base=args.base, ch_mult=tuple(args.ch_mult), latent_ch=args.latent_ch)
