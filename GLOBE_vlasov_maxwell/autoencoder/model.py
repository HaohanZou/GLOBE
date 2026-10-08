"""3D convolutional autoencoder for the field stack with a constraint-exact decoder.

Encoder: the 10 store channels (scaled per channel), a frozen orthonormal Haar
transform (isotropic /2), then strided convolutions with parameter-free
space<->channel shortcuts; GroupNorm + SiLU residual blocks. The decoder
mirrors it and emits the 9 dynamical channels (driver, witness, wake
perturbation, E, B); the two spectral projections (maxwell.py) then impose
div B = 0 and div E = rho. The ion background is a fixed profile and is not
predicted. Latent (B, latent_ch, Z/8, Y/8, X/8).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (B, CHANNEL_RMS, CHANNEL_WEIGHTS, CHANNELS, DECODED, E, FLOOR_RMS,
                        INPUT_RMS, RHO, SEAM_CELLS)
from .maxwell import JointGaussProjection, WeightedDivFreeProjection

STRIDES = ((2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 1, 1))        # Haar stem + three stages: /8 isotropic


def _gn(ch):
    return nn.GroupNorm(min(8, ch), ch)


def _prod(t):
    p = 1
    for v in t:
        p *= v
    return p


def _space_to_channel(x, s):
    Bn, C, Z, Y, X = x.shape
    sz, sy, sx = s
    x = x.reshape(Bn, C, Z // sz, sz, Y // sy, sy, X // sx, sx)
    return x.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(Bn, C * sz * sy * sx, Z // sz, Y // sy, X // sx)


def _channel_to_space(x, s):
    Bn, C, Z, Y, X = x.shape
    sz, sy, sx = s
    c = C // (sz * sy * sx)
    x = x.reshape(Bn, c, sz, sy, sx, Z, Y, X)
    return x.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(Bn, c, Z * sz, Y * sy, X * sx)


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
    """One-level orthonormal 3D Haar transform per channel; inv(fwd(x)) == x."""

    def __init__(self, ch):
        super().__init__()
        lo = torch.tensor([1.0, 1.0]) / 2 ** 0.5
        hi = torch.tensor([1.0, -1.0]) / 2 ** 0.5
        ker = torch.stack([torch.einsum("i,j,k->ijk", a, b, c)
                           for a in (lo, hi) for b in (lo, hi) for c in (lo, hi)])[:, None]
        self.register_buffer("w", ker.repeat(ch, 1, 1, 1, 1))
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


class Backbone(nn.Module):
    """Convolutional encoder/decoder; the decoder may emit a different channel count."""

    def __init__(self, in_ch, out_ch, base, ch_mult, latent_ch, strides):
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
        self.to_subbands = nn.Conv3d(chs[0], out_ch * 8, 3, 1, 1)
        self.haar_out = Haar3D(out_ch) if out_ch != in_ch else self.haar

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
        return self.haar_out.inv(self.to_subbands(h))


def projection_metric():
    """Per-channel scale of the projections = the metric of the training loss:
    max(dataset RMS, floor) / sqrt(channel weight), weights normalized to mean 1."""
    w = torch.tensor(CHANNEL_WEIGHTS)
    w = w * (w.numel() / w.sum())
    return [max(r, f) / float(wi) ** 0.5 for r, f, wi in zip(CHANNEL_RMS, FLOOR_RMS, w.tolist())]


class VlasovMaxwellAE(nn.Module):
    def __init__(self, grid_shape, h, base=64, ch_mult=(1, 2, 4, 4), latent_ch=320, seam_cells=SEAM_CELLS):
        super().__init__()
        self.grid_shape = tuple(int(n) for n in grid_shape)
        self.backbone = Backbone(len(CHANNELS), len(DECODED), base, tuple(ch_mult), latent_ch, STRIDES)
        for n, f in zip(self.grid_shape, self.backbone.axis_factors()):
            if n % f:
                raise ValueError(f"grid {self.grid_shape} is not divisible by the stride factors")
        self.register_buffer("input_scale", torch.tensor(INPUT_RMS).view(1, -1, 1, 1, 1), persistent=False)
        s = projection_metric()
        self.proj_gauss = JointGaussProjection(self.grid_shape, h, s[RHO], s[E], seam_cells)
        self.proj_B = WeightedDivFreeProjection(self.grid_shape, h, s[B])

    @property
    def latent_ch(self):
        return self.backbone.latent_ch

    def latent_grid(self):
        return tuple(n // f for n, f in zip(self.grid_shape, self.backbone.axis_factors()))

    def encode(self, x):
        """x (B, 10, Z, Y, X) in the model input view (dataset.wake_view)."""
        return self.backbone.encode(x / self.input_scale.to(x.dtype))

    def decode(self, z):
        """z -> (B, 9, Z, Y, X) fields in constants.DECODED order, after the Maxwell projections."""
        raw = self.backbone.decode(z)
        rho, Ef = self.proj_gauss(raw[:, RHO], raw[:, E])
        return torch.cat((rho, Ef, self.proj_B(raw[:, B])), dim=1)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def build_model(args, grid_shape, h):
    """Model from CLI args (or the args dict of a checkpoint) and the store's grid and spacing."""
    return VlasovMaxwellAE(grid_shape, h, base=args.base, ch_mult=tuple(args.ch_mult), latent_ch=args.latent_ch)


def expand_to_store_channels(fields9, rho_ion):
    """Decoded fields (B, 9, Z, Y, X) + the ion background (Z, Y, X) or (B, Z, Y, X)
    -> the 10 store channels (rho_plasma_e = delta_rho_e - rho_ion)."""
    ion = rho_ion.to(fields9).expand(fields9.shape[0], *rho_ion.shape[-3:]).unsqueeze(1)
    return torch.cat((fields9[:, 0:2], fields9[:, 2:3] - ion, ion, fields9[:, 3:9]), dim=1)
