"""Reconstruction objective: relative L2 + band-weighted spectral error + H1 error."""
from __future__ import annotations

import torch

SD = (-3, -2, -1)


def _rel(pred, tgt, eps=1e-8):
    """Relative L2 over the spatial dims, unreduced (..., C)."""
    return ((pred - tgt).pow(2).sum(SD) / tgt.pow(2).sum(SD).clamp_min(eps)).sqrt()


def rel_l2(pred, tgt):
    """Per-sample relative L2, averaged over batch and channel."""
    return _rel(pred, tgt).mean()


def spectral_loss(pred, tgt, hi_power=1.0, band_boost=5.0, eps=1e-8):
    """Relative L2 between the wavenumber-weighted spectra w * F(pred) and w * F(tgt).

    w(k) = (1 + |k|/kmax)^hi_power, multiplied by band_boost on the middle
    third of |k|. rfft columns are weighted by their Hermitian multiplicity,
    so the unweighted case equals the relative L2 (Parseval)."""
    N = pred.shape[-3:]
    Fp, Ft = torch.fft.rfftn(pred, dim=SD), torch.fft.rfftn(tgt, dim=SD)
    ks = [torch.fft.fftfreq(N[0], device=pred.device), torch.fft.fftfreq(N[1], device=pred.device),
          torch.fft.rfftfreq(N[2], device=pred.device)]
    kg = torch.meshgrid(*ks, indexing="ij")
    k = torch.sqrt(kg[0] ** 2 + kg[1] ** 2 + kg[2] ** 2)
    kmax = k.max().clamp_min(eps)
    w = (1.0 + k / kmax) ** hi_power
    w = torch.where((k > kmax / 3) & (k <= 2 * kmax / 3), w * band_boost, w)
    mult = torch.full((N[2] // 2 + 1,), 2.0, device=w.device)
    mult[0] = 1.0
    if N[2] % 2 == 0:
        mult[-1] = 1.0
    w2 = w * w * mult
    num = (w2 * (Fp - Ft).abs().pow(2)).sum(SD)
    den = (w2 * Ft.abs().pow(2)).sum(SD).clamp_min(eps)
    return (num / den).sqrt().mean()


def _grad(f):
    """Centred differences (one-sided at the boundary planes), unit spacing:
    (..., Z, Y, X) -> (..., 3, Z, Y, X)."""
    out = []
    for d in SD:
        n = f.shape[d]
        fwd = torch.cat([f.narrow(d, 1, n - 1), f.narrow(d, n - 1, 1)], d)
        bwd = torch.cat([f.narrow(d, 0, 1), f.narrow(d, 0, n - 1)], d)
        scale = torch.ones(n, device=f.device, dtype=f.dtype)
        scale[0] = scale[-1] = 2.0                      # boundary: one interval, not two
        shp = [1] * f.dim()
        shp[d] = n
        out.append((fwd - bwd) / 2.0 * scale.view(shp))
    return torch.stack(out, dim=-4)


def h1_loss(pred, tgt):
    """Sum over the three axes of the relative L2 of the finite-difference derivatives."""
    return _rel(_grad(pred), _grad(tgt)).sum(-1).mean()


def ae_loss(pred, tgt, w_spec=0.1, w_h1=0.1, hi_power=1.0, band_boost=5.0):
    """Weighted composite; returns (total, unweighted terms)."""
    zero = pred.new_zeros(())
    terms = {"rel_l2": rel_l2(pred, tgt),
             "spectral": spectral_loss(pred, tgt, hi_power, band_boost) if w_spec else zero,
             "h1": h1_loss(pred, tgt) if w_h1 else zero}
    return terms["rel_l2"] + w_spec * terms["spectral"] + w_h1 * terms["h1"], terms
