"""Reconstruction objective: relative L2 + band-weighted spectral error + H1 error,
per channel, with energy floors for legitimately (near-)empty channels.

rel_l2:   the denominator of channel c is max(||tgt_c||^2, floor_c^2 * n_voxels),
          so an all-zero channel is supervised on an absolute scale instead of
          exploding the ratio; channels are combined by a weighted mean, and an
          optional per-voxel weight (the shell taper) applies to both error and
          target energy.
spectral, h1: channels whose target energy sits below the floor are excluded.
"""
from __future__ import annotations

import torch

SD = (-3, -2, -1)


def _energies(pred, tgt, spatial=None):
    """Per-channel error and target energies (..., C) and the effective voxel count."""
    err, t2 = (pred - tgt).pow(2), tgt.pow(2)
    if spatial is not None:
        err, t2 = err * spatial, t2 * spatial
        n = float(spatial.sum())
    else:
        n = float(tgt.shape[-3] * tgt.shape[-2] * tgt.shape[-1])
    return err.sum(SD), t2.sum(SD), n


def channel_mask(tgt, floor, spatial=None):
    """(..., C) bool: channels whose target energy clears the floor."""
    _, t2, n = _energies(tgt, tgt, spatial)
    return t2 >= floor.to(t2) ** 2 * n


def rel_l2(pred, tgt, floor=None, weights=None, spatial=None, reduce=True, eps=1e-8):
    err, den, n = _energies(pred, tgt, spatial)
    if floor is not None:
        den = torch.maximum(den, floor.to(den) ** 2 * n)
    r = (err / den.clamp_min(eps)).sqrt()                          # (..., C)
    if not reduce:
        return r
    if weights is None:
        return r.mean()
    w = weights.to(r)
    return ((r * w).sum(-1) / w.sum()).mean()


def _masked_mean(vals, mask):
    if mask is None:
        return vals.mean()
    m = mask.to(vals)
    return (vals * m).sum() / m.sum().clamp_min(1.0)


def spectral_loss(pred, tgt, hi_power=1.0, band_boost=5.0, floor=None, eps=1e-8):
    """Relative L2 between the wavenumber-weighted spectra w * F(pred) and w * F(tgt), per channel:
    w(k) = (1 + |k|/kmax)^hi_power, times band_boost on the middle third of |k|; rfft columns are
    weighted by their Hermitian multiplicity."""
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
    ratio = (w2 * (Fp - Ft).abs().pow(2)).sum(SD) / (w2 * Ft.abs().pow(2)).sum(SD).clamp_min(eps)
    return _masked_mean(ratio.sqrt(), None if floor is None else channel_mask(tgt, floor))


def _grad_axis(f, d):
    """Centred difference along dim d (one-sided at the boundary planes), unit spacing."""
    n = f.shape[d]
    fwd = torch.cat([f.narrow(d, 1, n - 1), f.narrow(d, n - 1, 1)], d)
    bwd = torch.cat([f.narrow(d, 0, 1), f.narrow(d, 0, n - 1)], d)
    scale = torch.ones(n, device=f.device, dtype=f.dtype)
    scale[0] = scale[-1] = 2.0
    shp = [1] * f.dim()
    shp[d] = n
    return (fwd - bwd) / 2.0 * scale.view(shp)


def h1_loss(pred, tgt, floor=None, spatial=None):
    """Sum over the three axes of the relative L2 of the finite-difference derivatives;
    sub-floor channels excluded."""
    mask = None if floor is None else channel_mask(tgt, floor, spatial)
    acc = pred.new_zeros(())
    for d in SD:
        with torch.no_grad():
            gt = _grad_axis(tgt, d)
        acc = acc + _masked_mean(rel_l2(_grad_axis(pred, d), gt, spatial=spatial, reduce=False), mask)
    return acc


def ae_loss(pred, tgt, floor, weights, spatial, w_spec=0.1, w_h1=0.1, hi_power=1.0, band_boost=5.0):
    """Weighted composite; returns (total, unweighted terms)."""
    zero = pred.new_zeros(())
    terms = {"rel_l2": rel_l2(pred, tgt, floor, weights, spatial),
             "spectral": spectral_loss(pred, tgt, hi_power, band_boost, floor=floor) if w_spec else zero,
             "h1": h1_loss(pred, tgt, floor, spatial) if w_h1 else zero}
    return terms["rel_l2"] + w_spec * terms["spectral"] + w_h1 * terms["h1"], terms
