"""Parameter-free constraint projections for the decoded fields.

The store holds cell-centred fields evolved by a finite-difference Maxwell
solver (Cole-Karkkainen-Cowan stencil) on a periodic grid with spacing
h = (dz, dy, dx) in plasma units. Maxwell's two constraint equations,
div B = 0 (in the solver's matched discrete sense) and div E = rho (wide
central differences), are linear in the channel values, so the admissible
fields form a linear subspace. Both operators are convolutions, hence
diagonal in Fourier space, and the minimum weighted-norm correction onto
the subspace is one scalar division per mode:

    dY_c(k) = -s_c^2 conj(a_c(k)) D(k) / sum_c' s_c'^2 |a_c'(k)|^2,   D(k) = sum_c a_c(k) Y_c(k)

with a_c the constraint symbols and s_c the per-channel metric (the metric of
the training loss). The projections carry no parameters.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

SPATIAL = (-3, -2, -1)
_DIM = {"x": -1, "y": -2, "z": -3}
_HI = {"z": 0, "y": 1, "x": 2}
_TRANSVERSE = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}


def _dc(f, c, h):
    """Wide central difference along component c's axis."""
    return (torch.roll(f, -1, _DIM[c]) - torch.roll(f, 1, _DIM[c])) / (2.0 * h[_HI[c]])


def divc(F, h):
    """Wide central-difference divergence of a cell-centred vector (..., 3, Z, Y, X)."""
    Fx, Fy, Fz = F.unbind(-4)
    return _dc(Fx, "x", h) + _dc(Fy, "y", h) + _dc(Fz, "z", h)


def ckc_coeffs(h):
    """Cole-Karkkainen-Cowan stencil coefficients for h = (dz, dy, dx)."""
    inv = {"x": 1.0 / h[2], "y": 1.0 / h[1], "z": 1.0 / h[0]}
    delta = max(inv.values())
    r = {c: (v / delta) ** 2 for c, v in inv.items()}
    inv_r_fac = 1.0 / (r["y"] * r["z"] + r["z"] * r["x"] + r["x"] * r["y"])
    beta = 0.125 * (1.0 - r["x"] * r["y"] * r["z"] * inv_r_fac)
    gamma = {c: r[t1] * r[t2] * (0.0625 - 0.125 * r[t1] * r[t2] * inv_r_fac)
             for c, (t1, t2) in _TRANSVERSE.items()}
    return r, beta, gamma


def _theta_axes(shape, device):
    """Angular grid frequencies per axis on the rfft grid (real last axis), float64."""
    th = [2 * math.pi * torch.fft.fftfreq(shape[0], device=device, dtype=torch.float64),
          2 * math.pi * torch.fft.fftfreq(shape[1], device=device, dtype=torch.float64),
          2 * math.pi * torch.fft.rfftfreq(shape[2], device=device, dtype=torch.float64)]
    return dict(zip("zyx", torch.meshgrid(*th, indexing="ij")))


def matched_div_symbols(shape, h, device="cpu"):
    """Fourier symbols (3, Z, Y, X//2+1) of the solver's matched divergence of the stored
    (cell-centred) B: forward difference x two transverse averages x the CKC smoothing."""
    TH = _theta_axes(shape, device)
    r, beta, gamma = ckc_coeffs(h)
    cos = {c: torch.cos(TH[c]) for c in "xyz"}
    av = {c: (1.0 + torch.exp(1j * TH[c])) / 2.0 for c in "xyz"}
    zero = torch.zeros((), dtype=torch.complex128, device=device)
    av = {c: torch.where(av[c].abs() < 1e-12, zero, av[c]) for c in "xyz"}   # exact zeros at Nyquist
    out = []
    for c in "xyz":
        t1, t2 = _TRANSVERSE[c]
        T = (1.0 - 2.0 * beta * (r[t1] * (1.0 - cos[t1]) + r[t2] * (1.0 - cos[t2]))
             - 4.0 * gamma[c] * (1.0 - cos[t1] * cos[t2]))
        out.append((torch.exp(1j * TH[c]) - 1.0) / h[_HI[c]] * av[t1] * av[t2] * T)
    return torch.stack(out)


def divc_symbols(shape, h, device="cpu"):
    """Fourier symbols (3, Z, Y, X//2+1) of ``divc``: i sin(theta_c) / h_c."""
    TH = _theta_axes(shape, device)
    return torch.stack([1j * torch.sin(TH[c]) / h[_HI[c]] for c in "xyz"])


def _face_distance(shape, device):
    """Per-axis distance (in cells) to the nearest domain face, broadcastable to (Z, Y, X)."""
    out = []
    for ax, n in enumerate(shape):
        i = torch.arange(n, device=device, dtype=torch.float64)
        out.append(torch.minimum(i, (n - 1) - i).view(*([1] * ax), n, *([1] * (2 - ax))))
    return out


def boundary_mask(shape, cells, device="cpu", dtype=torch.float32):
    """1 on the interior, 0 within ``cells`` of any face."""
    dz, dy, dx = _face_distance(shape, device)
    return ((dz >= cells) & (dy >= cells) & (dx >= cells)).to(dtype)


def shell_taper(shape, cells, floor, device="cpu", dtype=torch.float32):
    """Raised-cosine weight from ``floor`` at every face to 1 ``cells`` inside."""
    dz, dy, dx = _face_distance(shape, device)
    n = float(max(int(cells), 1))
    t = torch.minimum(torch.minimum((dz / n).clamp(0, 1), (dy / n).clamp(0, 1)), (dx / n).clamp(0, 1))
    return (floor + (1.0 - floor) * 0.5 * (1.0 - torch.cos(math.pi * t))).to(dtype)


class WeightedDivFreeProjection(nn.Module):
    """B (..., 3, Z, Y, X) -> the matched divergence-free subspace. ``scales`` is the (3,)
    per-component metric. Buffers are rebuilt at construction and are not part of the state
    dict; the projection runs in fp32 with autocast disabled."""

    def __init__(self, grid_shape, h, scales, compute_dtype=torch.float32, tol=1e-12):
        super().__init__()
        self.grid_shape = tuple(int(n) for n in grid_shape)
        self.compute_dtype = compute_dtype
        s2 = torch.as_tensor([float(v) for v in scales], dtype=torch.float64).view(3, 1, 1, 1) ** 2
        d = matched_div_symbols(self.grid_shape, tuple(float(v) for v in h))
        den = (s2 * d.abs() ** 2).sum(0)
        inv = torch.where(den > den.max() * tol, 1.0 / den.clamp_min(1e-300), torch.zeros_like(den))
        cdt = torch.complex64 if compute_dtype == torch.float32 else torch.complex128
        self.register_buffer("dsym", d.to(cdt), persistent=False)
        self.register_buffer("wdconj", (s2 * d.conj()).to(cdt), persistent=False)
        self.register_buffer("inv_den", inv.to(compute_dtype), persistent=False)

    def forward(self, B):
        with torch.autocast(device_type=B.device.type, enabled=False):
            Bf = torch.fft.rfftn(B.to(self.compute_dtype), dim=SPATIAL)
            D = (self.dsym * Bf).sum(dim=-4, keepdim=True)
            return torch.fft.irfftn(Bf - self.wdconj * (D * self.inv_den), s=self.grid_shape, dim=SPATIAL)


class JointGaussProjection(nn.Module):
    """(rho (..., 3, Z, Y, X), E (..., 3, Z, Y, X)) -> the subspace where divc(E) equals the
    sum of the charge channels; the correction moves rho and E jointly. The residual is masked
    within ``seam_cells`` of the domain faces before the solve."""

    def __init__(self, grid_shape, h, rho_scales, e_scales, seam_cells=2, compute_dtype=torch.float32):
        super().__init__()
        self.grid_shape = tuple(int(n) for n in grid_shape)
        self.h = tuple(float(v) for v in h)
        self.compute_dtype = compute_dtype
        sr = torch.as_tensor([float(v) for v in rho_scales], dtype=torch.float64)
        se = torch.as_tensor([float(v) for v in e_scales], dtype=torch.float64).view(3, 1, 1, 1)
        dE = divc_symbols(self.grid_shape, self.h)
        den = sr.pow(2).sum() + (se ** 2 * dE.abs() ** 2).sum(0)
        cdt = torch.complex64 if compute_dtype == torch.float32 else torch.complex128
        self.register_buffer("weconj", (se ** 2 * dE.conj()).to(cdt), persistent=False)
        self.register_buffer("s2_rho", sr.pow(2).to(compute_dtype), persistent=False)
        self.register_buffer("inv_den", (1.0 / den).to(compute_dtype), persistent=False)
        self.register_buffer("seam", boundary_mask(self.grid_shape, seam_cells, dtype=compute_dtype)
                             if seam_cells > 0 else torch.ones(1, dtype=compute_dtype), persistent=False)

    def forward(self, rho, E):
        with torch.autocast(device_type=E.device.type, enabled=False):
            rho, E = rho.to(self.compute_dtype), E.to(self.compute_dtype)
            D = (divc(E, self.h) - rho.sum(dim=-4)) * self.seam
            c = (torch.fft.rfftn(D, dim=SPATIAL) * self.inv_den).unsqueeze(-4)
            corr = torch.fft.irfftn(torch.cat((self.s2_rho.view(-1, 1, 1, 1) * c, -self.weconj * c), dim=-4),
                                    s=self.grid_shape, dim=SPATIAL)
            return rho + corr[..., :3, :, :, :], E + corr[..., 3:, :, :, :]
