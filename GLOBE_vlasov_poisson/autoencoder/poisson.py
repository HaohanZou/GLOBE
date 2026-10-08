"""Electric field from charge density on the grounded cylindrical bore.

Solves  -Lap phi = rho  (packed units: eps0 = 1, lengths in units of L_scale)
with phi = 0 on the analytic cylinder r = R and periodic z, then E = -grad phi.
The Laplacian is the 7-point cell-centred stencil with Shortley-Weller cut-cell
arms to the wall; rho is restricted to the vacuum cells. The system is solved
matrix-free with Jacobi-preconditioned BiCGStab.

  python -m autoencoder.poisson --store <store> --run 9 --frames 60 [--ckpt <ae>]

derives phi and E from the stored rho of a full-field run (and, with --ckpt,
from its autoencoder reconstruction) and reports the relative L2 error against
the stored phi and E on the interior vacuum cells. --convention selects the
finite-difference registration of the stored E (centred or forward).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .dataset import STORE, BunchedBeamStore


class CylPoisson:
    """Operator A = -Lap on (..., Z, Y, X) tensors; geometry from BunchedBeamStore.geometry()."""

    def __init__(self, geom, device="cpu", dtype=torch.float64, theta_min=1e-6):
        self.dx, self.dy, self.dz = geom["dx"], geom["dy"], geom["dz"]
        xc = torch.as_tensor(np.asarray(geom["xc"]), device=device, dtype=dtype)
        yc = torch.as_tensor(np.asarray(geom["yc"]), device=device, dtype=dtype)
        X, Y = xc[None, :].expand(len(yc), len(xc)), yc[:, None].expand(len(yc), len(xc))
        R = geom["bore_radius"]
        self.vac = (X ** 2 + Y ** 2) < R ** 2

        def arm(axis, sign):
            """Fraction of the cell size from each vacuum centre to the wall along +/-axis."""
            if axis == "x":
                nb, h, along, other = torch.roll(self.vac, -sign, 1), self.dx, X, Y
            else:
                nb, h, along, other = torch.roll(self.vac, -sign, 0), self.dy, Y, X
            t = sign * (-along + sign * (R ** 2 - other ** 2).clamp_min(0.0).sqrt())
            theta = torch.where(self.vac & ~nb, (t / h).clamp(theta_min, 1.0), torch.ones_like(along))
            return theta, (nb & self.vac).to(dtype)

        (tE, oE), (tW, oW) = arm("x", +1), arm("x", -1)
        (tN, oN), (tS, oS) = arm("y", +1), arm("y", -1)
        v = self.vac.to(dtype)
        self.cE = 2.0 / (self.dx ** 2 * tE * (tE + tW)) * oE * v
        self.cW = 2.0 / (self.dx ** 2 * tW * (tE + tW)) * oW * v
        self.cN = 2.0 / (self.dy ** 2 * tN * (tN + tS)) * oN * v
        self.cS = 2.0 / (self.dy ** 2 * tS * (tN + tS)) * oS * v
        self.cZ = 1.0 / self.dz ** 2
        self.diag = (2.0 / (self.dx ** 2 * tE * tW) + 2.0 / (self.dy ** 2 * tN * tS) + 2.0 * self.cZ) * v
        self.inv_diag = torch.where(self.diag > 0, 1.0 / self.diag, torch.zeros_like(self.diag))
        self.interior = self.vac.clone()          # vacuum cells whose x/y neighbours are vacuum
        for shift, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            self.interior &= torch.roll(self.vac, shift, ax)

    def apply(self, phi):
        p = phi * self.vac
        out = self.diag * p
        out = out - self.cE * torch.roll(p, -1, -1) - self.cW * torch.roll(p, 1, -1)
        out = out - self.cN * torch.roll(p, -1, -2) - self.cS * torch.roll(p, 1, -2)
        out = out - self.cZ * (torch.roll(p, -1, -3) + torch.roll(p, 1, -3))
        return out * self.vac

    def solve(self, rhs, tol=1e-8, max_iters=20000):
        """Jacobi-preconditioned BiCGStab for A phi = rhs. Returns (phi, iterations, residual)."""
        b = rhs * self.vac
        x = torch.zeros_like(b)
        r = b - self.apply(x)
        r0 = r.clone()
        rho_old = alpha = omega = torch.ones((), dtype=b.dtype, device=b.device)
        p = torch.zeros_like(b)
        vv = torch.zeros_like(b)
        b_norm = torch.linalg.vector_norm(b).clamp_min(1e-300)
        resid = float("nan")
        for it in range(1, max_iters + 1):
            rho = torch.sum(r0 * r)
            if rho.abs() < 1e-300:
                break
            p = r + (rho / rho_old) * (alpha / omega) * (p - omega * vv)
            ph = self.inv_diag * p
            vv = self.apply(ph)
            alpha = rho / torch.sum(r0 * vv)
            s = r - alpha * vv
            resid = float(torch.linalg.vector_norm(s) / b_norm)
            if resid < tol:
                return (x + alpha * ph) * self.vac, it, resid
            sh = self.inv_diag * s
            t = self.apply(sh)
            omega = torch.sum(t * s) / torch.sum(t * t).clamp_min(1e-300)
            x = x + alpha * ph + omega * sh
            r = s - omega * t
            resid = float(torch.linalg.vector_norm(r) / b_norm)
            if resid < tol:
                break
            rho_old = rho
        return x * self.vac, it, resid

    def efield(self, phi, convention="centered"):
        """E = -grad phi as (Ex, Ey, Ez); 'centered' or 'forward' differences, matching
        the registration of the stored fields (periodic wrap; y/x rolls never cross the vacuum)."""
        p = phi * self.vac
        if convention == "centered":
            g = lambda d, h: (torch.roll(p, -1, d) - torch.roll(p, 1, d)) / (2 * h)
        else:
            g = lambda d, h: (torch.roll(p, -1, d) - p) / h
        return tuple(-g(d, h) * self.vac for d, h in ((-1, self.dx), (-2, self.dy), (-3, self.dz)))

    def fields(self, rho, convention="centered", tol=1e-8, max_iters=20000):
        """rho -> (phi, (Ex, Ey, Ez), iterations, residual)."""
        phi, it, resid = self.solve(rho, tol, max_iters)
        return phi, self.efield(phi, convention), it, resid


def rel(a, b, mask):
    return float(torch.linalg.vector_norm((a - b)[mask]) / torch.linalg.vector_norm(b[mask]).clamp_min(1e-30))


def rel_vec(ea, eb, mask):
    num = sum(float(((x - y) ** 2)[mask].sum()) for x, y in zip(ea, eb))
    den = sum(float((y ** 2)[mask].sum()) for y in eb)
    return float(np.sqrt(num / max(den, 1e-30)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--run", type=int, required=True, help="a full-field run")
    ap.add_argument("--frames", default="60")
    ap.add_argument("--ckpt", default="", help="autoencoder checkpoint: also score its reconstruction")
    ap.add_argument("--convention", default="centered", choices=["centered", "forward"],
                    help="finite-difference registration of the stored E")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args(argv)
    dev, conv = args.device, args.convention
    st = BunchedBeamStore(args.store)
    ev = st.eval_channels(args.run)
    if ev is None:
        raise SystemExit(f"run {args.run} stores rho only; pick a full-field run")
    arr, ch = ev
    ci = {c: i for i, c in enumerate(ch)}
    op = CylPoisson(st.geometry(), dev)
    mask = op.interior[None].expand(st.Z, st.Y, st.X)
    ae = None
    if args.ckpt:
        from .model import build_model
        ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
        ae = build_model(argparse.Namespace(**ck["args"])).to(dev).eval()
        ae.load_state_dict(ck["model"])
    print(f"run {args.run}: grid {st.Z}x{st.Y}x{st.X}, {int(op.vac.sum())} vacuum cells per plane, "
          f"stored-E convention {conv}", flush=True)
    for t in [int(f) for f in args.frames.split(",") if f]:
        f = torch.from_numpy(np.asarray(arr[t], np.float32)).to(dev, torch.float64)
        rho_s, phi_s = f[ci["rho"]], f[ci["phi"]]
        E_s = tuple(f[ci[c]] for c in ("Ex", "Ey", "Ez"))
        sources = {"stored rho": rho_s}
        if ae is not None:
            with torch.no_grad():
                sources["reconstructed rho"] = ae(rho_s[None, None].float())[0][0, 0].double()
        for name, rho in sources.items():
            phi, E, it, resid = op.fields(rho, conv, tol=args.tol)
            print(f"frame {t:3d} | {name:18s} | rho rel-L2 {rel(rho, rho_s, mask):.2e} | "
                  f"phi rel-L2 {rel(phi, phi_s, mask):.2e} | E rel-L2 {rel_vec(E, E_s, mask):.2e} | "
                  f"{it} iterations, residual {resid:.1e}", flush=True)


if __name__ == "__main__":
    main()
