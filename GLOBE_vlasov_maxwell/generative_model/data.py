"""Windows for residual flow matching.

A sample holds k + d_max*h consecutive standardized latent frames ending at t
(hist_ext), the h target frames after t and the standardized controls. The
frozen operator builds the training history from the k true frames ending at
t - d*h by rolling itself d times ("chain" history of depth d). Windows too
early for the deepest roll are front-padded by edge replication; ``feasible``
is the largest depth a window supports and padded frames are never consumed.
"""
from __future__ import annotations

import numpy as np
import torch

from operator_learning.data import load_meta, run_array


class ResidualWindows(torch.utils.data.Dataset):
    def __init__(self, root, split, k, h, zmean, zstd, cmean, cstd, d_max=1):
        assert k >= h and d_max >= 1
        self.k, self.h, self.d_max = k, h, int(d_max)
        self.span = k + self.d_max * h
        meta = load_meta(root)
        self.T = meta["T"]
        self.runs, self.Z, self.C = [], [], []
        for r, info in meta["runs"].items():
            if info["split"] != split:
                continue
            self.runs.append(int(r))
            self.Z.append(run_array(root, r))
            self.C.append((np.asarray(info["controls"], np.float32) - cmean) / cstd)
        if not self.runs:
            raise RuntimeError(f"no runs in split {split!r} under {root}")
        self.zmean, self.zstd = zmean, zstd
        self.items = [(i, t) for i in range(len(self.Z)) for t in range(k + h - 1, self.T - h)]
        if not self.items:
            raise RuntimeError(f"T={self.T} is too short for k={k}, h={h}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, t = self.items[idx]
        z = self.Z[i]
        lo = t - self.span + 1
        hist_ext = np.asarray(z[max(lo, 0): t + 1], np.float32)
        if lo < 0:
            hist_ext = np.concatenate([np.repeat(hist_ext[:1], -lo, axis=0), hist_ext], axis=0)
        hist_ext = (hist_ext - self.zmean) / self.zstd
        targ = (np.asarray(z[t + 1: t + 1 + self.h], np.float32) - self.zmean) / self.zstd
        return {"hist_ext": torch.from_numpy(hist_ext), "targ": torch.from_numpy(targ),
                "ctrl": torch.from_numpy(self.C[i]),
                "feasible": torch.tensor(min(self.d_max, (t - self.k + 1) // self.h), dtype=torch.long)}


@torch.no_grad()
def chain_roll(det_fwd, hist_ext, ctrl, k, h, d_eff):
    """k-frame histories ending at t whose last min(k, d*h) frames are the frozen operator's
    own forecasts: sample i starts from the k true frames ending at t - d_eff[i]*h and rolls
    the operator d_eff[i] times. det_fwd(hist, ctrl) -> (B, h, C, g); samples are grouped by depth."""
    span = hist_ext.shape[1]
    out = torch.empty(hist_ext.shape[0], k, *hist_ext.shape[2:], device=hist_ext.device, dtype=hist_ext.dtype)
    for e in sorted(set(int(v) for v in d_eff.tolist())):
        idx = (d_eff == e).nonzero(as_tuple=True)[0].to(hist_ext.device)
        hh, cj = hist_ext[idx, span - k - e * h: span - e * h], ctrl[idx]
        for _ in range(e):
            hh = torch.cat([hh[:, h:], det_fwd(hh, cj)], dim=1)
        out[idx] = hh
    return out
