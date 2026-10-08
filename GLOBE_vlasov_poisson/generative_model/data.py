"""Windows for residual flow matching.

A sample holds k + h consecutive standardized latent frames ending at t
(hist_ext), the h target frames after t, the standardized controls and the
drive phase of frame t. The frozen operator predicts the last h history frames
from the k frames before them ("chain" history).
"""
from __future__ import annotations

import numpy as np
import torch

from operator_learning.data import load_meta, phase_feats, run_array


class ResidualWindows(torch.utils.data.Dataset):
    def __init__(self, root, split, k, h, zmean, zstd, cmean, cstd):
        assert k >= h, "chain histories need k >= h"
        self.k, self.h = k, h
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
        hist_ext = (np.asarray(z[t - self.k - self.h + 1: t + 1], np.float32) - self.zmean) / self.zstd
        targ = (np.asarray(z[t + 1: t + 1 + self.h], np.float32) - self.zmean) / self.zstd
        return {"hist_ext": torch.from_numpy(hist_ext), "targ": torch.from_numpy(targ),
                "ctrl": torch.from_numpy(self.C[i]), "phase": torch.from_numpy(phase_feats(t)),
                "t_idx": torch.tensor(t, dtype=torch.long)}


def chain_history(det_fwd, hist_ext, ctrl, t_idx, k, h):
    """k-frame history ending at t whose last h frames are the frozen operator's forecast
    from the k true frames ending at t - h. det_fwd(hist, ctrl, phase) -> (B, h, C, g)."""
    ph = torch.from_numpy(phase_feats(t_idx.cpu().numpy() - h)).to(hist_ext.device)
    return torch.cat([hist_ext[:, h:k], det_fwd(hist_ext[:, :k], ctrl, ph)], dim=1)
