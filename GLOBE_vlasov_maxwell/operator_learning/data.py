"""Latent-window dataset for operator training.

The latent cache (operator_learning.latents) holds one run_XXXXXX.npy
(T, C, Lz, Ly, Lx) float16 per run, read memory-mapped, and a meta.json with
each run's split and machine controls. A sample is (k past latents,
controls) -> the next n_blocks*h latents; the extra block is the target of
the pushforward step. Latents and controls are standardized with train-split
statistics that are frozen in the operator checkpoint.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch


def load_meta(root):
    return json.loads((pathlib.Path(root) / "meta.json").read_text())


def run_array(root, r):
    return np.load(pathlib.Path(root) / f"run_{int(r):06d}.npy", mmap_mode="r")


class LatentWindows(torch.utils.data.Dataset):
    def __init__(self, root, split, k, h, zmean, zstd, cmean, cstd, n_blocks=1):
        self.k, self.h, self.span = k, h, h * n_blocks
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
        self.items = [(i, t) for i in range(len(self.Z)) for t in range(k - 1, self.T - self.span)]
        if not self.items:
            raise RuntimeError(f"T={self.T} is too short for k={k} and {self.span} target frames")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, t = self.items[idx]
        z = self.Z[i]
        hist = (np.asarray(z[t - self.k + 1: t + 1], np.float32) - self.zmean) / self.zstd
        targ = (np.asarray(z[t + 1: t + 1 + self.span], np.float32) - self.zmean) / self.zstd
        return {"hist": torch.from_numpy(hist), "targ": torch.from_numpy(targ),
                "ctrl": torch.from_numpy(self.C[i])}


def control_stats(root, split="train"):
    """Per-control mean/std over the training runs (raw controls span many decades)."""
    C = np.stack([np.asarray(i["controls"], np.float32)
                  for i in load_meta(root)["runs"].values() if i["split"] == split])
    s = C.std(0)
    return C.mean(0), np.where(s > 0, s, 1.0).astype(np.float32)


def latent_stats(root, split="train", max_runs=20, stride=10):
    """Per-channel latent mean/std (C, 1, 1, 1) over a strided sample of training runs."""
    acc = []
    for r, info in load_meta(root)["runs"].items():
        if info["split"] == split and len(acc) < max_runs:
            acc.append(np.asarray(run_array(root, r)[::stride]))
    Z = np.concatenate(acc, 0).astype(np.float32)
    return (Z.mean(axis=(0, 2, 3, 4), keepdims=True)[0],
            Z.std(axis=(0, 2, 3, 4), keepdims=True)[0].clip(1e-6))
