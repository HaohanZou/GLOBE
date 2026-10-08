"""Reader for the packed plasma-wakefield field store.

Layout (one directory per run):
  <store>/runs/run_XXXXXX/
      fields.zarr         (T, C, Z, Y, X) float16, channels constants.CHANNELS
      pack_meta.json      arrays.fields.shape, channels, channel_scales_si,
                          grid.grid_spacing_si_zyx, burn_in_frames, qa_gates
      manifest_row.json   split ("train" / "val" / "test"), derived (machine controls)

Plasma units: every discrete Maxwell identity is unit-coefficient (div E = rho,
div B = 0) with the grid spacing h in units of the plasma skin depth
L = eps0 E_scale / rho_scale, derived from each run's recorded scales.
Training consumes frames >= burn_in_frames (frame 0 is the field-solver
initialization). The model input view (wake_view, formed in single precision)
replaces the plasma-electron channel by the wake perturbation rho_e + rho_ion.
Point the code at a store with --store or PWFA_STORE.
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .constants import CHANNELS, EPS0

STORE = os.environ.get("PWFA_STORE", "")
# deployment-known machine controls (keys of manifest_row["derived"]), in this order
CTRL_KEYS = ("q_driver_C", "gamma_d", "sigma_z_m", "sigma_r_m", "x_off_m",
             "q_witness_C", "delay_m", "witness_on")


class _RunArray:
    """frame index -> (10, Z, Y, X) float16 (the leading channels of the stored stack)."""

    def __init__(self, path, n_ch):
        import zarr
        self._z = zarr.open_array(str(path), mode="r")
        self._n = n_ch

    def __getitem__(self, t):
        return np.asarray(self._z[int(t), :self._n])


class PwfaStore:
    def __init__(self, root=STORE):
        if not root:
            raise SystemExit("no store given: pass --store or set PWFA_STORE")
        root = pathlib.Path(root)
        self.root = root / "runs" if (root / "runs").is_dir() else root
        self.runs = sorted(int(p.name[4:]) for p in self.root.glob("run_*")
                           if p.name[4:].isdigit() and (p / "pack_meta.json").exists()
                           and (p / "fields.zarr").exists())
        if not self.runs:
            raise FileNotFoundError(f"no packed runs under {self.root}")
        self._meta, self._rows, self._arr = {}, {}, {}
        m0 = self.pack_meta(self.runs[0])
        if tuple(m0["channels"][:len(CHANNELS)]) != CHANNELS:
            raise RuntimeError(f"channel order {m0['channels']} does not lead with {CHANNELS}")
        shape = m0["arrays"]["fields"]["shape"]
        self.T, self.Z, self.Y, self.X = int(shape[0]), int(shape[2]), int(shape[3]), int(shape[4])
        self.burn_in = int(m0["burn_in_frames"])
        sc = m0["channel_scales_si"]
        self.L_scale = EPS0 * float(sc["Ex"]) / float(sc["rho_driver"])
        self.h = tuple(float(v) / self.L_scale for v in m0["grid"]["grid_spacing_si_zyx"])   # (dz, dy, dx)

    def rpath(self, r):
        return self.root / f"run_{r:06d}"

    def pack_meta(self, r):
        if r not in self._meta:
            self._meta[r] = json.loads((self.rpath(r) / "pack_meta.json").read_text())
        return self._meta[r]

    def row(self, r):
        if r not in self._rows:
            self._rows[r] = json.loads((self.rpath(r) / "manifest_row.json").read_text())
        return self._rows[r]

    def fields(self, r):
        if r not in self._arr:
            self._arr[r] = _RunArray(self.rpath(r) / "fields.zarr", len(CHANNELS))
        return self._arr[r]

    def frames(self):
        """The trainable frame indices (burn-in excluded)."""
        return range(self.burn_in, self.T)

    def controls(self, r):
        d = self.row(r)["derived"]
        return np.array([float(d[key]) for key in CTRL_KEYS], dtype=np.float32)

    def split_runs(self, split):
        """Runs of a split whose numerical QA gates did not fail."""
        return [r for r in self.runs if self.row(r)["split"] == split
                and (self.pack_meta(r).get("qa_gates") or {}).get("numerics_pass", True)]


def wake_view(x):
    """Model input view of a store frame (..., 10, Z, Y, X): the plasma-electron channel
    becomes the wake perturbation delta_rho_e = rho_e + rho_ion."""
    x = x.clone()
    x[..., 2, :, :, :] += x[..., 3, :, :, :]
    return x


class PwfaFrameDataset(Dataset):
    """Per-frame samples: (10, Z, Y, X) float16 store frames, burn-in excluded."""

    def __init__(self, store, split):
        runs = store.split_runs(split)
        if not runs:
            raise RuntimeError(f"no runs in split {split!r}")
        self.st = store
        self.items = [(r, t) for r in runs for t in store.frames()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r, t = self.items[i]
        return torch.from_numpy(np.ascontiguousarray(self.st.fields(r)[t]))


def make_loader(ds, batch_size, shuffle, num_workers=8):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle,
                      persistent_workers=num_workers > 0)


def strided_subset(ds, n):
    """Deterministic subset spanning every run and the whole time range."""
    return Subset(ds, np.linspace(0, len(ds) - 1, num=min(len(ds), n), dtype=int).tolist())
