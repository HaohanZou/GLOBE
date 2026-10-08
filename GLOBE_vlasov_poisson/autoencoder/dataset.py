"""Reader for the packed bunched-beam charge-density store.

Layout (one directory per run):
  <store>/norm.json                     packed-unit normalization (L_scale; eps0 = 1)
  <store>/runs/run_XXXXXX/
      rho.zarr            (T, Z, Y, X) float16 charge density              ordinary runs
      fields.zarr         (T, C, Z, Y, X) float16 [phi, Ex, Ey, Ez, rho]   full-field runs
      pack_meta.json      array_shape, channels, qa_gates, openpmd_mesh_metadata
      manifest_row.json   split ("train" / "val" / "test"), physical (machine
                          controls), context (bore_radius_m, box_half_width_m)

Training consumes rho only; phi and E are derived from rho by poisson.py.
Splits are assigned per design, so val/test runs are unseen machine settings.
Point the code at a store with --store or the BUNCHED_BEAM_STORE variable.
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

STORE = os.environ.get("BUNCHED_BEAM_STORE", "")
# deployment-known machine controls (keys of manifest_row["physical"]), in this order
CTRL_KEYS = ("EQ0_v_per_m2", "E0T_v_per_m", "q_tot_c")


class _RunArray:
    """frame index -> (1, Z, Y, X) float16 rho, for both run layouts."""

    def __init__(self, path, rho_index=None):
        import zarr
        self._z = zarr.open_array(str(path), mode="r")
        self._rho = rho_index

    def __getitem__(self, t):
        if self._rho is None:
            return np.asarray(self._z[int(t)])[None]
        return np.asarray(self._z[int(t), self._rho])[None]


class BunchedBeamStore:
    def __init__(self, root=STORE):
        if not root:
            raise SystemExit("no store given: pass --store or set BUNCHED_BEAM_STORE")
        root = pathlib.Path(root)
        self.top = root if (root / "runs").is_dir() else root.parent
        self.root = self.top / "runs"
        self.runs = sorted(int(p.name[4:]) for p in self.root.glob("run_*")
                           if p.name[4:].isdigit() and (p / "pack_meta.json").exists())
        if not self.runs:
            raise FileNotFoundError(f"no packed runs under {self.root}")
        self._meta, self._rows, self._arr = {}, {}, {}
        shape = self.pack_meta(self.runs[0])["array_shape"]
        self.T = shape[0]
        self.Z, self.Y, self.X = shape[-3:]

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
        """Training view: (1, Z, Y, X) rho per frame, for every run."""
        if r not in self._arr:
            path, idx = self.rpath(r) / "rho.zarr", None
            if not path.exists():
                path = self.rpath(r) / "fields.zarr"
                idx = list(self.pack_meta(r)["channels"]).index("rho")
            self._arr[r] = _RunArray(path, idx)
        return self._arr[r]

    def eval_channels(self, r):
        """Full-field runs: (array (T, C, Z, Y, X), channel names); None otherwise."""
        path = self.rpath(r) / "fields.zarr"
        if not path.exists():
            return None
        import zarr
        return zarr.open_array(str(path), mode="r"), list(self.pack_meta(r)["channels"])

    def controls(self, r):
        p = self.row(r)["physical"]
        return np.array([p[key] for key in CTRL_KEYS], dtype=np.float32)

    def split_runs(self, split):
        """Runs of a split whose numerical QA gates passed."""
        return [r for r in self.runs if self.row(r)["split"] == split
                and (self.pack_meta(r).get("qa_gates") or {}).get("numerics_pass", True)]

    def geometry(self):
        """Grid and bore in packed units (x~ = x / L_scale), for the Poisson solver."""
        L = json.loads((self.top / "norm.json").read_text())["L_scale"]
        pm = self.pack_meta(self.runs[0])["openpmd_mesh_metadata"]
        off = np.asarray(pm["gridGlobalOffset"], np.float64) / L      # (z, y, x) lower corner
        sp = np.asarray(pm["gridSpacing"], np.float64) / L
        pos = np.asarray(pm["position"], np.float64)                   # in-cell position
        ctx = self.row(self.runs[0])["context"]
        return dict(dz=float(sp[0]), dy=float(sp[1]), dx=float(sp[2]),
                    yc=off[1] + (np.arange(self.Y) + pos[1]) * sp[1],
                    xc=off[2] + (np.arange(self.X) + pos[2]) * sp[2],
                    bore_radius=float(ctx["bore_radius_m"]) / L, L_scale=L)


class BunchedBeamFrameDataset(Dataset):
    """Per-frame samples: (1, Z, Y, X) float16 rho."""

    def __init__(self, store, split):
        runs = store.split_runs(split)
        if not runs:
            raise RuntimeError(f"no runs in split {split!r}")
        self.st = store
        self.items = [(r, t) for r in runs for t in range(store.T)]

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
