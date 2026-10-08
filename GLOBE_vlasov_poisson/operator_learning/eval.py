"""Evaluate a trained operator on a split of the latent cache with its training
objective: the relative L2, in standardized latent space, of every h-block
predicted from the true k-frame history (mean over blocks, then over runs).

  python -m operator_learning.eval --ckpt runs/operator/checkpoints/model.pt --split val
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .data import load_meta, phase_feats, run_array
from .model import load_operator
from .train import autocast


@torch.no_grad()
def teacher_forced(net, Zn, ctrl):
    """Mean block relative L2 over one standardized trajectory (T, C, g)."""
    k, h, T = net.k, net.h, Zn.shape[0]
    if T - k < h:
        raise SystemExit(f"T={T} is too short for k={k}, h={h}")
    errs = []
    for t in range(k - 1, T - h, h):
        ph = torch.from_numpy(phase_feats(t))[None].to(Zn.device)
        with autocast(Zn.device.type):
            y = net(Zn[t - k + 1: t + 1][None], ctrl, ph).float()[0]
        targ = Zn[t + 1: t + 1 + h]
        errs.append(float((y - targ).norm() / targ.norm().clamp_min(1e-8)))
    return float(np.mean(errs))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--latents", default="", help="default: the cache recorded in the checkpoint")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--runs", default="", help="comma list of run ids (default: the whole split)")
    ap.add_argument("--json", default="")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args(argv)
    dev = args.device
    net, a, st, step = load_operator(args.ckpt, dev)
    lat = args.latents or a.latents
    meta = load_meta(lat)
    zm, zs = (torch.from_numpy(st[key])[None].to(dev) for key in ("zmean", "zstd"))
    runs = ([int(r) for r in args.runs.split(",") if r] or
            [int(r) for r, i in meta["runs"].items() if i["split"] == args.split])
    if not runs:
        raise SystemExit(f"no runs to evaluate in split {args.split!r}")
    res = {}
    for r in runs:
        Zn = (torch.from_numpy(np.asarray(run_array(lat, r), np.float32)).to(dev) - zm) / zs
        ctrl = torch.from_numpy(((np.asarray(meta["runs"][str(r)]["controls"], np.float32)
                                  - st["cmean"]) / st["cstd"])[None]).to(dev)
        res[r] = teacher_forced(net, Zn, ctrl)
        print(f"run {r:4d}: block rel-L2 {res[r]:.4f}", flush=True)
    mean = float(np.mean(list(res.values())))
    print(f"operator step {step}, {len(res)} runs: block rel-L2 {mean:.4f}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"runs": res, "mean": mean}, f, indent=2)
    return mean


if __name__ == "__main__":
    main()
