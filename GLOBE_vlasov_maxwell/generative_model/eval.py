"""Evaluate the residual flow on a split with its training objective: the
flow-matching loss on depth-1 chain histories, averaged over tau in
{0.1, 0.3, 0.5, 0.7, 0.9} (EMA weights, standardized latent space).

  python -m generative_model.eval --ckpt runs/flow/checkpoints/model.pt --split val
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from operator_learning.model import load_operator
from operator_learning.train import autocast
from .data import ResidualWindows, chain_roll
from .model import LatentResidualFM
from .sample import residual_sigma

TAUS = (0.1, 0.3, 0.5, 0.7, 0.9)


def load_flow(path, device, det=""):
    """(flow net with EMA weights, frozen operator, args, stats, sigma_r, step);
    ``det`` overrides the operator checkpoint path recorded in the flow checkpoint."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    op, da, stats, _ = load_operator(det or a.det, device)
    ch = stats["zmean"].shape[0]
    net = LatentResidualFM(ch, da.k, da.h, dim=a.width, depth=a.blocks, heads=a.heads,
                           n_ctrl=len(stats["cmean"]), mlp_ratio=a.mlp_ratio, amp=a.amp,
                           tok_floor=a.tok_floor).to(device).eval()
    net.load_state_dict(ck["ema"])
    return net, op, a, stats, ck["sigma_r"].to(device), int(ck["step"])


def det_forward(det, device):
    @torch.no_grad()
    def f(hist, ctrl):
        with autocast(device):
            return det(hist, ctrl).float()
    return f


@torch.no_grad()
def cfm_loss(net, det, sigma_r, loader, device, seed=1234):
    """Flow-matching loss on depth-1 chain histories, averaged over the tau grid and the batches."""
    det_fwd = det_forward(det, device)
    gen = torch.Generator(device=device).manual_seed(seed)
    acc = []
    for b in loader:
        hist_ext, targ, ctrl = (b[key].to(device) for key in ("hist_ext", "targ", "ctrl"))
        hist = chain_roll(det_fwd, hist_ext, ctrl, net.k, net.h, torch.ones_like(b["feasible"]))
        fdet = det_fwd(hist, ctrl)
        r = (targ - fdet) / residual_sigma(net, sigma_r, fdet, hist)
        for tv in TAUS:
            eps = torch.randn(r.shape, device=device, generator=gen)
            tau = torch.full((r.shape[0],), tv, device=device)
            with autocast(device):
                v = net((1 - tv) * eps + tv * r, fdet, hist, ctrl, tau).float()
            acc.append(float((v - (r - eps)).pow(2).mean()))
    return float(np.mean(acc))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--latents", default="", help="default: the cache recorded in the checkpoint")
    ap.add_argument("--det", default="", help="default: the operator recorded in the checkpoint")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--json", default="")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args(argv)
    dev = args.device
    net, det, a, stats, sigma_r, step = load_flow(args.ckpt, dev, args.det)
    lat = args.latents or a.latents
    ds = ResidualWindows(lat, args.split, net.k, net.h, *(stats[key] for key in ("zmean", "zstd", "cmean", "cstd")))
    idx = np.linspace(0, len(ds) - 1, min(len(ds), args.batches * args.batch), dtype=int).tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=args.batch)
    cfm = cfm_loss(net, det, sigma_r, loader, dev)
    print(f"flow step {step}, {len(idx)} {args.split} windows: cfm {cfm:.4f}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"cfm": cfm}, f, indent=2)
    return cfm


if __name__ == "__main__":
    main()
