"""Train the autoencoder on charge-density frames.

  L = rel_L2 + w_spec * spectral + w_h1 * H1
      + w_noise * [the same composite on decode(z + sigma * std(z) * eps)],
        sigma ~ U(0, latent_noise_max)

The second line decodes a perturbed latent against the clean target.
AdamW, cosine schedule, bf16 autocast, gradient clipping with non-finite
guards, no EMA. Defaults are the settings of the released models.

  python -m autoencoder.train --store <store> --out runs/ae
"""
from __future__ import annotations

import argparse
import contextlib
import os
import time

import numpy as np
import torch

from .dataset import (STORE, BunchedBeamFrameDataset, BunchedBeamStore, make_loader,
                      strided_subset)
from .losses import ae_loss
from .model import build_model


def autocast(device):
    return (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda"
            else contextlib.nullcontext())


def save_ckpt(path, obj):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


@torch.no_grad()
def validate(model, loader, device, args, n_batches=8, noise_seed=0, amp=True):
    """Objective terms on val batches; the noisy-decode term is seeded (sigma and eps).
    amp=False runs the model in fp32 (the standalone evaluation)."""
    model.eval()
    rng, acc = np.random.default_rng(noise_seed), {}
    gen = torch.Generator(device=device).manual_seed(noise_seed)
    ctx = autocast(device) if amp else contextlib.nullcontext()
    for i, x in enumerate(loader):
        x = x.to(device).float()
        with ctx:
            z = model.encode(x)
            rec = model.decode(z)
        total, terms = ae_loss(rec.float(), x, args.w_spec, args.w_h1, args.hi_power,
                               args.band_boost)
        vals = {"objective": total, **terms}
        if args.w_noise > 0:
            sigma = float(rng.uniform(0.0, args.latent_noise_max))
            eps = torch.randn(z.shape, generator=gen, device=z.device, dtype=z.dtype)
            with ctx:
                rec_n = model.decode(z + sigma * z.std() * eps)
            noisy, _ = ae_loss(rec_n.float(), x, args.w_spec, args.w_h1, args.hi_power,
                               args.band_boost)
            vals["noisy_decode"] = noisy
            vals["objective"] = total + args.w_noise * noisy
        for k, v in vals.items():
            acc.setdefault(k, []).append(float(v))
        if i + 1 >= n_batches:
            break
    model.train()
    return {k: float(np.mean(v)) for k, v in acc.items()}


def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    st = BunchedBeamStore(args.store)
    model = build_model(args).to(device)
    fz, fy, fx = model.axis_factors()
    print(f"autoencoder: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params | "
          f"latent ({args.latent_ch}, {st.Z // fz}, {st.Y // fy}, {st.X // fx}) | "
          f"{len(st.runs)} runs", flush=True)
    tr = make_loader(BunchedBeamFrameDataset(st, "train"), args.batch, shuffle=True,
                     num_workers=args.num_workers)
    if len(tr) == 0:
        raise SystemExit("train split has fewer frames than --batch")
    va = make_loader(strided_subset(BunchedBeamFrameDataset(st, "val"), 8 * args.batch),
                     args.batch, shuffle=False, num_workers=min(4, args.num_workers))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = int(ck["step"])
        print(f"resumed {args.resume} at step {step}", flush=True)
    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
    ckpt_path = os.path.join(args.out, "checkpoints", "model.pt")

    def checkpoint():
        save_ckpt(ckpt_path, {"model": model.state_dict(), "opt": opt.state_dict(),
                              "sched": sched.state_dict(), "step": step, "args": vars(args)})

    def skip(reason):
        nonlocal step, nonfinite
        nonfinite += 1
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        print(f"step {step} skipped ({reason}); total {nonfinite}", flush=True)
        if nonfinite > 0.2 * step and step > 500:
            raise SystemExit(f"aborting: {nonfinite}/{step} steps non-finite")

    nonfinite, t0 = 0, time.time()
    while step < args.steps:
        for x in tr:
            x = x.to(device).float()
            with autocast(device):
                z = model.encode(x)
                rec = model.decode(z)
            loss, terms = ae_loss(rec.float(), x, args.w_spec, args.w_h1, args.hi_power,
                                  args.band_boost)
            if args.w_noise > 0:
                sigma = float(rng.uniform(0.0, args.latent_noise_max))
                with autocast(device):
                    rec_n = model.decode(z + sigma * z.detach().std() * torch.randn_like(z))
                loss_n, _ = ae_loss(rec_n.float(), x, args.w_spec, args.w_h1, args.hi_power,
                                    args.band_boost)
                loss = loss + args.w_noise * loss_n
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skip("non-finite loss")
            else:
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gnorm):
                    skip("non-finite grad norm")
                else:
                    opt.step()
                    sched.step()
                    step += 1
                    if step % args.log_every == 0:
                        dt = (time.time() - t0) / args.log_every
                        t0 = time.time()
                        print(f"step {step} loss {loss.item():.4f} rel {terms['rel_l2']:.4f} "
                              f"spec {terms['spectral']:.4f} h1 {terms['h1']:.4f} "
                              f"lr {sched.get_last_lr()[0]:.2e} {dt:.2f}s/it", flush=True)
            if step % args.val_every == 0 or step >= args.steps:
                v = validate(model, va, device, args)
                print("  [val] " + " ".join(f"{k} {x:.4f}" for k, x in v.items()), flush=True)
            if step % args.ckpt_every == 0 or step >= args.steps:
                checkpoint()
            if step >= args.steps:
                break
    checkpoint()
    print("done.", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--out", default="runs/ae")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--ch-mult", type=int, nargs="+", default=[1, 2, 4, 4])
    ap.add_argument("--latent-ch", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--w-spec", type=float, default=0.1)
    ap.add_argument("--w-h1", type=float, default=0.1)
    ap.add_argument("--hi-power", type=float, default=1.0)
    ap.add_argument("--band-boost", type=float, default=5.0)
    ap.add_argument("--w-noise", type=float, default=0.5)
    ap.add_argument("--latent-noise-max", type=float, default=0.15)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    train(ap.parse_args(argv))


if __name__ == "__main__":
    main()
