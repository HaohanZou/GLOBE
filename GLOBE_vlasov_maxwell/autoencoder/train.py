"""Train the autoencoder on the field stack.

  L = rel_L2 + w_spec * spectral + w_h1 * H1        (decoded 9 channels vs the store)
      + w_noise * [the same composite on decode(z + sigma * std(z) * eps)],
        sigma ~ U(0, latent_noise_max)

Relative terms use per-channel energy floors and channel weights (constants.py)
and a raised-cosine spatial weight that decays toward the domain faces.
AdamW (beta2 0.95), linear warmup then cosine, bf16 autocast (the projections
run in fp32), gradient clipping with non-finite guards, no EMA. Defaults are the
settings of the released models.

  single GPU:  python -m autoencoder.train --store <store> --out runs/ae
  multi GPU:   torchrun --nproc_per_node=N -m autoencoder.train ... --batch <per rank>
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from .constants import CHANNEL_WEIGHTS, FLOOR_RMS, KEEP, SHELL_TAPER_CELLS, SHELL_TAPER_MIN
from .dataset import STORE, PwfaFrameDataset, PwfaStore, make_loader, strided_subset, wake_view
from .losses import ae_loss
from .maxwell import shell_taper
from .model import build_model


def autocast(device):
    return (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda"
            else contextlib.nullcontext())


def save_ckpt(path, obj):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def loss_terms(st, device):
    """(floor, weights, spatial weight) tensors of the objective for a store's grid."""
    return (torch.tensor(FLOOR_RMS, device=device), torch.tensor(CHANNEL_WEIGHTS, device=device),
            shell_taper((st.Z, st.Y, st.X), SHELL_TAPER_CELLS, SHELL_TAPER_MIN, device))


@torch.no_grad()
def validate(model, loader, device, args, floor, weights, spatial, n_batches=8, noise_seed=0, amp=True):
    """Objective terms on val batches; the noisy-decode term is seeded (sigma and eps).
    amp=False runs the model in fp32 (the standalone evaluation)."""
    model.eval()
    rng, acc = np.random.default_rng(noise_seed), {}
    gen = torch.Generator(device=device).manual_seed(noise_seed)
    ctx = autocast(device) if amp else contextlib.nullcontext()
    for i, xb in enumerate(loader):
        x = wake_view(xb.to(device).float())
        tgt = x[:, list(KEEP)]
        with ctx:
            z = model.encode(x)
            rec = model.decode(z)
        total, terms = ae_loss(rec.float(), tgt, floor, weights, spatial, args.w_spec, args.w_h1,
                               args.hi_power, args.band_boost)
        vals = {"objective": total, **terms}
        if args.w_noise > 0:
            sigma = float(rng.uniform(0.0, args.latent_noise_max))
            eps = torch.randn(z.shape, generator=gen, device=z.device, dtype=z.dtype)
            with ctx:
                rec_n = model.decode(z + sigma * z.std() * eps)
            noisy, _ = ae_loss(rec_n.float(), tgt, floor, weights, spatial, args.w_spec, args.w_h1,
                               args.hi_power, args.band_boost)
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
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank, world = 0, 1
    if ddp:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
        rank, world = dist.get_rank(), dist.get_world_size()
    log = print if rank == 0 else (lambda *a, **k: None)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + rank)
    st = PwfaStore(args.store)
    model = build_model(args, (st.Z, st.Y, st.X), st.h).to(device)
    floor, weights, spatial = loss_terms(st, device)
    log(f"autoencoder: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params | "
        f"latent ({model.latent_ch}, {', '.join(str(v) for v in model.latent_grid())}) | "
        f"h = ({st.h[0]:.5f}, {st.h[1]:.5f}, {st.h[2]:.5f}) | {len(st.runs)} runs", flush=True)
    tr_ds = PwfaFrameDataset(st, "train")
    sampler = (torch.utils.data.distributed.DistributedSampler(
        tr_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True) if ddp else None)
    tr = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch, shuffle=sampler is None, sampler=sampler,
                                     num_workers=args.num_workers, pin_memory=device == "cuda",
                                     drop_last=True, persistent_workers=args.num_workers > 0)
    if len(tr) == 0:
        raise SystemExit("train split has fewer frames than --batch")
    va = make_loader(strided_subset(PwfaFrameDataset(st, "val"), 8 * args.batch), args.batch,
                     shuffle=False, num_workers=min(4, args.num_workers))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5, betas=(0.9, args.beta2))
    if args.warmup > 0:
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=args.warmup)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.steps - args.warmup))
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[args.warmup])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = int(ck["step"])
        log(f"resumed {args.resume} at step {step}", flush=True)
    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
    ckpt_path = os.path.join(args.out, "checkpoints", "model.pt")

    def checkpoint():
        if rank == 0:
            save_ckpt(ckpt_path, {"model": model.state_dict(), "opt": opt.state_dict(),
                                  "sched": sched.state_dict(), "step": step, "args": vars(args),
                                  "grid_shape": [st.Z, st.Y, st.X], "h": list(st.h)})

    def skip(reason):
        nonlocal step, nonfinite
        nonfinite += 1
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        log(f"step {step} skipped ({reason}); total {nonfinite}", flush=True)
        if nonfinite > 0.2 * step and step > 500:
            raise SystemExit(f"aborting: {nonfinite}/{step} steps non-finite")

    def composite(fields, tgt):
        return ae_loss(fields.float(), tgt, floor, weights, spatial, args.w_spec, args.w_h1,
                       args.hi_power, args.band_boost)

    nonfinite, epoch, t0 = 0, 0, time.time()
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for xb in tr:
            x = wake_view(xb.to(device).float())
            tgt = x[:, list(KEEP)]
            with autocast(device):
                z = model.encode(x)
                fields = model.decode(z)
            loss, terms = composite(fields, tgt)
            if args.w_noise > 0:
                sigma = float(rng.uniform(0.0, args.latent_noise_max))
                with autocast(device):
                    fields_n = model.decode(z + sigma * z.detach().std() * torch.randn_like(z))
                loss_n, _ = composite(fields_n, tgt)
                loss = loss + args.w_noise * loss_n
            opt.zero_grad(set_to_none=True)
            ok = bool(torch.isfinite(loss))
            if ddp:
                flag = torch.tensor(1.0 if ok else 0.0, device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                ok = bool(flag.item())
            if not ok:
                skip("non-finite loss")
            else:
                loss.backward()
                if ddp:
                    for p in model.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
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
                        log(f"step {step} loss {loss.item():.4f} rel {terms['rel_l2']:.4f} "
                            f"spec {terms['spectral']:.4f} h1 {terms['h1']:.4f} "
                            f"lr {sched.get_last_lr()[0]:.2e} {dt:.2f}s/it", flush=True)
            if rank == 0 and (step % args.val_every == 0 or step >= args.steps):
                v = validate(model, va, device, args, floor, weights, spatial)
                print("  [val] " + " ".join(f"{k} {val:.4f}" for k, val in v.items()), flush=True)
            if step % args.ckpt_every == 0 or step >= args.steps:
                checkpoint()
            if step >= args.steps:
                break
    checkpoint()
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    log("done.", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--out", default="runs/ae")
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--ch-mult", type=int, nargs="+", default=[1, 2, 4, 4])
    ap.add_argument("--latent-ch", type=int, default=320)
    ap.add_argument("--batch", type=int, default=3, help="per rank under torchrun")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--w-spec", type=float, default=0.1)
    ap.add_argument("--w-h1", type=float, default=0.1)
    ap.add_argument("--hi-power", type=float, default=1.0)
    ap.add_argument("--band-boost", type=float, default=5.0)
    ap.add_argument("--w-noise", type=float, default=0.5)
    ap.add_argument("--latent-noise-max", type=float, default=0.15)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    train(ap.parse_args(argv))


if __name__ == "__main__":
    main()
