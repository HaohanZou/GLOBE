"""Train the latent operator.

Objective: relative L2, in standardized latent space, between the emitted
h-block and the true future latents, with two additional terms:
  history noise   the true history is corrupted with sigma ~ U(0, hist_noise)
                  Gaussian noise (targets stay true);
  pushforward     a second block is predicted from the model's own detached
                  first-block output and supervised (--unroll 2); the term
                  ramps in after the learning-rate warmup.
AdamW, linear warmup then cosine, bf16 autocast, gradient clipping with
non-finite guards. Defaults are the settings of the released models.

  single GPU:  python -m operator_learning.train --latents latents/ae --out runs/operator
  multi GPU:   torchrun --nproc_per_node=N -m operator_learning.train ... --batch <per rank>
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

from .data import LatentWindows, control_stats, latent_stats, load_meta
from .model import LatentNeuralOperator


def rel(a, b):
    """Per-sample relative L2 over everything but the batch axis."""
    return (a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1).clamp_min(1e-8)


def autocast(device):
    return (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda"
            else contextlib.nullcontext())


def ddp_init():
    """(ddp, rank, world) under torchrun; (False, 0, 1) otherwise."""
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False, 0, 1
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    return True, dist.get_rank(), dist.get_world_size()


def all_ok(ok, ddp, device):
    """Every rank must take the same skip decision or the grad all-reduce deadlocks."""
    if not ddp:
        return ok
    flag = torch.tensor(1.0 if ok else 0.0, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def allreduce_grads(params):
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def make_sched(opt, warmup, steps):
    if warmup <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-2, total_iters=warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps - warmup))
    return torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[warmup])


def save_ckpt(path, obj):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def train(args):
    dev = args.device
    ddp, rank, world = ddp_init()
    log = print if rank == 0 else (lambda *a, **k: None)
    torch.manual_seed(args.seed)
    if args.unroll not in (1, 2):
        raise SystemExit("--unroll must be 1 (off) or 2 (one pushforward block)")
    meta = load_meta(args.latents)
    ck = torch.load(args.resume, map_location=dev, weights_only=False) if args.resume else None
    if ck is not None:      # statistics are frozen at run start; the latent space must match
        if ck.get("ae_sha256_16") != meta.get("ae_sha256_16"):
            raise SystemExit("the checkpoint was trained on a cache of a different autoencoder")
        zm, zs, cm, cs = (np.asarray(ck[key], np.float32) for key in ("zmean", "zstd", "cmean", "cstd"))
    else:
        zm = zs = cm = cs = None
        if rank == 0:
            zm, zs = latent_stats(args.latents)
            cm, cs = control_stats(args.latents)
        if ddp:
            obj = [zm, zs, cm, cs]
            dist.broadcast_object_list(obj, src=0)
            zm, zs, cm, cs = obj
    tr = LatentWindows(args.latents, "train", args.k, args.h, zm, zs, cm, cs, n_blocks=args.unroll)
    va = LatentWindows(args.latents, "val", args.k, args.h, zm, zs, cm, cs)
    log(f"train {len(tr.runs)} runs / {len(tr)} windows | val {len(va.runs)} runs / {len(va)} windows",
        flush=True)

    C, G = tr.Z[0].shape[1], tuple(tr.Z[0].shape[2:])
    net = LatentNeuralOperator(C, args.k, args.h, dim=args.width, depth=args.blocks, heads=args.heads,
                               n_ctrl=len(cm), mlp_ratio=args.mlp_ratio, persist=args.persist).to(dev)
    log(f"operator: grid {G} -> {int(np.prod(G))} tokens | dim {args.width} x depth {args.blocks} | "
        f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params | persist {args.persist} | "
        f"hist_noise {args.hist_noise} | unroll {args.unroll}", flush=True)
    sampler = (torch.utils.data.distributed.DistributedSampler(
        tr, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True) if ddp else None)
    dl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=sampler is None, sampler=sampler,
                                     num_workers=args.num_workers, pin_memory=dev == "cuda",
                                     drop_last=True, persistent_workers=args.num_workers > 0)
    if len(dl) == 0:
        raise SystemExit(f"train split has {len(tr)} windows < batch {args.batch}")
    vidx = np.linspace(0, len(va) - 1, min(len(va), 256), dtype=int).tolist()
    vdl = torch.utils.data.DataLoader(torch.utils.data.Subset(va, vidx), batch_size=args.batch,
                                      num_workers=min(2, args.num_workers))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sch = make_sched(opt, args.warmup, args.steps)
    step = 0
    if ck is not None:
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sched"])
        step = int(ck["step"])
        log(f"resumed {args.resume} at step {step}", flush=True)
    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
    ckpt_path = os.path.join(args.out, "checkpoints", "model.pt")

    def checkpoint():
        if rank == 0:
            save_ckpt(ckpt_path, {"model": net.state_dict(), "opt": opt.state_dict(),
                                  "sched": sch.state_dict(), "step": step, "args": vars(args),
                                  "zmean": zm, "zstd": zs, "cmean": cm, "cstd": cs,
                                  "ae_sha256_16": meta.get("ae_sha256_16")})

    @torch.no_grad()
    def validate():
        net.eval()
        errs = []
        for b in vdl:
            with autocast(dev):
                pred = net(b["hist"].to(dev), b["ctrl"].to(dev)).float()
            errs.append(rel(pred, b["targ"].to(dev)).mean().item())
        net.train()
        print(f"  [val] block rel-L2 {np.mean(errs):.4f}", flush=True)

    if ddp:      # decorrelate the per-rank noise streams after the shared-seed init
        torch.manual_seed(args.seed + rank)
    nonfinite, loss_win, t0, epoch = 0, [], time.time(), 0

    def skip(reason):
        nonlocal step, nonfinite
        nonfinite += 1
        opt.zero_grad(set_to_none=True)
        sch.step()
        step += 1
        log(f"step {step} skipped ({reason}); total {nonfinite}", flush=True)
        if nonfinite > 0.2 * step and step > 500:
            raise SystemExit(f"aborting: {nonfinite}/{step} steps non-finite")

    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for b in dl:
            hist, targ = b["hist"].to(dev, non_blocking=True), b["targ"].to(dev, non_blocking=True)
            ctrl = b["ctrl"].to(dev)
            hist_in = hist
            if args.hist_noise > 0:
                sig = args.hist_noise * torch.rand(hist.shape[0], 1, 1, 1, 1, 1, device=dev)
                hist_in = hist + sig * torch.randn_like(hist)
            pf_w = min(1.0, max(0.0, (step - args.warmup) / max(1, args.warmup))) if args.unroll > 1 else 0.0
            opt.zero_grad(set_to_none=True)
            with autocast(dev):
                pred = net(hist_in, ctrl)
            loss = rel(pred.float(), targ[:, :args.h]).mean()
            ok = bool(torch.isfinite(loss))
            if ok:
                loss.backward()          # block-1 graph is freed before block 2 is built
                log_loss = float(loss)
                if pf_w > 0:
                    hist2 = torch.cat([hist, pred.detach().float()], dim=1)[:, -args.k:]
                    with autocast(dev):
                        pred2 = net(hist2, ctrl)
                    loss2 = pf_w * rel(pred2.float(), targ[:, args.h:2 * args.h]).mean()
                    ok = bool(torch.isfinite(loss2))
                    if ok:
                        loss2.backward()
                        log_loss += float(loss2)
            if not all_ok(ok, ddp, dev):
                skip("non-finite loss")
                if step >= args.steps:
                    break
                continue
            if ddp:
                allreduce_grads(net.parameters())
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if not torch.isfinite(gnorm):
                skip("non-finite grad norm")
                if step >= args.steps:
                    break
                continue
            opt.step()
            sch.step()
            step += 1
            loss_win.append(log_loss)
            if step % args.log_every == 0:
                avg = float(np.mean(loss_win))
                loss_win.clear()
                dt = (time.time() - t0) / args.log_every
                t0 = time.time()
                log(f"step {step} loss {avg:.4f} lr {sch.get_last_lr()[0]:.2e} pf_w {pf_w:.2f} {dt:.2f}s/it",
                    flush=True)
            if rank == 0 and (step % args.val_every == 0 or step >= args.steps):
                validate()
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
    ap.add_argument("--latents", required=True)
    ap.add_argument("--out", default="runs/operator")
    ap.add_argument("--k", type=int, default=4, help="history frames")
    ap.add_argument("--h", type=int, default=4, help="emitted horizon frames")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--mlp-ratio", type=int, default=4)
    ap.add_argument("--persist", action="store_true", default=False,
                    help="predict a correction to the last history frame instead of the state")
    ap.add_argument("--hist-noise", type=float, default=0.3)
    ap.add_argument("--unroll", type=int, default=2, help="1 = off, 2 = one pushforward block")
    ap.add_argument("--batch", type=int, default=8, help="per rank under torchrun")
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    train(ap.parse_args(argv))


if __name__ == "__main__":
    main()
