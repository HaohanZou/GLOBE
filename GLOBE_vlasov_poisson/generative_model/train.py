"""Train the residual flow on top of a frozen operator.

  r = (z_block - f_det(hist)) / sigma_r,  x_tau = (1 - tau) eps + tau r,  v* = r - eps

Rectified-flow MSE with tau ~ logit-normal, plus an energy score on one-step
samples (--w-energy). The operator runs on the fly, frozen, and every history
is a chain history (data.chain_history).

Normalizer (--amp wintok, the default):
  (0) sigma_r: per-channel residual RMS over --stat-batches batches;
  (1) s_win: one scale per window, fit by the Gaussian scale likelihood
      MS / s^2 + 2 log s on r / sigma_r, then frozen;
  (2) s_tok: one scale per grid point, fit the same way on the remainder, then frozen;
  (3) the RMS of the whitened residual over --stat-batches batches is folded into sigma_r.
--amp none skips (1)-(3). The EMA weights (evaluated and stored) copy the frozen heads.

Stability: EMA, QK normalization, AdamW beta2 0.95 with weight decay on
matrices only, gradient clipping, non-finite guards and a loss-spike guard
(skip when the loss exceeds spike_mult x its running median). Defaults are the
settings of the released models.

  single GPU:  python -m generative_model.train --latents latents/ae \
                   --det runs/operator/checkpoints/model.pt --out runs/flow
  multi GPU:   torchrun --nproc_per_node=N -m generative_model.train ... \
                   --batch <per rank> --energy-batch <per rank>
"""
from __future__ import annotations

import argparse
import collections
import os
import statistics
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from operator_learning.model import load_operator
from operator_learning.train import (all_ok, allreduce_grads, autocast, ddp_init, make_sched,
                                     save_ckpt)
from .data import ResidualWindows, chain_history
from .eval import cfm_loss, det_forward
from .model import LatentResidualFM
from .sample import residual_sigma


def train(args):
    dev = args.device
    ddp, rank, world = ddp_init()
    log = print if rank == 0 else (lambda *a, **k: None)
    torch.manual_seed(args.seed)
    det, da, stats, det_step = load_operator(args.det, dev)
    det.requires_grad_(False)
    k, h = da.k, da.h
    zm, zs, cm, cs = (stats[key] for key in ("zmean", "zstd", "cmean", "cstd"))
    tr = ResidualWindows(args.latents, "train", k, h, zm, zs, cm, cs)
    va = ResidualWindows(args.latents, "val", k, h, zm, zs, cm, cs)
    ch = tr.Z[0].shape[1]
    log(f"frozen operator {args.det} (step {det_step}), k={k} h={h} | train {len(tr.runs)} runs / "
        f"{len(tr)} windows | val {len(va.runs)} runs / {len(va)} windows", flush=True)
    net = LatentResidualFM(ch, k, h, dim=args.width, depth=args.blocks, heads=args.heads, n_ctrl=len(cm),
                           mlp_ratio=args.mlp_ratio, amp=args.amp, tok_floor=args.tok_floor).to(dev)
    log(f"residual flow: dim {args.width} x depth {args.blocks} | "
        f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params | amp {args.amp} | "
        f"w_energy {args.w_energy}", flush=True)
    opt = torch.optim.AdamW(
        [{"params": [p for p in net.parameters() if p.ndim >= 2], "weight_decay": args.wd},
         {"params": [p for p in net.parameters() if p.ndim < 2], "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, args.beta2))
    sch = make_sched(opt, args.warmup, args.steps)

    def loader(seed, workers, shard=True):
        s = (DistributedSampler(tr, num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=True)
             if ddp and shard else None)
        return DataLoader(tr, batch_size=args.batch, shuffle=s is None, sampler=s,
                          generator=torch.Generator().manual_seed(seed), num_workers=workers,
                          pin_memory=dev == "cuda", drop_last=True, persistent_workers=workers > 0)

    dl = loader(args.seed, args.num_workers)
    if len(dl) == 0:
        raise SystemExit(f"train split has {len(tr)} windows < batch {args.batch}")
    vidx = np.linspace(0, len(va) - 1, min(len(va), 8 * args.batch), dtype=int).tolist()
    vdl = DataLoader(Subset(va, vidx), batch_size=args.batch, num_workers=min(2, args.num_workers))
    det_fwd = det_forward(det, dev)

    def batch_inputs(b):
        hist_ext, targ, ctrl, ph = (b[key].to(dev, non_blocking=True)
                                    for key in ("hist_ext", "targ", "ctrl", "phase"))
        hist = chain_history(det_fwd, hist_ext, ctrl, b["t_idx"], k, h)
        return hist, det_fwd(hist, ctrl, ph), targ, ctrl, ph

    @torch.no_grad()
    def stat_pass(seed, fn):
        """rank-0 mean of fn(hist, fdet, targ) -> (sum, count) over --stat-batches batches."""
        acc, n = 0.0, 0
        for bi, b in enumerate(loader(seed, 2, shard=False)):
            if bi >= args.stat_batches:
                break
            v, cnt = fn(*batch_inputs(b)[:3])
            acc, n = acc + v, n + cnt
        return acc / max(n, 1)

    step, ema_sd = 0, None
    ck = torch.load(args.resume, map_location=dev, weights_only=False) if args.resume else None
    if ck is not None:
        for key in ("amp", "tok_floor"):
            if ck["args"].get(key) != getattr(args, key):
                raise SystemExit(f"--{key} differs from the checkpoint's ({ck['args'].get(key)})")
        net.load_state_dict(ck["model"])
        ema_sd = {n: v.to(dev) for n, v in ck["ema"].items()}
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sched"])
        sigma_r = ck["sigma_r"].to(dev)
        step = int(ck["step"])
        log(f"resumed {args.resume} at step {step}", flush=True)
    else:
        sigma_r = torch.zeros(ch, device=dev)
        if rank == 0:
            def sq(hist, fdet, targ):
                r = (targ - fdet).transpose(1, 2).reshape(targ.shape[0], ch, -1)
                return r.double().pow(2).sum(dim=(0, 2)), r.shape[0] * r.shape[2]
            sigma_r = stat_pass(args.seed + 2, sq).sqrt().float().clamp_min(1e-4)
            log(f"sigma_r over {args.stat_batches} batches: mean {sigma_r.mean():.4f} "
                f"min {sigma_r.min():.4f} max {sigma_r.max():.4f}", flush=True)
        if ddp:
            dist.broadcast(sigma_r, src=0)
    sig = sigma_r.view(1, 1, ch, 1, 1, 1)

    if net.amp is not None and ck is None and args.prefit_steps > 0:
        def prefit(mod, label, loss_fn, seed):
            popt = torch.optim.Adam(mod.parameters(), lr=1e-3)
            n, win, t0, dl_p = 0, [], time.time(), loader(seed, args.num_workers)
            ep = 0
            while n < args.prefit_steps:
                if isinstance(dl_p.sampler, DistributedSampler):
                    dl_p.sampler.set_epoch(ep)
                ep += 1
                for b in dl_p:
                    hist, fdet, targ, _, _ = batch_inputs(b)
                    ploss = loss_fn(fdet, hist, targ)
                    if all_ok(bool(torch.isfinite(ploss)), ddp, dev):
                        popt.zero_grad(set_to_none=True)
                        ploss.backward()
                        if ddp:
                            allreduce_grads(mod.parameters())
                        popt.step()
                        win.append(float(ploss))
                    n += 1
                    if n % 100 == 0 and win:
                        log(f"[prefit {label}] step {n} loss {np.mean(win):.4f} "
                            f"{(time.time() - t0) / 100:.2f}s/it", flush=True)
                        win.clear()
                        t0 = time.time()
                    if n >= args.prefit_steps:
                        break
            mod.requires_grad_(False)

        def loss_win(fdet, hist, targ):
            ms = ((targ - fdet) / sig).pow(2).flatten(1).mean(1)
            ls = net.amp.win(fdet, hist)
            return (ms * torch.exp(-2 * ls) + 2 * ls).mean()

        def loss_tok(fdet, hist, targ):
            with torch.no_grad():
                sw = net.amp.win(fdet, hist).exp().view(-1, 1, 1, 1, 1, 1)
            ms = ((targ - fdet) / (sig * sw)).pow(2).mean(dim=(1, 2))           # (B, Z, Y, X)
            ls = net.amp.tok(fdet, hist).reshape(ms.shape)
            return (ms * torch.exp(-2 * ls) + 2 * ls).mean()

        prefit(net.amp.win, "s_win", loss_win, args.seed + 11)
        prefit(net.amp.tok, "s_tok", loss_tok, args.seed + 12)
        if rank == 0:
            def z2(hist, fdet, targ):
                z = (targ - fdet) / (sig * net.residual_scale(fdet, hist))
                return z.double().pow(2).sum(), z.numel()
            c = float(stat_pass(args.seed + 3, z2).sqrt())
            sigma_r.mul_(c)
            log(f"whitened residual rms {c:.4f} folded into sigma_r", flush=True)
        if ddp:
            dist.broadcast(sigma_r, src=0)
        sig = sigma_r.view(1, 1, ch, 1, 1, 1)
    if net.amp is not None:
        net.amp.requires_grad_(False)
    if ema_sd is None:
        ema_sd = {n: v.detach().clone().float() for n, v in net.state_dict().items()}
    frozen = ({n for n, p in net.named_parameters() if not p.requires_grad}
              | {n for n, _ in net.named_buffers()})

    def ema_update():
        d = min(args.ema, (1 + step) / (10 + step))
        sd = net.state_dict()
        for n, v in ema_sd.items():
            if n in frozen:
                v.copy_(sd[n].detach().float())
            else:
                v.mul_(d).add_(sd[n].detach().float(), alpha=1 - d)

    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
    ckpt_path = os.path.join(args.out, "checkpoints", "model.pt")

    def checkpoint():
        if rank == 0:
            save_ckpt(ckpt_path, {"model": net.state_dict(), "ema": ema_sd, "opt": opt.state_dict(),
                                  "sched": sch.state_dict(), "step": step, "args": vars(args),
                                  "zmean": zm, "zstd": zs, "cmean": cm, "cstd": cs,
                                  "sigma_r": sigma_r.cpu(), "det_step": det_step})

    def validate():
        raw = {n: v.detach().clone() for n, v in net.state_dict().items()}
        net.load_state_dict(ema_sd)
        net.eval()
        print(f"  [val] cfm {cfm_loss(net, det, sigma_r, vdl, dev):.4f} (ema)", flush=True)
        net.load_state_dict(raw)
        net.train()

    def skip(reason):
        nonlocal step
        opt.zero_grad(set_to_none=True)
        sch.step()
        step += 1
        log(f"step {step} skipped ({reason})", flush=True)

    if ddp:      # decorrelate the per-rank noise streams after the shared-seed init
        torch.manual_seed(args.seed + rank)
    loss_hist = collections.deque(maxlen=256)
    nonfinite = spikes = epoch = 0
    t0 = time.time()
    while step < args.steps:
        if isinstance(dl.sampler, DistributedSampler):
            dl.sampler.set_epoch(epoch)
        epoch += 1
        for b in dl:
            hist, fdet, targ, ctrl, ph = batch_inputs(b)
            B = hist.shape[0]
            r = (targ - fdet) / residual_sigma(net, sigma_r, fdet, hist)
            opt.zero_grad(set_to_none=True)
            ok, en_val = True, None
            if args.w_energy > 0:      # backwarded first so its graph is freed before the main pass
                nb = min(args.energy_batch, B)
                ysub, Xs = r[:nb].flatten(1), []
                for _ in range(2):
                    e = torch.randn_like(r[:nb])
                    with autocast(dev):
                        v0 = net(e, fdet[:nb], hist[:nb], ctrl[:nb], ph[:nb], torch.zeros(nb, device=dev))
                    Xs.append((e + v0.float()).flatten(1))
                en = (((Xs[0] - ysub).norm(dim=1) + (Xs[1] - ysub).norm(dim=1)) / 2
                      - 0.5 * (Xs[0] - Xs[1]).norm(dim=1)).mean() / ysub.shape[1] ** 0.5
                ok = bool(torch.isfinite(en))
                if ok:
                    (args.w_energy * en).backward()
                    en_val = float(en)
                del Xs, ysub
            tau = torch.sigmoid(torch.randn(B, device=dev))
            eps = torch.randn_like(r)
            tv = tau.view(-1, 1, 1, 1, 1, 1)
            with autocast(dev):
                v = net((1 - tv) * eps + tv * r, fdet, hist, ctrl, ph, tau)
            loss = (v.float() - (r - eps)).pow(2).mean()
            if not all_ok(ok and bool(torch.isfinite(loss)), ddp, dev):
                nonfinite += 1
                skip("non-finite loss")
                if step >= args.steps:
                    break
                continue
            lf = float(loss)
            if ddp:
                lt = torch.tensor(lf, device=dev)
                dist.all_reduce(lt, op=dist.ReduceOp.AVG)
                lf = float(lt)
            if step > args.guard_start and len(loss_hist) >= 64 and lf > args.spike_mult * statistics.median(loss_hist):
                spikes += 1
                skip(f"loss spike {lf:.3f} > {args.spike_mult} x median {statistics.median(loss_hist):.3f}")
                if step >= args.steps:
                    break
                continue
            loss_hist.append(lf)
            loss.backward()
            if ddp:
                allreduce_grads(net.parameters())
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if not torch.isfinite(gnorm):
                nonfinite += 1
                skip("non-finite grad norm")
                if step >= args.steps:
                    break
                continue
            opt.step()
            sch.step()
            step += 1
            ema_update()
            if step % args.log_every == 0:
                dt = (time.time() - t0) / args.log_every
                t0 = time.time()
                log(f"step {step} loss {lf:.4f}" + (f" energy {en_val:.3f}" if en_val is not None else "")
                    + f" lr {sch.get_last_lr()[0]:.2e} gnorm {float(gnorm):.2f} {dt:.2f}s/it", flush=True)
            if step % args.ckpt_every == 0 or step >= args.steps:
                checkpoint()
            if rank == 0 and (step % args.val_every == 0 or step >= args.steps):
                validate()
            if step >= args.steps:
                break
    checkpoint()
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    log(f"done. non-finite skips {nonfinite} | spike skips {spikes}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--latents", required=True)
    ap.add_argument("--det", required=True, help="frozen operator checkpoint")
    ap.add_argument("--out", default="runs/flow")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--mlp-ratio", type=int, default=4)
    ap.add_argument("--amp", choices=["wintok", "none"], default="wintok",
                    help="residual normalizer: sigma_r x frozen window x token scales, or sigma_r only")
    ap.add_argument("--prefit-steps", type=int, default=2000, help="per amplitude head")
    ap.add_argument("--tok-floor", type=float, default=0.05, help="lower clamp of the token scale")
    ap.add_argument("--w-energy", type=float, default=0.2)
    ap.add_argument("--energy-batch", type=int, default=4, help="per rank under torchrun")
    ap.add_argument("--batch", type=int, default=8, help="per rank under torchrun")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--stat-batches", type=int, default=48)
    ap.add_argument("--guard-start", type=int, default=1000)
    ap.add_argument("--spike-mult", type=float, default=3.0)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args(argv)
    if args.amp == "wintok" and not 0.0 < args.tok_floor < 10.0:
        raise SystemExit("--tok-floor must be in (0, 10)")
    train(args)


if __name__ == "__main__":
    main()
