"""Sampling: one residual block, one stochastic next-block prediction, ensemble rollouts.

Everything is in standardized latent space (the operator's zmean/zstd);
residuals are scaled by sigma_r (stored in the flow checkpoint) times the
model's amplitude, if any. Un-standardize before decoding.
"""
from __future__ import annotations

import torch

from operator_learning.train import autocast


@torch.no_grad()
def rollout_det(det, Zn, ctrl):
    """Block-autoregressive rollout of the operator on its own predictions, seeded with
    the first k true frames of the standardized trajectory Zn (T, C, g)."""
    k, h, T = det.k, det.h, Zn.shape[0]
    pred, t = Zn.clone(), k - 1
    while t + 1 < T:
        with autocast(Zn.device.type):
            y = det(pred[t - k + 1: t + 1][None], ctrl).float()
        n = min(h, T - t - 1)
        pred[t + 1: t + 1 + n] = y[0, :n]
        t += h
    return pred


@torch.no_grad()
def sample_residual(net, fdet, hist, ctrl, steps=16, generator=None):
    """One Euler-integrated draw of the normalized residual block (B, h, C, g)."""
    dev = hist.device
    x = torch.randn((hist.shape[0], net.h, net.ch, *hist.shape[-3:]), device=dev, generator=generator)
    for i in range(steps):
        tau = torch.full((x.shape[0],), i / steps, device=dev)
        with autocast(dev.type):
            v = net(x, fdet, hist, ctrl, tau).float()
        x = x + v / steps
    return x


@torch.no_grad()
def residual_sigma(net, sigma_r, fdet, hist):
    sig = sigma_r.view(1, 1, -1, 1, 1, 1).to(fdet.device)
    s = net.residual_scale(fdet, hist)
    return sig if s is None else sig * s


@torch.no_grad()
def predict_block(det, net, sigma_r, hist, ctrl, steps=16, generator=None):
    """Deterministic forecast + one sampled residual -> (next block, forecast)."""
    with autocast(hist.device.type):
        fdet = det(hist, ctrl).float()
    x = sample_residual(net, fdet, hist, ctrl, steps, generator)
    return fdet + residual_sigma(net, sigma_r, fdet, hist) * x, fdet


@torch.no_grad()
def rollout_ensemble(det, net, sigma_r, Zn, ctrl, members=4, steps=16, seed=0, feedback=True):
    """M-member ensemble over one trajectory: (M, T, C, g), seeded with the true first k frames.
    feedback=True: each member's history is its own sampled trajectory, i.e. the sampled
    residuals are fed back into the operator; False: every block is conditioned on the
    deterministic rollout and the residuals are not fed back."""
    dev, k, h, T = Zn.device, net.k, net.h, Zn.shape[0]
    gen = torch.Generator(device=dev).manual_seed(seed)
    pred = Zn[None].repeat(members, 1, 1, 1, 1, 1).clone()
    D = None if feedback else rollout_det(det, Zn, ctrl)
    ctrl_m = ctrl.expand(members, -1)
    t = k - 1
    while t + 1 < T:
        hist = pred[:, t - k + 1: t + 1] if feedback else D[t - k + 1: t + 1][None].expand(members, -1, -1, -1, -1, -1)
        y, _ = predict_block(det, net, sigma_r, hist, ctrl_m, steps, gen)
        n = min(h, T - t - 1)
        pred[:, t + 1: t + 1 + n] = y[:, :n]
        t += h
    return pred
