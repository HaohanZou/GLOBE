"""Encode the store once with a trained autoencoder into a latent cache.

Writes run_XXXXXX.npy (T, C, Lz, Ly, Lx) float16 per run and meta.json with
each run's split and machine controls plus the autoencoder checkpoint hash. A
cache is valid only for the autoencoder that produced it: re-running against
an existing cache with a different checkpoint is an error, re-running with
the same one resumes.

  python -m operator_learning.latents --ae-ckpt runs/ae/checkpoints/model.pt --store <store> --out latents/ae
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib

import numpy as np
import torch

from autoencoder.dataset import STORE, BunchedBeamStore
from autoencoder.model import build_model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--ae-ckpt", required=True)
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args(argv)
    dev = args.device
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ae_ckpt, map_location=dev, weights_only=False)
    ae = build_model(argparse.Namespace(**ck["args"])).to(dev).eval()
    ae.load_state_dict(ck["model"])
    sha = hashlib.sha256(pathlib.Path(args.ae_ckpt).read_bytes()).hexdigest()[:16]
    meta_path = out / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("ae_sha256_16") != sha:
            raise SystemExit(f"{out} was encoded by a different autoencoder "
                             f"({old.get('ae_sha256_16')} != {sha}); use a fresh --out")
    elif any(out.glob("run_*.npy")):
        raise SystemExit(f"{out} holds arrays of unknown provenance; use a fresh --out")

    st = BunchedBeamStore(args.store)
    runs = sorted(r for s in ("train", "val", "test") for r in st.split_runs(s))
    meta = dict(ae_ckpt=str(args.ae_ckpt), ae_sha256_16=sha, ae_step=ck.get("step"),
                T=st.T, runs={})
    if not meta_path.exists():   # provenance stub, so an interrupted first pass can resume
        meta_path.with_suffix(".json.tmp").write_text(json.dumps(meta, indent=2))
        os.replace(meta_path.with_suffix(".json.tmp"), meta_path)
    for i, r in enumerate(runs):
        dst = out / f"run_{r:06d}.npy"
        if not dst.exists():
            zs = []
            with torch.no_grad():
                for s in range(0, st.T, args.batch):
                    x = np.stack([np.asarray(st.fields(r)[t]) for t in range(s, min(s + args.batch, st.T))])
                    zs.append(ae.encode(torch.from_numpy(x).float().to(dev)).half().cpu().numpy())
            with open(str(dst) + ".tmp", "wb") as f:
                np.save(f, np.concatenate(zs, 0))
            os.replace(str(dst) + ".tmp", dst)
        meta["runs"][str(r)] = dict(split=st.row(r)["split"],
                                    controls=[float(c) for c in st.controls(r)])
        if (i + 1) % 10 == 0 or i + 1 == len(runs):
            print(f"  {i + 1}/{len(runs)} runs", flush=True)
    meta_path.with_suffix(".json.tmp").write_text(json.dumps(meta, indent=2))
    os.replace(meta_path.with_suffix(".json.tmp"), meta_path)
    print(f"done: {len(runs)} runs -> {out}", flush=True)


if __name__ == "__main__":
    main()
