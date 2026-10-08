"""Reconstruction objective of a trained autoencoder on a split.

  python -m autoencoder.eval --ckpt runs/ae/checkpoints/model.pt --store <store> --split val
"""
from __future__ import annotations

import argparse
import json

import torch

from .dataset import STORE, BunchedBeamFrameDataset, BunchedBeamStore, make_loader, strided_subset
from .model import build_model
from .train import validate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--batch", type=int, default=0, help="0 = the checkpoint's batch size")
    ap.add_argument("--json", default="")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args(argv)

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    model = build_model(a).to(args.device)
    model.load_state_dict(ck["model"])
    batch = args.batch or a.batch
    st = BunchedBeamStore(args.store)
    loader = make_loader(strided_subset(BunchedBeamFrameDataset(st, args.split), args.batches * batch),
                         batch, shuffle=False, num_workers=4)
    res = validate(model, loader, args.device, a, n_batches=args.batches, amp=False)
    print(f"autoencoder step {ck['step']} on {args.batches} {args.split} batches: "
          + " | ".join(f"{k} {v:.4f}" for k, v in res.items()), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    main()
