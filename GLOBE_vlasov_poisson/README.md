# GLOBE Vlasov-Poisson surrogate

Latent-space surrogate for the Vlasov-Poisson dynamics of an intense charged-particle
bunch in a periodic focusing channel with a grounded cylindrical bore. Three packages:

- `autoencoder/` compresses a charge-density frame ρ into a grid latent; `autoencoder/poisson.py`
  derives the potential φ and the field **E** from ρ by an exact Poisson solve on the bore.
- `operator_learning/` predicts the next h latents from the past k latents and the machine controls.
- `generative_model/` samples ensembles by flow matching on the residual of the operator's forecast.

## Setup

```bash
pip install -r requirements.txt          # Python >= 3.10
export BUNCHED_BEAM_STORE=/path/to/store # or pass --store to every command
```

## Data layout

```
<store>/norm.json                       packed-unit normalization (L_scale)
<store>/runs/run_XXXXXX/
    rho.zarr           (T, Z, Y, X) float16 charge density
    fields.zarr        (T, 5, Z, Y, X) float16 [phi, Ex, Ey, Ez, rho]   (full-field runs instead of rho.zarr)
    pack_meta.json     array_shape, channels, qa_gates, openpmd_mesh_metadata
    manifest_row.json  split (train/val/test), physical (EQ0_v_per_m2, E0T_v_per_m, q_tot_c),
                       context (bore_radius_m, box_half_width_m)
```

## Train and evaluate

Run from this directory, in this order. Every default is the production setting
(`--help` lists them); `--resume <ckpt>` continues a run; each `eval` prints the
training objective on a split.

```bash
# 1. autoencoder
python -m autoencoder.train --out runs/ae
python -m autoencoder.eval  --ckpt runs/ae/checkpoints/model.pt --split val
python -m autoencoder.poisson --run <full-field run id> --frames 60 --ckpt runs/ae/checkpoints/model.pt   # E from rho

# 2. operator: encode the store once, then train on the latent cache
python -m operator_learning.latents --ae-ckpt runs/ae/checkpoints/model.pt --out latents/ae
python -m operator_learning.train --latents latents/ae --out runs/operator
python -m operator_learning.eval  --ckpt runs/operator/checkpoints/model.pt --split val

# 3. residual flow on the frozen operator
python -m generative_model.train --latents latents/ae --det runs/operator/checkpoints/model.pt --out runs/flow
python -m generative_model.eval  --ckpt runs/flow/checkpoints/model.pt --split val
```

Notes

- Multi-GPU: `torchrun --nproc_per_node=N -m <module> ... --batch <per rank>`. The production
  operator and flow used two GPUs with `--batch 4` (`--energy-batch 2`), the same global batch
  as the single-GPU defaults.
- Sampling: `generative_model.sample.predict_block` draws one block; `rollout_ensemble`
  draws M trajectories, each member feeding its sampled residuals back into the operator
  (`feedback=False` anchors all members to the deterministic rollout). Both return
  standardized latents: un-standardize with the operator checkpoint's `zmean`/`zstd` and
  decode with the autoencoder's `decode`.
