# GLOBE Vlasov-Maxwell surrogate

Latent-space surrogate for the Vlasov-Maxwell dynamics of a beam-driven plasma wakefield
(plasma wakefield acceleration, PWFA) in a co-moving window. Three packages:

- `autoencoder/` compresses a 10-channel field frame into a grid latent and decodes the 9
  dynamical channels (driver, witness, wake perturbation, **E**, **B**); the projections in
  `autoencoder/maxwell.py` enforce div **B** = 0 and div **E** = ρ exactly on every decoded field.
- `operator_learning/` predicts the next h latents from the past k latents and the machine controls.
- `generative_model/` samples ensembles by flow matching on the residual of the operator's forecast.

## Setup

```bash
pip install -r requirements.txt   # Python >= 3.10
export PWFA_STORE=/path/to/store  # or pass --store to every command
```

## Data layout

```
<store>/runs/run_XXXXXX/
    fields.zarr        (T, 10, Z, Y, X) float16: rho_driver, rho_witness, rho_plasma_e, rho_plasma_ion,
                       Ex, Ey, Ez, Bx, By, Bz  (plasma units)
    pack_meta.json     arrays.fields.shape, channels, channel_scales_si, grid.grid_spacing_si_zyx,
                       burn_in_frames, qa_gates
    manifest_row.json  split (train/val/test), derived (q_driver_C, gamma_d, sigma_z_m, sigma_r_m,
                       x_off_m, q_witness_C, delay_m, witness_on)
```

Frame 0 is the field-solver initialization and is skipped. The model input replaces
`rho_plasma_e` by the wake perturbation ρ_e + ρ_ion; the ion background is a fixed profile
that is not predicted (`autoencoder.model.expand_to_store_channels` restores the 10 channels).

## Train and evaluate

Run from this directory, in this order. Every default is the production setting
(`--help` lists them); `--resume <ckpt>` continues a run; each `eval` prints the
training objective on a split.

```bash
# 1. autoencoder
python -m autoencoder.train --out runs/ae
python -m autoencoder.eval  --ckpt runs/ae/checkpoints/model.pt --split val

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
  runs used two GPUs: the autoencoder with `--batch 3` (global batch 6), the operator and the
  flow with `--batch 4` (`--energy-batch 2`), the same global batch as their single-GPU defaults.
- Sampling: `generative_model.sample.predict_block` draws one block; `rollout_ensemble`
  draws M trajectories, each member feeding its sampled residuals back into the operator
  (`feedback=False` anchors all members to the deterministic rollout). Both return
  standardized latents: un-standardize with the operator checkpoint's `zmean`/`zstd` and
  decode with the autoencoder's `decode`.
