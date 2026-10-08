# GLOBE

Code for *Learn the Dynamics, Enforce the Physics: Generative Latent Operators for High-Dimensional Beam Evolution*.

GLOBE is a latent-space surrogate for 3D beam–field dynamics. An autoencoder compresses each frame into a grid latent, a neural operator predicts the next latents from the past ones and the machine controls, a flow-matching model samples the residual around that forecast, and a parameter-free physics operator enforces the field constraints on the decoded output.

## Layout

```
GLOBE_vlasov_poisson/   bunched-beam transport (electrostatic): predicts ρ, solves Poisson for E
GLOBE_vlasov_maxwell/   plasma-wakefield acceleration (electromagnetic): predicts ρ, E, B with a Maxwell projection
```

Each sub-project has the same three packages, run in this order:

| Package | Purpose |
|---|---|
| `autoencoder/` | compress frames into latents (`train.py`, `eval.py`; `poisson.py` or `maxwell.py` for the physics constraint) |
| `operator_learning/` | encode the dataset into a latent cache (`latents.py`), then train the operator (`train.py`, `eval.py`) |
| `generative_model/` | train the residual flow on the frozen operator (`train.py`, `eval.py`); `sample.py` does rollouts |

## Usage

```bash
cd GLOBE_vlasov_poisson            # or GLOBE_vlasov_maxwell
pip install -r requirements.txt
export BUNCHED_BEAM_STORE=/path/to/store   # PWFA_STORE for vlasov_maxwell, or pass --store

# 1. autoencoder
python -m autoencoder.train --out runs/ae

# 2. operator
python -m operator_learning.latents --ae-ckpt runs/ae/checkpoints/model.pt --out latents/ae
python -m operator_learning.train   --latents latents/ae --out runs/operator

# 3. residual flow
python -m generative_model.train --latents latents/ae --det runs/operator/checkpoints/model.pt --out runs/flow
```

Each step has a matching `eval` module (`python -m <package>.eval --ckpt <ckpt> --split val`). For rollouts use `rollout_ensemble` in `generative_model/sample.py`.

See the `README.md` inside each sub-project for the data layout, all options, and multi-GPU settings.
