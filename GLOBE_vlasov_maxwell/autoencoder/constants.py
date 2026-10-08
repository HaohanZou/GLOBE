"""Frozen channel layout and per-channel constants of the packed store (plasma units)."""

# fields.zarr channel order
CHANNELS = ("rho_driver", "rho_witness", "rho_plasma_e", "rho_plasma_ion",
            "Ex", "Ey", "Ez", "Bx", "By", "Bz")
# decoded channels: the immobile-ion background is a fixed profile and is not
# predicted; the plasma-electron channel is decoded as the wake perturbation
# delta_rho_e = rho_e + rho_ion (rho_e = delta - rho_ion at deployment)
DECODED = ("rho_driver", "rho_witness", "delta_rho_e", "Ex", "Ey", "Ez", "Bx", "By", "Bz")
KEEP = (0, 1, 2, 4, 5, 6, 7, 8, 9)       # store index of each decoded channel's target (after wake_view)
RHO, E, B = slice(0, 3), slice(3, 6), slice(6, 9)

EPS0 = 8.8541878188e-12

# per-channel constants in DECODED order
FLOOR_RMS = (5e-3, 1e-2, 5e-3, 5e-3, 5e-3, 5e-3, 2e-3, 2e-3, 2e-4)   # rel-L2 denominator floors (per-element RMS)
CHANNEL_WEIGHTS = (2.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.15)      # rel-L2 channel weights (normalized to mean 1)
CHANNEL_RMS = (2.024e-1, 3.537e-1, 2.035e-1, 7.465e-2, 7.476e-2, 4.355e-2,
               7.408e-2, 7.387e-2, 1.281e-3)                          # dataset RMS: the projection metric
# encoder input scales in STORE order (channel 2 = delta_rho_e after wake_view)
INPUT_RMS = (0.149, 0.136, 0.15, 0.965, 4.58e-2, 4.57e-2, 3.89e-2, 4.34e-2, 4.35e-2, 1.3e-3)

SEAM_CELLS = 2                 # Gauss projection: residual masked within this many cells of the faces
SHELL_TAPER_CELLS, SHELL_TAPER_MIN = 16, 0.1   # loss weight ramps from 0.1 at the faces to 1 over 16 cells
