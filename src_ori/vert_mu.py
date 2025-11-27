"""
vert_mu.py
==========

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from __future__ import annotations

from typing import Dict, Tuple
import jax.numpy as jnp
from data_constants import CHEM_SPECIES_DATA


_SPECIES_MASS = {entry["symbol"]: float(entry["molecular_weight"]) for entry in CHEM_SPECIES_DATA}

def compute_mean_molecular_weight(
    vmr_lay: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, bool]:
    """
    Compute mean molecular weight from per-layer VMR dictionary.

    Parameters
    ----------
    vmr_lay : Dict[str, jnp.ndarray]
        Dictionary of per-layer volume mixing ratios for all species (including H2 and He).
        Each value should be a 1D array of shape (nlay,).

    Returns
    -------
    Tuple[jnp.ndarray, bool]
        Mean molecular weight profile (nlay,) and a flag indicating success.
    """

    # Sort species for consistent ordering (important for JAX tracing)
    species_list = sorted(species for species in vmr_lay.keys() if species in _SPECIES_MASS)

    # Stack VMRs and masses, then compute weighted sum
    vmr_arrays = [jnp.asarray(vmr_lay[sp]) for sp in species_list]
    masses = jnp.array([_SPECIES_MASS[sp] for sp in species_list])

    # Compute mean molecular weight: sum(VMR_i * mass_i)
    vmr_stack = jnp.stack(vmr_arrays, axis=0)  # (n_species, nlay)
    mu_profile = jnp.sum(vmr_stack * masses[:, None], axis=0)

    return mu_profile, True
