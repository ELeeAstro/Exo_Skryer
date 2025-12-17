"""
TODO: Module-level docstring placeholder.
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from .data_constants import CHEM_SPECIES_DATA


_SPECIES_MASS = {entry["symbol"]: float(entry["molecular_weight"]) for entry in CHEM_SPECIES_DATA}

__all__ = [
    "constant_mu",
    "compute_mu"
]


def constant_mu(params: Dict[str, jnp.ndarray], nlay: int) -> jnp.ndarray:
    """Generate a constant mean molecular weight profile.

    Parameters
    ----------
    params : dict[str, jnp.ndarray]
        Dictionary containing 'mu' for the mean molecular weight [amu].
    nlay : int
        Number of atmospheric layers.

    Returns
    -------
    `~jax.numpy.ndarray`
        Mean molecular weight profile of shape (nlay,).
    """
    if "mu" not in params:
        raise ValueError("vert_mu='constant' requires a 'mu' parameter.")
    mu_const = params["mu"]
    return jnp.full((nlay,), mu_const)


def compute_mu(vmr_lay: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute mean molecular weight from volume mixing ratios.

    Parameters
    ----------
    vmr_lay : dict[str, jnp.ndarray]
        Dictionary mapping species symbols to their VMR profiles. Each
        value should be an array of shape (nlay,).

    Returns
    -------
    `~jax.numpy.ndarray`
        Mean molecular weight profile of shape (nlay,) [amu].
    """
    species_list = sorted(species for species in vmr_lay.keys() if species in _SPECIES_MASS)
    if not species_list:
        raise ValueError("No valid species provided to compute mean molecular weight.")

    vmr_arrays = [vmr_lay[sp] for sp in species_list]
    masses = jnp.array([_SPECIES_MASS[sp] for sp in species_list])
    vmr_stack = jnp.stack(vmr_arrays, axis=0)
    mu_profile = jnp.sum(vmr_stack * masses[:, None], axis=0)
    return mu_profile
