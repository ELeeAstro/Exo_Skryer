"""
vert_mu.py
==========
"""

from __future__ import annotations

from typing import Callable, Dict

import jax.numpy as jnp

from .data_constants import CHEM_SPECIES_DATA


_SPECIES_MASS = {entry["symbol"]: float(entry["molecular_weight"]) for entry in CHEM_SPECIES_DATA}

__all__ = [
    "constant_mu",
    "compute_mu",
    "build_compute_mu",
]


def constant_mu(params: Dict[str, jnp.ndarray], nlay: int) -> jnp.ndarray:
    """Return a constant mean molecular weight (μ) profile.

    Parameters
    ----------
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `mu` : float
            Mean molecular weight in g mol⁻¹
    nlay : int
        Number of atmospheric layers.

    Returns
    -------
    mu_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Mean molecular weight profile in g mol⁻¹.
    """
    if "mu" not in params:
        raise ValueError("vert_mu='constant' requires a 'mu' parameter.")
    mu_const = params["mu"]
    return jnp.full((nlay,), mu_const)


def compute_mu(vmr_lay: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute mean molecular weight from volume mixing ratios.

    Parameters
    ----------
    vmr_lay : dict[str, `~jax.numpy.ndarray`]
        Dictionary mapping species symbols to their VMR profiles. Each value
        should be an array of shape (nlay,).

    Returns
    -------
    mu_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Mean molecular weight profile in g mol⁻¹.
    """
    # Electrons (e-) have negligible mass and should not affect mean molecular weight.
    species_list = sorted(
        species for species in vmr_lay.keys() if species in _SPECIES_MASS and species != "e-"
    )
    if not species_list:
        raise ValueError("No valid species provided to compute mean molecular weight.")

    vmr_arrays = [vmr_lay[sp] for sp in species_list]
    masses = jnp.array([_SPECIES_MASS[sp] for sp in species_list])
    vmr_stack = jnp.stack(vmr_arrays, axis=0)
    mu_profile = jnp.sum(vmr_stack * masses[:, None], axis=0)
    return mu_profile


def build_compute_mu(species_order: tuple[str, ...]) -> Callable[[Dict[str, jnp.ndarray]], jnp.ndarray]:
    """Build a mean-molecular-weight kernel with a fixed species ordering."""
    valid_species = tuple(
        species for species in species_order
        if species in _SPECIES_MASS and species != "e-"
    )
    if not valid_species:
        raise ValueError("No valid non-electron species were provided for mean molecular weight.")

    masses = jnp.asarray([_SPECIES_MASS[species] for species in valid_species])

    def _compute_mu_fixed(vmr_lay: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        vmr_stack = jnp.stack([vmr_lay[species] for species in valid_species], axis=0)
        return jnp.sum(vmr_stack * masses[:, None], axis=0)

    return _compute_mu_fixed
