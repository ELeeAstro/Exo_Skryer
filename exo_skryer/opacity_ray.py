"""
opacity_ray.py
==============
"""

from __future__ import annotations
from typing import Dict

import jax.numpy as jnp
from . import registry_ray as XR

__all__ = [
    "zero_ray_opacity",
    "compute_ray_opacity"
]


def zero_ray_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Return a zero Rayleigh scattering opacity array.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        State dictionary containing:

        - `nlay` : int
            Number of atmospheric layers.
        - `nwl` : int
            Number of wavelength points.

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary (unused; kept for API compatibility).

    Returns
    -------
    zeros : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Zero-valued Rayleigh opacity array in cm² g⁻¹.
    """
    # Use shape directly without int() conversion for JIT compatibility
    shape = (state["nlay"], state["nwl"])
    return jnp.zeros(shape)


def compute_ray_opacity(state: Dict[str, jnp.ndarray], opac: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute Rayleigh scattering mass opacity for the configured scatterers.

    This function converts precomputed Rayleigh scattering cross-sections from
    `exo_skryer.registry_ray` into a layer-by-wavelength mass opacity in cm² g⁻¹.
    If no Rayleigh data are loaded, it returns zeros with the expected shape.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        Atmospheric state dictionary containing:

        - `wl` : `~jax.numpy.ndarray`, shape `(nwl,)`
            Forward-model wavelength grid in microns.
        - `nd_lay` : `~jax.numpy.ndarray`, shape `(nlay,)`
            Layer total number density in cm⁻³.
        - `rho_lay` : `~jax.numpy.ndarray`, shape `(nlay,)`
            Layer mass density in g cm⁻³.
        - `vmr_lay` : dict[str, `~jax.numpy.ndarray`]
            Volume mixing ratios for each Rayleigh species. Keys must match
            `registry_ray.ray_species_names()`. Values may be scalars or arrays
            with shape (nlay,).

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary (unused; kept for API compatibility).

    Returns
    -------
    kappa_ray : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Rayleigh scattering mass opacity in cm² g⁻¹ at each layer and wavelength.
    """
    wavelengths = state["wl"]
    number_density = state["nd_lay"]
    density = state["rho_lay"]
    layer_vmr = state["vmr_lay"]
    layer_count = number_density.shape[0]

    master_wavelength = opac["ray_master_wavelength"]
    if master_wavelength.shape != wavelengths.shape:
        raise ValueError("Rayleigh wavelength grid must match forward-model grid.")

    sigma_log = opac["ray_sigma_table"]
    sigma_values = 10.0 ** sigma_log.astype(jnp.float64)  # (n_species, nwl)
    species_names = XR.ray_species_names()
    # Accumulate directly into (nlay, nwl) without materializing (n_species, nlay).
    # Use a Python loop so species-name dict lookups happen at trace time (no dynamic indexing).
    sigma_weighted = jnp.zeros((layer_count, wavelengths.shape[0]), dtype=sigma_values.dtype)
    for i, name in enumerate(species_names):
        vmr_i = jnp.broadcast_to(layer_vmr[name], (layer_count,))  # (nlay,)
        sigma_weighted = sigma_weighted + vmr_i[:, None] * sigma_values[i][None, :]
    return (number_density[:, None] * sigma_weighted) / density[:, None]
