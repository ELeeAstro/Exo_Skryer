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


def compute_ray_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
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
    if not XR.has_ray_data():
        return zero_ray_opacity(state, params)
    wavelengths = state["wl"]
    number_density = state["nd_lay"]
    density = state["rho_lay"]
    layer_vmr = state["vmr_lay"]
    layer_count = number_density.shape[0]

    master_wavelength = XR.ray_master_wavelength()
    if master_wavelength.shape != wavelengths.shape:
        raise ValueError("Rayleigh wavelength grid must match forward-model grid.")

    sigma_log = jnp.asarray(XR.ray_sigma_table(), dtype=jnp.float64)
    sigma_values = 10.0**sigma_log
    species_names = XR.ray_species_names()

    # Direct lookup - species names must match VMR keys exactly
    # VMR values are already JAX arrays, no need to wrap
    mixing_ratios = jnp.stack(
        [jnp.broadcast_to(layer_vmr[name], (layer_count,)) for name in species_names],
        axis=0,
    )

    # Use einsum to avoid transpose: (n_species, nwl) x (n_species, nlay) -> (nlay, nwl)
    sigma_weighted = jnp.einsum('sw,sl->lw', sigma_values, mixing_ratios)
    return (number_density[:, None] * sigma_weighted) / density[:, None]
