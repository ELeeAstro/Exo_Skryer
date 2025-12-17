"""
opacity_ray.py
==============

Overview:
    Rayleigh-scattering opacity contribution for the forward model.

    This module computes a per-layer, per-wavelength opacity (cm^2 g^-1) from
    species-specific Rayleigh cross-sections stored in `registry_ray`.

Sections to complete:
    - Usage
    - Key Functions
    - Notes

Usage:
    The retrieval pipeline passes a shared `state` dictionary into the forward
    model components. This module expects the forward-model wavelength grid in
    `state["wl"]` to exactly match the wavelength grid loaded by
    `registry_ray.ray_master_wavelength()`.

Key Functions:
    - `compute_ray_opacity`: Computes the Rayleigh opacity (cm^2 g^-1) on the
      model grid, returning zeros when no Rayleigh registry data are available.
    - `zero_ray_opacity`: Convenience helper returning an all-zero array with
      the expected shape.

Notes:
    The Rayleigh registry provides log10 cross-sections per species; these are
    converted to linear space and combined with per-layer number density and
    species mixing ratios from `state["vmr_lay"]`. The result is divided by the
    layer mass density to produce cm^2 g^-1.
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
    """Return a zero Rayleigh opacity array.

    Parameters
    ----------
    state : dict[str, jnp.ndarray]
        State dictionary containing scalar entries `nlay` (number of layers)
        and `nwl` (number of wavelengths).
    params : dict[str, jnp.ndarray]
        Unused; kept for API compatibility.

    Returns
    -------
    `~jax.numpy.ndarray`
        Array of zeros with shape `(nlay, nwl)`.
    """
    # Use shape directly without int() conversion for JIT compatibility
    shape = (state["nlay"], state["nwl"])
    return jnp.zeros(shape)


def compute_ray_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute Rayleigh scattering opacity.

    Parameters
    ----------
    state : dict[str, jnp.ndarray]
        State dictionary with at least:

        - `wl` : `~jax.numpy.ndarray`, shape `(nwl,)`
            Forward-model wavelength grid.
        - `nd_lay` : `~jax.numpy.ndarray`, shape `(nlay,)`
            Layer number density.
        - `rho_lay` : `~jax.numpy.ndarray`, shape `(nlay,)`
            Layer mass density.
        - `vmr_lay` : Mapping[str, Any]
            Mapping from species name to volume mixing ratio per layer. Values
            may be a scalar or a length-`nlay` vector.
    params : dict[str, jnp.ndarray]
        Unused; kept for API compatibility.

    Returns
    -------
    `~jax.numpy.ndarray`
        Rayleigh opacity in cm^2 g^-1 with shape `(nlay, nwl)`. Returns zeros
        when the Rayleigh registry has no data loaded.
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

    sigma_log = XR.ray_sigma_table()
    sigma_values = 10.0**sigma_log
    species_names = XR.ray_species_names()

    # Direct lookup - species names must match VMR keys exactly
    # VMR values are already JAX arrays, no need to wrap
    mixing_ratios = jnp.stack(
        [jnp.broadcast_to(layer_vmr[name], (layer_count,)) for name in species_names],
        axis=0,
    )

    sigma_weighted = jnp.sum(sigma_values[:, :, None] * mixing_ratios[:, None, :], axis=0)  # (nwl, nlay)
    return (number_density[:, None] * sigma_weighted.T) / density[:, None]
