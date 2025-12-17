"""
opacity_special.py
==================

Overview:
    Custom opacity sources that do not fit the standard line / CIA / Rayleigh /
    cloud separation.

    This module is intended as an extension point for additional continuum or
    ad-hoc opacity terms. Each special source should expose a small, JAX-safe
    function that returns a mass opacity on the forward-model wavelength grid.

Sections to complete:
    - Usage
    - Key Functions
    - Notes

Usage:
    The forward model calls `compute_special_opacity(state, params)` and adds
    the result into the total opacity budget. This module should be safe to
    call regardless of which special sources are configured; when no relevant
    data are available it returns zeros.

Key Functions:
    - `compute_special_opacity`: Computes and sums all enabled special opacity
      terms.
    - `compute_hminus_opacity`: Computes the H- continuum opacity using the
      H- entry in the CIA registry (if present).
    - `zero_special_opacity`: Convenience helper returning an all-zero array.

Notes:
    Special opacities are expected to be 2D arrays `(nlay, nwl)` and are
    broadcast across g-points in correlated-k RT (if used).
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from . import build_opacities as XS

__all__ = [
    "compute_hminus_opacity",
    "compute_special_opacity"
]


def zero_special_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Return a zero special-opacity array.

    Parameters
    ----------
    state : dict[str, jnp.ndarray]
        State dictionary containing scalar entries `nlay` and `nwl`.
    params : dict[str, jnp.ndarray]
        Unused; kept for API compatibility.

    Returns
    -------
    `~jax.numpy.ndarray`
        Array of zeros with shape `(nlay, nwl)`.
    """
    del params
    # Use shape directly without int() conversion for JIT compatibility
    shape = (state["nlay"], state["nwl"])
    return jnp.zeros(shape)


def _interpolate_logsigma_1d(
    sigma_log: jnp.ndarray,
    temperature_grid: jnp.ndarray,
    layer_temperatures: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate a log10 cross-section table on a log10(T) grid.

    Parameters
    ----------
    sigma_log : `~jax.numpy.ndarray`, shape `(nT, nwl)`
        Log10 cross-sections.
    temperature_grid : `~jax.numpy.ndarray`, shape `(nT,)`
        Temperature grid in K.
    layer_temperatures : `~jax.numpy.ndarray`, shape `(nlay,)`
        Layer temperatures in K.

    Returns
    -------
    `~jax.numpy.ndarray`
        Log10 cross-sections interpolated to layers, shape `(nlay, nwl)`.
    """
    log_t_layers = jnp.log10(layer_temperatures)
    log_t_grid = jnp.log10(temperature_grid)

    t_idx = jnp.searchsorted(log_t_grid, log_t_layers) - 1
    t_idx = jnp.clip(t_idx, 0, log_t_grid.shape[0] - 2)
    t_weight = (log_t_layers - log_t_grid[t_idx]) / (log_t_grid[t_idx + 1] - log_t_grid[t_idx])
    t_weight = jnp.clip(t_weight, 0.0, 1.0)

    s_t0 = sigma_log[t_idx, :]
    s_t1 = sigma_log[t_idx + 1, :]
    s_interp = (1.0 - t_weight)[:, None] * s_t0 + t_weight[:, None] * s_t1

    min_temp = temperature_grid[0]
    below_min = layer_temperatures < min_temp
    tiny = jnp.array(-199.0, dtype=s_interp.dtype)
    return jnp.where(below_min[:, None], tiny, s_interp)


def compute_hminus_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute H- continuum opacity (mass opacity).

    This uses the `H-` entry from the CIA registry (if loaded) and applies the
    `nd / rho` normalization appropriate for a single-absorber continuum.

    Parameters
    ----------
    state : dict[str, jnp.ndarray]
        State dictionary with at least `wl`, `T_lay`, `nd_lay`, `rho_lay`,
        `vmr_lay`, and scalar `nlay`/`nwl`.
    params : dict[str, jnp.ndarray]
        Unused; kept for API compatibility.

    Returns
    -------
    `~jax.numpy.ndarray`
        H- opacity in cm^2 g^-1 with shape `(nlay, nwl)`. Returns zeros when the
        CIA registry is not loaded or does not contain `H-`.
    """
    if not XS.has_cia_data():
        return zero_special_opacity(state, params)

    species_names = XS.cia_species_names()
    try:
        hm_index = [name.strip() for name in species_names].index("H-")
    except ValueError:
        return zero_special_opacity(state, params)

    wavelengths = state["wl"]
    master_wavelength = XS.cia_master_wavelength()
    if master_wavelength.shape != wavelengths.shape:
        raise ValueError("CIA wavelength grid must match the forward-model master grid.")

    layer_temperatures = state["T_lay"]
    number_density = state["nd_lay"]
    density = state["rho_lay"]
    layer_vmr = state["vmr_lay"]
    layer_count = state["nlay"]

    if "H-" not in layer_vmr:
        return zero_special_opacity(state, params)

    sigma_cube = XS.cia_sigma_cube()
    temperature_grids = XS.cia_temperature_grids()
    sigma_log = sigma_cube[hm_index]
    temperature_grid = temperature_grids[hm_index]
    sigma_values = 10.0 ** _interpolate_logsigma_1d(sigma_log, temperature_grid, layer_temperatures)

    # VMR value is already a JAX array, no need to wrap
    vmr_hm = jnp.broadcast_to(layer_vmr["H-"], (layer_count,))
    normalization = vmr_hm * (number_density / density)
    return normalization[:, None] * sigma_values


def compute_special_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute the summed special opacity contribution.

    Parameters
    ----------
    state : dict[str, jnp.ndarray]
        Forward-model state dictionary.
    params : dict[str, jnp.ndarray]
        Parameter dictionary (currently unused).

    Returns
    -------
    `~jax.numpy.ndarray`
        Total special opacity in cm^2 g^-1 with shape `(nlay, nwl)`.
    """
    return compute_hminus_opacity(state, params)
