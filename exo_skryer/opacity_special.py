"""
opacity_special.py
==================
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from . import build_opacities as XS

__all__ = [
    "zero_special_opacity",
    "compute_hminus_opacity",
    "compute_special_opacity"
]


def zero_special_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Return a zero special-opacity array.

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
        Zero-valued special-opacity array in cm² g⁻¹.
    """
    del params
    # Use shape directly without int() conversion for JIT compatibility
    shape = (state["nlay"], state["nwl"])
    return jnp.zeros(shape)


def _interpolate_logsigma_1d(
    sigma_log: jnp.ndarray,
    log_temperature_grid: jnp.ndarray,
    temperature_grid: jnp.ndarray,
    layer_temperatures: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate a log10 cross-section table on a log10(T) grid.

    Parameters
    ----------
    sigma_log : `~jax.numpy.ndarray`, shape `(nT, nwl)`
        Log₁₀ cross-sections as a function of temperature.
    log_temperature_grid : `~jax.numpy.ndarray`, shape `(nT,)`
        Log₁₀ of temperature grid (pre-computed).
    temperature_grid : `~jax.numpy.ndarray`, shape `(nT,)`
        Temperature grid in Kelvin (for minimum temperature check).
    layer_temperatures : `~jax.numpy.ndarray`, shape `(nlay,)`
        Layer temperatures in Kelvin.

    Returns
    -------
    sigma_interp_log : `~jax.numpy.ndarray`, shape `(nlay, nwl)`
        Log₁₀ cross-sections interpolated to each layer temperature.
    """
    log_t_layers = jnp.log10(layer_temperatures)
    log_t_grid = log_temperature_grid

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
    """Compute H⁻ continuum mass opacity from the CIA registry.

    This function uses the `"H-"` entry from the CIA registry (if loaded) as a
    temperature-dependent cross-section table and applies the `(n / ρ)`
    normalization appropriate for a single-absorber continuum term.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        Atmospheric state dictionary containing:

        - `wl` : `~jax.numpy.ndarray`, shape (nwl,)
            Forward-model wavelength grid in microns.
        - `T_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer temperatures in Kelvin.
        - `nd_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer total number density in cm⁻³.
        - `rho_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer mass density in g cm⁻³.
        - `vmr_lay` : dict[str, `~jax.numpy.ndarray`]
            Volume mixing ratios per species. Must include `"H-"` to enable this
            term. Values may be scalars or arrays with shape (nlay,).
        - `nlay` : int
            Number of atmospheric layers.
        - `nwl` : int
            Number of wavelength points.

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary (unused; kept for API compatibility).

    Returns
    -------
    kappa_hminus : `~jax.numpy.ndarray`, shape (nlay, nwl)
        H⁻ continuum mass opacity in cm² g⁻¹. Returns zeros when the CIA registry
        is not loaded, does not contain `"H-"`, or when `state["vmr_lay"]` does
        not provide an `"H-"` mixing ratio.
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
    log_temperature_grids = XS.cia_log10_temperature_grids()
    temperature_grids = XS.cia_temperature_grids()
    sigma_log = sigma_cube[hm_index]
    log_temperature_grid = log_temperature_grids[hm_index]
    temperature_grid = temperature_grids[hm_index]
    sigma_values = 10.0 ** _interpolate_logsigma_1d(sigma_log, log_temperature_grid, temperature_grid, layer_temperatures)

    # VMR value is already a JAX array, no need to wrap
    vmr_hm = jnp.broadcast_to(layer_vmr["H-"], (layer_count,))
    normalization = vmr_hm * (number_density / density)
    return normalization[:, None] * sigma_values


def compute_special_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute the summed special-opacity contribution.

    This is the top-level entry point for special opacity sources. It returns a
    single array with shape (nlay, nwl) in cm² g⁻¹ that can be added to the total
    opacity in the forward model.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        Forward-model state dictionary.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary (currently unused).

    Returns
    -------
    kappa_special : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Total special mass opacity in cm² g⁻¹.

    See Also
    --------
    compute_hminus_opacity : H⁻ continuum special term
    """
    return compute_hminus_opacity(state, params)
