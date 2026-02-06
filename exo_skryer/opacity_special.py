"""
opacity_special.py
==================
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

__all__ = [
    "zero_special_opacity",
    "compute_hminus_bf_opacity",
    "compute_hminus_ff_opacity",
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


def compute_hminus_bf_opacity(state: Dict[str, jnp.ndarray], opac: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute H⁻ bound-free continuum mass opacity from the special registry.

    This function uses the precomputed H⁻ bound-free cross-section table and
    applies the `(n / ρ)`
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
    kappa_hminus_bf : `~jax.numpy.ndarray`, shape (nlay, nwl)
        H⁻ bound-free continuum mass opacity in cm² g⁻¹. Returns zeros when the
        special registry is not loaded or when `state["vmr_lay"]` does not
        provide an `"H-"` mixing ratio.
    """
    required = (
        "hminus_master_wavelength",
        "hminus_temperature_grid",
        "hminus_log10_temperature_grid",
        "hminus_bf_log10_sigma",
    )
    if any(k not in opac for k in required):
        return zero_special_opacity(state, params)

    wavelengths = state["wl"]
    master_wavelength = opac["hminus_master_wavelength"]
    if master_wavelength.shape != wavelengths.shape:
        raise ValueError("H- special wavelength grid must match the forward-model master grid.")

    layer_temperatures = state["T_lay"]
    number_density = state["nd_lay"]
    density = state["rho_lay"]
    layer_vmr = state["vmr_lay"]
    layer_count = state["nlay"]

    if "H-" not in layer_vmr:
        return zero_special_opacity(state, params)

    sigma_log = opac["hminus_bf_log10_sigma"]
    log_temperature_grid = opac["hminus_log10_temperature_grid"]
    temperature_grid = opac["hminus_temperature_grid"]
    sigma_values = 10.0 ** _interpolate_logsigma_1d(
        sigma_log, log_temperature_grid, temperature_grid, layer_temperatures
    )

    # VMR value is already a JAX array, no need to wrap
    vmr_hm = jnp.broadcast_to(layer_vmr["H-"], (layer_count,))
    normalization = vmr_hm * (number_density / density)
    return normalization[:, None] * sigma_values


def compute_hminus_ff_opacity(state: Dict[str, jnp.ndarray], opac: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute H⁻ free-free continuum mass opacity from the special registry.

    This term is treated as a two-body continuum source driven by electron and
    neutral-hydrogen abundances:

        κ_ff = f_e × f_H × (n_d)² / ρ × σ_ff(λ, T)

    where f_e and f_H are volume mixing ratios.
    """
    required = (
        "hminus_master_wavelength",
        "hminus_temperature_grid",
        "hminus_log10_temperature_grid",
        "hminus_ff_log10_sigma",
    )
    if any(k not in opac for k in required):
        return zero_special_opacity(state, params)

    wavelengths = state["wl"]
    master_wavelength = opac["hminus_master_wavelength"]
    if master_wavelength.shape != wavelengths.shape:
        raise ValueError("H- special wavelength grid must match the forward-model master grid.")

    layer_temperatures = state["T_lay"]
    number_density = state["nd_lay"]
    density = state["rho_lay"]
    layer_vmr = state["vmr_lay"]
    layer_count = state["nlay"]

    if "H" not in layer_vmr:
        raise ValueError(
            "H- free-free requires atomic hydrogen VMR key 'H' in state['vmr_lay']. "
            "For constant_vmr/constant_vmr_clr you can provide parameter "
            "'log_10_H_over_H2' to derive H from the filler."
        )
    if "log_10_ne_over_ntot" not in params:
        raise ValueError(
            "H- free-free requires parameter 'log_10_ne_over_ntot' (log10 of ne/n_tot)."
        )

    sigma_log = opac["hminus_ff_log10_sigma"]
    log_temperature_grid = opac["hminus_log10_temperature_grid"]
    temperature_grid = opac["hminus_temperature_grid"]
    sigma_values = 10.0 ** _interpolate_logsigma_1d(
        sigma_log, log_temperature_grid, temperature_grid, layer_temperatures
    )

    vmr_e = jnp.broadcast_to(10.0 ** params["log_10_ne_over_ntot"], (layer_count,))
    vmr_e = jnp.clip(vmr_e, 0.0, 1.0)
    vmr_h = jnp.broadcast_to(layer_vmr["H"], (layer_count,))
    normalization = (vmr_e * vmr_h) * ((number_density**2) / density)
    return normalization[:, None] * sigma_values


def compute_special_opacity(state: Dict[str, jnp.ndarray], opac: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
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
    compute_hminus_bf_opacity : H⁻ bound-free continuum term
    compute_hminus_ff_opacity : H⁻ free-free continuum term
    """
    kappa = compute_hminus_bf_opacity(state, opac, params)
    kappa = kappa + compute_hminus_ff_opacity(state, opac, params)
    return kappa
