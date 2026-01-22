"""
opacity_ck.py
=============
"""

from typing import Dict

import jax
import jax.numpy as jnp
from jax import lax

from . import build_opacities as XS
from .data_constants import amu, bar
from .ck_mix_RORR import mix_k_tables_rorr
from .ck_mix_PRAS import mix_k_tables_pras

__all__ = [
    "zero_ck_opacity",
    "compute_ck_opacity",
    "compute_ck_opacity_perspecies"
]


def _interpolate_sigma_log(layer_pressures_bar: jnp.ndarray, layer_temperatures: jnp.ndarray) -> jnp.ndarray:
    """Bilinear interpolation of correlated-k cross-sections on (log P, log T) grids.

    This function retrieves pre-loaded correlated-k opacity tables from the opacity
    registry and interpolates them to the specified atmospheric layer conditions using
    bilinear interpolation in log₁₀(P)-log₁₀(T) space. The interpolation is performed
    separately for each species and returns cross-sections still in log₁₀ space.

    Parameters
    ----------
    layer_pressures_bar : `~jax.numpy.ndarray`, shape (nlay,)
        Atmospheric layer pressures in bar.
    layer_temperatures : `~jax.numpy.ndarray`, shape (nlay,)
        Atmospheric layer temperatures in Kelvin.

    Returns
    -------
    sigma_interp : `~jax.numpy.ndarray`, shape (nspecies, nlay, nwl, ng)
        Interpolated cross-sections in log₁₀ space with units of log₁₀(cm² molecule⁻¹).
        The axes represent:
        - nspecies: Number of absorbing species
        - nlay: Number of atmospheric layers
        - nwl: Number of wavelength bins
        - ng: Number of g-points per wavelength bin
    """
    sigma_cube = XS.ck_sigma_cube()
    log_p_grid = XS.ck_log10_pressure_grid()
    log_temperature_grids = XS.ck_log10_temperature_grids()

    log_p_layers = jnp.log10(layer_pressures_bar)
    log_t_layers = jnp.log10(layer_temperatures)

    def _interp_one_layer(log_p: jnp.ndarray, log_t: jnp.ndarray) -> jnp.ndarray:
        # Pressure bracket indices and weights (shared across species)
        p_idx = jnp.searchsorted(log_p_grid, log_p) - 1
        p_idx = jnp.clip(p_idx, 0, log_p_grid.shape[0] - 2)
        p_weight = (log_p - log_p_grid[p_idx]) / (log_p_grid[p_idx + 1] - log_p_grid[p_idx])
        p_weight = jnp.clip(p_weight, 0.0, 1.0)

        def _interp_one_species(sigma_4d: jnp.ndarray, log_temp_grid: jnp.ndarray) -> jnp.ndarray:
            # Temperature bracket indices and weights (species-dependent)
            t_idx = jnp.searchsorted(log_temp_grid, log_t) - 1
            t_idx = jnp.clip(t_idx, 0, log_temp_grid.shape[0] - 2)
            t_weight = (log_t - log_temp_grid[t_idx]) / (log_temp_grid[t_idx + 1] - log_temp_grid[t_idx])
            t_weight = jnp.clip(t_weight, 0.0, 1.0)

            # Bilinear interpolation in (logT, logP). Indices are scalars here, so outputs are (nwl, ng).
            s_t0_p0 = sigma_4d[t_idx, p_idx, :, :]
            s_t0_p1 = sigma_4d[t_idx, p_idx + 1, :, :]
            s_t1_p0 = sigma_4d[t_idx + 1, p_idx, :, :]
            s_t1_p1 = sigma_4d[t_idx + 1, p_idx + 1, :, :]

            s_t0 = (1.0 - p_weight) * s_t0_p0 + p_weight * s_t0_p1
            s_t1 = (1.0 - p_weight) * s_t1_p0 + p_weight * s_t1_p1
            return (1.0 - t_weight) * s_t0 + t_weight * s_t1

        sigma_log_layer = jax.vmap(_interp_one_species)(sigma_cube, log_temperature_grids)  # (nspec, nwl, ng)
        return sigma_log_layer.astype(jnp.float64)

    # NOTE: This returns a large (nspec, nlay, nwl, ng) array; callers that care about peak
    # memory should perform layer-wise interpolation + mixing instead of using this helper.
    return jax.vmap(_interp_one_layer)(log_p_layers, log_t_layers)

def _get_ck_quadrature(state: Dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract g-points and quadrature weights for correlated-k integration.

    This function retrieves the g-points and their associated quadrature weights
    used for integrating over the k-distribution. The g-points represent cumulative
    probability values in [0, 1], and the weights are typically Gaussian quadrature
    weights that sum to 1.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        State dictionary that may contain pre-loaded 'g_weights' array.
        If not present, weights are retrieved from the opacity registry.

    Returns
    -------
    g_points : `~jax.numpy.ndarray`, shape (ng,)
        Cumulative probability points where k-distribution is sampled, in [0, 1].
    weights : `~jax.numpy.ndarray`, shape (ng,)
        Quadrature weights for numerical integration over g-space. Sum to 1.0.
    """
    g_points_all = XS.ck_g_points()
    g_weights_all = state.get("g_weights")
    if g_weights_all is None:
        g_weights_all = XS.ck_g_weights()

    # Values are already JAX arrays, no need to wrap
    if g_points_all.ndim == 1:
        g_eval = g_points_all
    else:
        g_eval = g_points_all[0]

    if g_weights_all.ndim == 1:
        weights = g_weights_all
    else:
        weights = g_weights_all[0]

    return g_eval, weights


def zero_ck_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Return a zero correlated-k opacity array.

    This function is used as a fallback when correlated-k opacities are disabled
    in the configuration. It maintains API compatibility with `compute_ck_opacity()`
    so the forward model can seamlessly switch between CK enabled/disabled.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        State dictionary containing:

        - `p_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer pressures (used only to determine array size).
        - `wl` : `~jax.numpy.ndarray`, shape (nwl,)
            Wavelength grid (used only to determine array size).

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary (unused; kept for API compatibility).

    Returns
    -------
    zeros : `~jax.numpy.ndarray`, shape (nlay, nwl, ng)
        Zero-valued correlated-k opacity array in cm² g⁻¹.
    """
    layer_pressures = state["p_lay"]
    wavelengths = state["wl"]
    layer_count = layer_pressures.shape[0]
    wavelength_count = wavelengths.shape[0]

    # Get number of g-points from loaded ck data.
    # Registry may store weights as (n_species, n_g); use state-provided weights if available.
    g_weights = state.get("g_weights")
    if g_weights is None:
        g_weights = XS.ck_g_weights()
        if g_weights.ndim > 1:
            g_weights = g_weights[0]
    n_g = g_weights.shape[-1]

    return jnp.zeros((layer_count, wavelength_count, n_g))


def compute_ck_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute correlated-k opacity with multi-species mixing.

    This function calculates the total atmospheric opacity using the correlated-k
    approximation. It performs the following steps:
    1. Interpolates pre-loaded k-tables to atmospheric (P, T) conditions
    2. Mixes k-distributions from multiple species using RORR or PRAS scheme
    3. Converts from cross-section (cm² molecule⁻¹) to mass opacity (cm² g⁻¹)

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        Atmospheric state dictionary containing:

        - `p_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer pressures in dyne cm⁻².
        - `T_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer temperatures in Kelvin.
        - `mu_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Mean molecular weight per layer in amu.
        - `vmr_lay` : dict[str, `~jax.numpy.ndarray`]
            Volume mixing ratios for each species. Keys must match species
            names in the loaded CK tables. Values can be scalars or arrays
            with shape (nlay,).
        - `wl` : `~jax.numpy.ndarray`, shape (nwl,)
            Wavelength grid in microns.
        - `ck_mix` : str or int, optional
            Mixing method selector. Either 'RORR' (default, code=1) or 'PRAS'
            (code=2). Can be specified as string or integer code.
        - `g_weights` : `~jax.numpy.ndarray`, optional
            Quadrature weights for g-point integration. If not provided,
            retrieved from opacity registry.

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary (unused; VMRs come from state['vmr_lay']).
        Kept for API compatibility with other opacity functions.

    Returns
    -------
    kappa_ck : `~jax.numpy.ndarray`, shape (nlay, nwl, ng)
        Total atmospheric mass opacity in cm² g⁻¹ at each layer, wavelength
        bin, and g-point.
    """
    layer_pressures = state["p_lay"]
    layer_temperatures = state["T_lay"]
    layer_mu = state["mu_lay"]
    layer_count = layer_pressures.shape[0]

    # Get species names and mixing ratios
    species_names = XS.ck_species_names()
    layer_vmr = state["vmr_lay"]

    # Direct lookup - species names must match VMR keys exactly
    # VMR values are already JAX arrays, no need to wrap
    mixing_ratios = jnp.stack(
        [jnp.broadcast_to(layer_vmr[name], (layer_count,)) for name in species_names],
        axis=0,
    )

    g_points, g_weights = _get_ck_quadrature(state)

    # Get mixing method from state (default to RORR).
    # Backwards-compatible: accept either a string ("RORR"/"PRAS") or an int code.
    ck_mix_raw = state.get("ck_mix", 1)
    if isinstance(ck_mix_raw, str):
        ck_mix_code = 2 if ck_mix_raw.upper() == "PRAS" else 1
    else:
        # Avoid int() conversion for JIT compatibility
        ck_mix_code = ck_mix_raw

    # Layer-wise interpolation + mixing to avoid materializing (n_species, n_layers, n_wl, n_g).
    sigma_cube = XS.ck_sigma_cube()
    log_p_grid = XS.ck_log10_pressure_grid()
    log_temperature_grids = XS.ck_log10_temperature_grids()

    log_p_layers = jnp.log10(layer_pressures / bar)
    log_t_layers = jnp.log10(layer_temperatures)

    n_species = sigma_cube.shape[0]
    n_wl = sigma_cube.shape[-2]
    n_g = sigma_cube.shape[-1]

    def _interp_sigma_log_layer(log_p: jnp.ndarray, log_t: jnp.ndarray) -> jnp.ndarray:
        p_idx = jnp.searchsorted(log_p_grid, log_p) - 1
        p_idx = jnp.clip(p_idx, 0, log_p_grid.shape[0] - 2)
        p_weight = (log_p - log_p_grid[p_idx]) / (log_p_grid[p_idx + 1] - log_p_grid[p_idx])
        p_weight = jnp.clip(p_weight, 0.0, 1.0)

        def _interp_one_species(sigma_4d: jnp.ndarray, log_temp_grid: jnp.ndarray) -> jnp.ndarray:
            t_idx = jnp.searchsorted(log_temp_grid, log_t) - 1
            t_idx = jnp.clip(t_idx, 0, log_temp_grid.shape[0] - 2)
            t_weight = (log_t - log_temp_grid[t_idx]) / (log_temp_grid[t_idx + 1] - log_temp_grid[t_idx])
            t_weight = jnp.clip(t_weight, 0.0, 1.0)

            s_t0_p0 = sigma_4d[t_idx, p_idx, :, :]
            s_t0_p1 = sigma_4d[t_idx, p_idx + 1, :, :]
            s_t1_p0 = sigma_4d[t_idx + 1, p_idx, :, :]
            s_t1_p1 = sigma_4d[t_idx + 1, p_idx + 1, :, :]

            s_t0 = (1.0 - p_weight) * s_t0_p0 + p_weight * s_t0_p1
            s_t1 = (1.0 - p_weight) * s_t1_p0 + p_weight * s_t1_p1
            return (1.0 - t_weight) * s_t0 + t_weight * s_t1

        return jax.vmap(_interp_one_species)(sigma_cube, log_temperature_grids).astype(jnp.float64)

    def _mix_one_layer(layer_idx: jnp.ndarray, out: jnp.ndarray) -> jnp.ndarray:
        log_p = log_p_layers[layer_idx]
        log_t = log_t_layers[layer_idx]
        sigma_log_layer = _interp_sigma_log_layer(log_p, log_t)  # (nspec, nwl, ng)
        vmr_layer = mixing_ratios[:, layer_idx]                  # (nspec,)

        if ck_mix_code == 2:
            mixed = mix_k_tables_pras(
                sigma_log_layer[:, None, :, :],
                vmr_layer[:, None],
                g_points,
                g_weights,
            )[0]
        else:
            mixed = mix_k_tables_rorr(
                10.0 ** sigma_log_layer[:, None, :, :],
                vmr_layer[:, None],
                g_points,
                g_weights,
            )[0]

        out = out.at[layer_idx].set(mixed)
        return out

    mixed_sigma = lax.fori_loop(
        0,
        layer_count,
        _mix_one_layer,
        jnp.zeros((layer_count, n_wl, n_g), dtype=jnp.float64),
    )

    # Convert to mass opacity (cm^2 / g)
    total_opacity = mixed_sigma / (layer_mu[:, None, None] * amu)

    return total_opacity


def compute_ck_opacity_perspecies(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute per-species correlated-k opacities WITHOUT mixing.

    This function is used with the transmission multiplication random overlap
    method (ck_mix: trans), where species mixing happens during the RT
    calculation rather than at the opacity computation stage.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        Atmospheric state dictionary containing:

        - `p_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer pressures in dyne cm⁻².
        - `T_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer temperatures in Kelvin.
        - `mu_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Mean molecular weight per layer in amu.
        - `vmr_lay` : dict[str, `~jax.numpy.ndarray`]
            Volume mixing ratios for each species.
        - `wl` : `~jax.numpy.ndarray`, shape (nwl,)
            Wavelength grid in microns.

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary (unused; kept for API compatibility).

    Returns
    -------
    sigma_perspecies : `~jax.numpy.ndarray`, shape (n_species, nlay, nwl, ng)
        Per-species mass opacities in cm² g⁻¹. Note: these are NOT yet
        weighted by VMR - that happens in the RT calculation.
    vmr_perspecies : `~jax.numpy.ndarray`, shape (n_species, nlay)
        Volume mixing ratios for each species at each layer.
    """
    layer_pressures = state["p_lay"]
    layer_temperatures = state["T_lay"]
    layer_mu = state["mu_lay"]
    layer_count = layer_pressures.shape[0]

    # Get species names and mixing ratios
    species_names = XS.ck_species_names()
    layer_vmr = state["vmr_lay"]

    # Stack VMRs for all species
    mixing_ratios = jnp.stack(
        [jnp.broadcast_to(layer_vmr[name], (layer_count,)) for name in species_names],
        axis=0,
    )  # (n_species, nlay)

    # Get k-table data
    sigma_cube = XS.ck_sigma_cube()
    log_p_grid = XS.ck_log10_pressure_grid()
    log_temperature_grids = XS.ck_log10_temperature_grids()

    log_p_layers = jnp.log10(layer_pressures / bar)
    log_t_layers = jnp.log10(layer_temperatures)

    def _interp_sigma_log_layer(log_p: jnp.ndarray, log_t: jnp.ndarray) -> jnp.ndarray:
        """Interpolate cross-sections for all species at one layer."""
        p_idx = jnp.searchsorted(log_p_grid, log_p) - 1
        p_idx = jnp.clip(p_idx, 0, log_p_grid.shape[0] - 2)
        p_weight = (log_p - log_p_grid[p_idx]) / (log_p_grid[p_idx + 1] - log_p_grid[p_idx])
        p_weight = jnp.clip(p_weight, 0.0, 1.0)

        def _interp_one_species(sigma_4d: jnp.ndarray, log_temp_grid: jnp.ndarray) -> jnp.ndarray:
            t_idx = jnp.searchsorted(log_temp_grid, log_t) - 1
            t_idx = jnp.clip(t_idx, 0, log_temp_grid.shape[0] - 2)
            t_weight = (log_t - log_temp_grid[t_idx]) / (log_temp_grid[t_idx + 1] - log_temp_grid[t_idx])
            t_weight = jnp.clip(t_weight, 0.0, 1.0)

            s_t0_p0 = sigma_4d[t_idx, p_idx, :, :]
            s_t0_p1 = sigma_4d[t_idx, p_idx + 1, :, :]
            s_t1_p0 = sigma_4d[t_idx + 1, p_idx, :, :]
            s_t1_p1 = sigma_4d[t_idx + 1, p_idx + 1, :, :]

            s_t0 = (1.0 - p_weight) * s_t0_p0 + p_weight * s_t0_p1
            s_t1 = (1.0 - p_weight) * s_t1_p0 + p_weight * s_t1_p1
            return (1.0 - t_weight) * s_t0 + t_weight * s_t1

        return jax.vmap(_interp_one_species)(sigma_cube, log_temperature_grids).astype(jnp.float64)

    # Interpolate for all layers - returns (nlay, nspec, nwl, ng) then transpose
    sigma_log_all = jax.vmap(_interp_sigma_log_layer)(log_p_layers, log_t_layers)
    # sigma_log_all has shape (nlay, nspec, nwl, ng), transpose to (nspec, nlay, nwl, ng)
    sigma_log_all = jnp.transpose(sigma_log_all, (1, 0, 2, 3))

    # Convert from log10 to linear space
    sigma_linear = 10.0 ** sigma_log_all

    # Convert to mass opacity (cm² / g)
    # sigma_linear is cross-section (cm² molecule⁻¹)
    # Divide by (mu * amu) to get mass opacity
    sigma_perspecies = sigma_linear / (layer_mu[None, :, None, None] * amu)

    return sigma_perspecies, mixing_ratios
