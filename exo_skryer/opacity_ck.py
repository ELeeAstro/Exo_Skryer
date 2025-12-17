"""Correlated-k opacity with species mixing."""

from typing import Dict

import jax
import jax.numpy as jnp
from jax import lax

from . import build_opacities as XS
from .data_constants import amu
from .ck_mix_RORR import mix_k_tables_rorr
from .ck_mix_PRAS import mix_k_tables_pras

__all__ = [
    "zero_ck_opacity",
    "compute_ck_opacity"
]


def _interpolate_sigma_log(layer_pressures_bar: jnp.ndarray, layer_temperatures: jnp.ndarray) -> jnp.ndarray:
    """Bilinear interpolation of correlated-k cross sections on (log T, log P) grids.

    Parameters
    ----------
    layer_pressures_bar : `~jax.numpy.ndarray`
        Atmospheric layer pressures in bar, shape (n_layers,)
    layer_temperatures : `~jax.numpy.ndarray`
        Atmospheric layer temperatures in K, shape (n_layers,)

    Returns
    -------
    `~jax.numpy.ndarray`
        Interpolated cross-sections in cm^2/molecule
        Shape: (n_species, n_layers, n_wavelength, n_g)
    """
    sigma_cube = XS.ck_sigma_cube()
    pressure_grid = XS.ck_pressure_grid()
    temperature_grids = XS.ck_temperature_grids()

    # Convert to log10 space for interpolation
    log_p_grid = jnp.log10(pressure_grid)
    log_p_layers = jnp.log10(layer_pressures_bar)
    log_t_layers = jnp.log10(layer_temperatures)

    # Find pressure bracket indices and weights in log space (same for all species)
    p_idx = jnp.searchsorted(log_p_grid, log_p_layers) - 1
    p_idx = jnp.clip(p_idx, 0, log_p_grid.shape[0] - 2)
    p_weight = (log_p_layers - log_p_grid[p_idx]) / (log_p_grid[p_idx + 1] - log_p_grid[p_idx])
    p_weight = jnp.clip(p_weight, 0.0, 1.0)

    def _interp_one_species(sigma_4d, temp_grid):
        """Interpolate cross sections for a single species.

        Parameters
        ----------
        sigma_4d : `~jax.numpy.ndarray`
            Cross-sections for one species in log10 space
            Shape: (n_temp, n_pressure, n_wavelength, n_g)
        temp_grid : `~jax.numpy.ndarray`
            Temperature grid for this species in K, shape (n_temp,)

        Returns
        -------
        jnp.ndarray
            Interpolated cross-sections in log10 space
            Shape: (n_layers, n_wavelength, n_g)
        """

        # Convert temperature grid to log space
        log_t_grid = jnp.log10(temp_grid)

        # Find temperature bracket indices and weights in log space
        t_idx = jnp.searchsorted(log_t_grid, log_t_layers) - 1
        t_idx = jnp.clip(t_idx, 0, log_t_grid.shape[0] - 2)
        t_weight = (log_t_layers - log_t_grid[t_idx]) / (log_t_grid[t_idx + 1] - log_t_grid[t_idx])
        t_weight = jnp.clip(t_weight, 0.0, 1.0)

        # Get four corners of bilinear interpolation rectangle
        # Indexing: sigma_4d[temp, pressure, wavelength, g]
        s_t0_p0 = sigma_4d[t_idx, p_idx, :, :]              # shape: (n_layers, n_wavelength, n_g)
        s_t0_p1 = sigma_4d[t_idx, p_idx + 1, :, :]
        s_t1_p0 = sigma_4d[t_idx + 1, p_idx, :, :]
        s_t1_p1 = sigma_4d[t_idx + 1, p_idx + 1, :, :]

        # Bilinear interpolation: first interpolate in pressure, then temperature
        # Expand weights to broadcast over wavelength and g dimensions
        s_t0 = (1.0 - p_weight)[:, None, None] * s_t0_p0 + p_weight[:, None, None] * s_t0_p1
        s_t1 = (1.0 - p_weight)[:, None, None] * s_t1_p0 + p_weight[:, None, None] * s_t1_p1
        s_interp = (1.0 - t_weight)[:, None, None] * s_t0 + t_weight[:, None, None] * s_t1

        return s_interp

    # Vectorize over all species
    sigma_log = jax.vmap(_interp_one_species)(sigma_cube, temperature_grids)
    return sigma_log


def _interpolate_sigma(layer_pressures_bar: jnp.ndarray, layer_temperatures: jnp.ndarray) -> jnp.ndarray:
    """Interpolate correlated-k cross-sections and return linear-space values."""
    return 10.0 ** _interpolate_sigma_log(layer_pressures_bar, layer_temperatures)


def _get_ck_quadrature(state: Dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract g-points and quadrature weights for correlated-k integration.

    Parameters
    ----------
    state : dict[str, jnp.ndarray]
        State dictionary, optionally containing pre-loaded 'g_weights'

    Returns
    -------
    g_points : `~jax.numpy.ndarray`
        G-points for quadrature evaluation, shape (n_g,)
    weights : `~jax.numpy.ndarray`
        Quadrature weights for integration, shape (n_g,)
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
    """Return zero opacity (placeholder for when CK opacities are disabled).

    Parameters
    ----------
    state : dict[str, jnp.ndarray]
        State dictionary containing:
        - p_lay: Layer pressures
        - wl: Wavelengths
    params : dict[str, jnp.ndarray]
        Parameter dictionary (unused)

    Returns
    -------
    `~jax.numpy.ndarray`
        Zero array of shape (n_layers, n_wavelength, n_g)
    """
    layer_pressures = state["p_lay"]
    wavelengths = state["wl"]
    layer_count = layer_pressures.shape[0]
    wavelength_count = wavelengths.shape[0]

    # Get number of g-points from loaded ck data
    g_weights = XS.ck_g_weights()
    n_g = g_weights.shape[0]

    return jnp.zeros((layer_count, wavelength_count, n_g))


def compute_ck_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute correlated-k opacity with species mixing.

    Parameters
    ----------
    state : dict[str, jnp.ndarray]
        State dictionary containing:
        - p_lay : Layer pressures (microbar)
        - T_lay : Layer temperatures (K)
        - mu_lay : Mean molecular weight per layer (amu)
        - vmr_lay : Dictionary of volume mixing ratios by species
        - wl : Wavelengths (microns)
        - ck_mix : Mixing method, either 'RORR' or 'PRAS' (optional, defaults to 'RORR')
    params : dict[str, jnp.ndarray]
        Parameter dictionary (currently unused, VMRs come from state['vmr_lay'])

    Returns
    -------
    `~jax.numpy.ndarray`
        Total atmospheric opacity in cm^2/g
        Shape: (n_layers, n_wavelength, n_g)
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

    sigma_log = _interpolate_sigma_log(layer_pressures / 1e6, layer_temperatures)

    g_points, g_weights = _get_ck_quadrature(state)

    # Get mixing method from state (default to RORR).
    # Backwards-compatible: accept either a string ("RORR"/"PRAS") or an int code.
    ck_mix_raw = state.get("ck_mix", 1)
    if isinstance(ck_mix_raw, str):
        ck_mix_code = 2 if ck_mix_raw.upper() == "PRAS" else 1
    else:
        # Avoid int() conversion for JIT compatibility
        ck_mix_code = ck_mix_raw

    # Perform mixing based on selected method
    if ck_mix_code == 2:
        mixed_sigma = mix_k_tables_pras(sigma_log, mixing_ratios, g_points, g_weights)
    else:  # Default to RORR
        mixed_sigma = mix_k_tables_rorr(10.0 ** sigma_log, mixing_ratios, g_points, g_weights)

    # Convert to mass opacity (cm^2 / g)
    total_opacity = mixed_sigma / (layer_mu[:, None, None] * amu)

    return total_opacity
