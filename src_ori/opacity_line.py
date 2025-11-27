"""
opacity_line.py
===============

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from typing import Dict

import jax
import jax.numpy as jnp

import build_opacities as XS
from data_constants import amu

_SIGMA_CACHE: jnp.ndarray | None = None


def _load_sigma_cube() -> jnp.ndarray:
    global _SIGMA_CACHE
    if _SIGMA_CACHE is None:
        _SIGMA_CACHE = XS.line_sigma_cube()
    return _SIGMA_CACHE


def _interpolate_sigma(layer_pressures_bar: jnp.ndarray, layer_temperatures: jnp.ndarray) -> jnp.ndarray:
    """
    Bilinear interpolation of cross sections on (log T, log P) grids.

    sigma_cube shape: (n_species, n_temp, n_pressure, n_wavelength)
    Returns: (n_species, n_layers, n_wavelength)
    """
    sigma_cube = _load_sigma_cube()
    pressure_grid = XS.line_pressure_grid()
    temperature_grids = XS.line_temperature_grids()

    # Convert to log10 space for interpolation
    log_p_grid = jnp.log10(pressure_grid)
    log_p_layers = jnp.log10(layer_pressures_bar)
    log_t_layers = jnp.log10(layer_temperatures)

    # Find pressure bracket indices and weights in log space (same for all species)
    p_idx = jnp.searchsorted(log_p_grid, log_p_layers) - 1
    p_idx = jnp.clip(p_idx, 0, len(log_p_grid) - 2)
    p_weight = (log_p_layers - log_p_grid[p_idx]) / (log_p_grid[p_idx + 1] - log_p_grid[p_idx])
    p_weight = jnp.clip(p_weight, 0.0, 1.0)

    def _interp_one_species(sigma_3d, temp_grid):
        """Interpolate cross sections for one species."""
        # sigma_3d: (n_temp, n_pressure, n_wavelength)
        # temp_grid: (n_temp,)

        # Convert temperature grid to log space
        log_t_grid = jnp.log10(temp_grid)

        # Find temperature bracket indices and weights in log space
        t_idx = jnp.searchsorted(log_t_grid, log_t_layers) - 1
        t_idx = jnp.clip(t_idx, 0, len(log_t_grid) - 2)
        t_weight = (log_t_layers - log_t_grid[t_idx]) / (log_t_grid[t_idx + 1] - log_t_grid[t_idx])
        t_weight = jnp.clip(t_weight, 0.0, 1.0)

        # Get four corners of bilinear interpolation rectangle
        # Indexing: sigma_3d[temp, pressure, wavelength]
        s_t0_p0 = sigma_3d[t_idx, p_idx, :]              # shape: (n_layers, n_wavelength)
        s_t0_p1 = sigma_3d[t_idx, p_idx + 1, :]
        s_t1_p0 = sigma_3d[t_idx + 1, p_idx, :]
        s_t1_p1 = sigma_3d[t_idx + 1, p_idx + 1, :]

        # Bilinear interpolation: first interpolate in pressure, then temperature
        s_t0 = (1.0 - p_weight)[:, None] * s_t0_p0 + p_weight[:, None] * s_t0_p1
        s_t1 = (1.0 - p_weight)[:, None] * s_t1_p0 + p_weight[:, None] * s_t1_p1
        s_interp = (1.0 - t_weight)[:, None] * s_t0 + t_weight[:, None] * s_t1

        return s_interp

    # Vectorize over all species
    sigma_log = jax.vmap(_interp_one_species)(sigma_cube, temperature_grids)
    return 10.0 ** sigma_log


def zero_line_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]):
    layer_pressures = state["p_lay"]
    wavelengths = state["wl"]
    layer_count = jnp.size(layer_pressures)
    wavelength_count = jnp.size(wavelengths)
    return jnp.zeros((layer_count, wavelength_count))


def compute_line_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]):
    """
    Compute line-by-line opacity.

    Args:
        state: State dictionary containing:
            - p_lay: Layer pressures (microbar)
            - T_lay: Layer temperatures (K)
            - mu_lay: Mean molecular weight per layer
            - vmr_lay (optional): VMR dictionary indexed by species name
        params: Parameter dictionary (fallback for VMR if not in state)

    Returns:
        Opacity array of shape (n_layers, n_wavelength) in cm^2/g
    """
    layer_pressures = state["p_lay"]
    layer_temperatures = state["T_lay"]
    layer_mu = state["mu_lay"]
    layer_vmr = state["vmr_lay"]

    # Get species names and mixing ratios
    species_names = XS.line_species_names()

    mixing_arrays = []
    for name in species_names:
        # Try direct lookup first (if vmr dict), then fall back to f_ prefix
        if name in layer_vmr:
            arr = jnp.asarray(layer_vmr[name])
        else:
            arr = jnp.asarray(layer_vmr[f"f_{name}"])

        if arr.ndim == 0:
            arr = jnp.full((layer_pressures.shape[0],), arr)
        mixing_arrays.append(arr)
    mixing_ratios = jnp.stack(mixing_arrays)

    # Interpolate cross sections for all species at layer conditions
    # sigma_values shape: (n_species, n_layers, n_wavelength)
    sigma_values = _interpolate_sigma(layer_pressures / 1e6, layer_temperatures)

    # Compute mass opacity normalization
    normalization = mixing_ratios / (layer_mu[None, :] * amu)

    # Sum over species: (n_species, n_layers, n_wavelength) -> (n_layers, n_wavelength)
    return jnp.sum(sigma_values * normalization[:, :, None], axis=0)
