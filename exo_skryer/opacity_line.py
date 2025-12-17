"""
opacity_line.py
===============

Overview:
    Line (molecular/atomic) absorption opacity contribution for the forward
    model.

    This module interpolates preloaded line-by-line cross-sections from
    `build_opacities` onto the layer (P, T) structure and combines them with
    species mixing ratios to produce a mass opacity (cm^2 g^-1) on the forward
    model wavelength grid.

Sections to complete:
    - Usage
    - Key Functions
    - Notes

Usage:
    The retrieval pipeline builds and caches opacity tables via
    `build_opacities`. During forward-model evaluation, pass the shared `state`
    dictionary into `compute_line_opacity(state, params)` to obtain the line
    opacity contribution on the model wavelength grid.

Key Functions:
    - `_interpolate_sigma`: Bilinear interpolation in (log10 T, log10 P) for all
      species, returning cross-sections on the layer grid.
    - `compute_line_opacity`: Combines interpolated cross-sections with mixing
      ratios and mean molecular weight to produce cm^2 g^-1.
    - `zero_line_opacity`: Convenience helper returning an all-zero array with
      the expected shape.

Notes:
    Cross-sections are stored in log10 space in the registry and are converted
    back to linear space after interpolation. Pressure is interpolated in bar;
    `state["p_lay"]` is expected in microbar and is converted internally.
"""

from typing import Dict

import jax
import jax.numpy as jnp

from . import build_opacities as XS
from .data_constants import amu

__all__ = [
    "zero_line_opacity",
    "compute_line_opacity"
]


def _interpolate_sigma(layer_pressures_bar: jnp.ndarray, layer_temperatures: jnp.ndarray) -> jnp.ndarray:
    """Interpolate line cross-sections to layer conditions.

    Performs bilinear interpolation on each species' (log10 T, log10 P) grid.

    Parameters
    ----------
    layer_pressures_bar : `~jax.numpy.ndarray`, shape `(nlay,)`
        Layer pressures in bar.
    layer_temperatures : `~jax.numpy.ndarray`, shape `(nlay,)`
        Layer temperatures in K.

    Returns
    -------
    `~jax.numpy.ndarray`
        Cross-sections in linear space with shape `(nspecies, nlay, nwl)`.
    """
    # Direct access to cached registry data (no redundant caching needed)
    sigma_cube = XS.line_sigma_cube()
    pressure_grid = XS.line_pressure_grid()
    temperature_grids = XS.line_temperature_grids()

    # Convert to log10 space for interpolation
    log_p_grid = jnp.log10(pressure_grid)
    log_p_layers = jnp.log10(layer_pressures_bar)
    log_t_layers = jnp.log10(layer_temperatures)

    # Find pressure bracket indices and weights in log space (same for all species)
    p_idx = jnp.searchsorted(log_p_grid, log_p_layers) - 1
    p_idx = jnp.clip(p_idx, 0, log_p_grid.shape[0] - 2)
    p_weight = (log_p_layers - log_p_grid[p_idx]) / (log_p_grid[p_idx + 1] - log_p_grid[p_idx])
    p_weight = jnp.clip(p_weight, 0.0, 1.0)

    def _interp_one_species(sigma_3d, temp_grid):
        """Interpolate cross-sections for a single species.

        Parameters
        ----------
        sigma_3d : `~jax.numpy.ndarray`
            Log10 cross-sections with shape `(nT, nP, nwl)`.
        temp_grid : `~jax.numpy.ndarray`
            Temperature grid (K) with shape `(nT,)`.

        Returns
        -------
        jnp.ndarray
            Log10 cross-sections interpolated to layers with shape `(nlay, nwl)`.
        """
        # sigma_3d: (n_temp, n_pressure, n_wavelength)
        # temp_grid: (n_temp,)

        # Convert temperature grid to log space
        log_t_grid = jnp.log10(temp_grid)

        # Find temperature bracket indices and weights in log space
        t_idx = jnp.searchsorted(log_t_grid, log_t_layers) - 1
        t_idx = jnp.clip(t_idx, 0, log_t_grid.shape[0] - 2)
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


def zero_line_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Return a zero line-opacity array.

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
    # Use shape directly without jnp.size() for JIT compatibility
    shape = (state["nlay"], state["nwl"])
    return jnp.zeros(shape)


def compute_line_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute line-by-line mass opacity.

    Parameters
    ----------
    state : dict[str, jnp.ndarray]
        State dictionary with at least:

        - `p_lay` : `~jax.numpy.ndarray`, shape `(nlay,)`
            Layer pressures in microbar.
        - `T_lay` : `~jax.numpy.ndarray`, shape `(nlay,)`
            Layer temperatures in K.
        - `mu_lay` : `~jax.numpy.ndarray`, shape `(nlay,)`
            Mean molecular weight per layer (amu).
        - `vmr_lay` : Mapping[str, Any]
            Mapping from species name to volume mixing ratio per layer. Values
            may be a scalar or a length-`nlay` vector.
    params : dict[str, jnp.ndarray]
        Unused; kept for API compatibility.

    Returns
    -------
    `~jax.numpy.ndarray`
        Line absorption opacity in cm^2 g^-1 with shape `(nlay, nwl)`.
    """
    layer_pressures = state["p_lay"]
    layer_temperatures = state["T_lay"]
    layer_mu = state["mu_lay"]
    layer_vmr = state["vmr_lay"]

    # Get species names and mixing ratios
    species_names = XS.line_species_names()
    layer_count = layer_pressures.shape[0]

    # Direct lookup - species names must match VMR keys exactly
    # VMR values are already JAX arrays, no need to wrap
    mixing_ratios = jnp.stack(
        [jnp.broadcast_to(layer_vmr[name], (layer_count,)) for name in species_names],
        axis=0,
    )

    # Interpolate cross sections for all species at layer conditions
    # sigma_values shape: (n_species, n_layers, n_wavelength)
    sigma_values = _interpolate_sigma(layer_pressures / 1e6, layer_temperatures)

    # Sum over species, then apply mean-molecular-weight normalization
    weighted_sigma = jnp.sum(sigma_values * mixing_ratios[:, :, None], axis=0)
    return weighted_sigma / (layer_mu[:, None] * amu)
