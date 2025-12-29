"""
opacity_line.py
===============
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
    """Bilinear interpolation of line-by-line cross-sections on (log P, log T) grids.

    This function retrieves pre-loaded line-by-line absorption cross-section tables
    from the opacity registry and interpolates them to the specified atmospheric layer
    conditions using bilinear interpolation in log₁₀(P)-log₁₀(T) space. The interpolation
    is performed separately for each species and returns cross-sections in linear space.

    Parameters
    ----------
    layer_pressures_bar : `~jax.numpy.ndarray`, shape (nlay,)
        Atmospheric layer pressures in bar.
    layer_temperatures : `~jax.numpy.ndarray`, shape (nlay,)
        Atmospheric layer temperatures in Kelvin.

    Returns
    -------
    sigma_interp : `~jax.numpy.ndarray`, shape (nspecies, nlay, nwl)
        Interpolated absorption cross-sections in linear space with units of
        cm² molecule⁻¹. The axes represent:
        - nspecies: Number of absorbing species
        - nlay: Number of atmospheric layers
        - nwl: Number of wavelength points

    Notes
    -----
    The bilinear interpolation algorithm:
    1. Convert layer pressures and temperatures to log₁₀ space
    2. For each layer, find the bracketing (P, T) grid indices
    3. Compute interpolation weights in each dimension
    4. For each species, interpolate the four corners: (T₀,P₀), (T₀,P₁), (T₁,P₀), (T₁,P₁)
    5. First interpolate in pressure at fixed T₀ and T₁, then interpolate in temperature
    6. Convert from log₁₀ space back to linear space: σ = 10^(log₁₀σ_interp)

    Cross-sections are stored in log₁₀ space in the registry to maintain numerical
    stability across many orders of magnitude, then converted back to linear space
    after interpolation.
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
        sigma_3d : `~jax.numpy.ndarray`, shape (nT, nP, nwl)
            Cross-sections for one species in log₁₀ space.
        temp_grid : `~jax.numpy.ndarray`, shape (nT,)
            Temperature grid for this species in Kelvin.

        Returns
        -------
        s_interp : `~jax.numpy.ndarray`, shape (nlay, nwl)
            Interpolated cross-sections in log₁₀ space.
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
    """Return a zero line-by-line opacity array.

    This function is used as a fallback when line-by-line opacities are disabled
    in the configuration. It maintains API compatibility with `compute_line_opacity()`
    so the forward model can seamlessly switch between LBL enabled/disabled.

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
        Zero-valued line opacity array in cm² g⁻¹.

    Notes
    -----
    This function maintains API compatibility with other opacity calculation
    functions so that the forward model can seamlessly switch between different
    opacity schemes without changing function signatures.
    """
    # Use shape directly without jnp.size() for JIT compatibility
    shape = (state["nlay"], state["nwl"])
    return jnp.zeros(shape)


def compute_line_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute line-by-line mass opacity for all molecular/atomic absorbers.

    This function calculates the total line absorption opacity by:
    1. Interpolating pre-loaded cross-sections to atmospheric (P, T) conditions
    2. Weighting each species' opacity by its volume mixing ratio
    3. Summing contributions from all species
    4. Converting from molecular cross-section to mass opacity

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        Atmospheric state dictionary containing:

        - `p_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer pressures in microbar.
        - `T_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer temperatures in Kelvin.
        - `mu_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Mean molecular weight per layer in g mol^-1.
        - `vmr_lay` : dict[str, `~jax.numpy.ndarray`]
            Volume mixing ratios for each species. Keys must match species
            names in the loaded line opacity tables. Values can be scalars
            or arrays with shape (nlay,).

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary (unused; VMRs come from state['vmr_lay']).
        Kept for API compatibility with other opacity functions.

    Returns
    -------
    kappa_line : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Total line absorption mass opacity in cm² g⁻¹ at each layer and
        wavelength point.

    Notes
    -----
    The opacity formula is:

        κ_line(λ) = Σ_i [f_i × σ_i(λ, P, T)] / μ

    where:
    - f_i is the volume mixing ratio of species i
    - σ_i is the absorption cross-section in cm² molecule⁻¹
    - μ is the mean molecular weight in amu

    The pressure input to this function should be in microbar to match the forward
    model convention, but is converted to bar internally for table lookup.

    Cross-sections are stored in log₁₀ space in the registry to handle values
    spanning many orders of magnitude, then converted back to linear space after
    interpolation.

    See Also
    --------
    zero_line_opacity : Returns zero opacity when LBL is disabled
    build_opacities : Pre-loads and caches line-by-line cross-section tables
    """
    layer_pressures = state["p_lay"]
    layer_temperatures = state["T_lay"]
    layer_mu = state["mu_lay"]
    layer_vmr = state["vmr_lay"]

    # Get species names and mixing ratios
    species_names = XS.line_species_names()
    layer_count = layer_pressures.shape[0]
    sigma_cube = XS.line_sigma_cube()
    pressure_grid = XS.line_pressure_grid()
    temperature_grids = XS.line_temperature_grids()

    # Direct lookup - species names must match VMR keys exactly
    # VMR values are already JAX arrays, no need to wrap
    mixing_ratios = jnp.stack(
        [jnp.broadcast_to(layer_vmr[name], (layer_count,)) for name in species_names],
        axis=0,
    )

    layer_pressures_bar = layer_pressures / 1e6
    log_p_grid = jnp.log10(pressure_grid)
    log_p_layers = jnp.log10(layer_pressures_bar)
    log_t_layers = jnp.log10(layer_temperatures)

    p_idx = jnp.searchsorted(log_p_grid, log_p_layers) - 1
    p_idx = jnp.clip(p_idx, 0, log_p_grid.shape[0] - 2)
    p_weight = (log_p_layers - log_p_grid[p_idx]) / (log_p_grid[p_idx + 1] - log_p_grid[p_idx])
    p_weight = jnp.clip(p_weight, 0.0, 1.0)

    def _interp_one_species(sigma_3d, temp_grid):
        log_t_grid = jnp.log10(temp_grid)
        t_idx = jnp.searchsorted(log_t_grid, log_t_layers) - 1
        t_idx = jnp.clip(t_idx, 0, log_t_grid.shape[0] - 2)
        t_weight = (log_t_layers - log_t_grid[t_idx]) / (log_t_grid[t_idx + 1] - log_t_grid[t_idx])
        t_weight = jnp.clip(t_weight, 0.0, 1.0)

        s_t0_p0 = sigma_3d[t_idx, p_idx, :]
        s_t0_p1 = sigma_3d[t_idx, p_idx + 1, :]
        s_t1_p0 = sigma_3d[t_idx + 1, p_idx, :]
        s_t1_p1 = sigma_3d[t_idx + 1, p_idx + 1, :]

        s_t0 = (1.0 - p_weight)[:, None] * s_t0_p0 + p_weight[:, None] * s_t0_p1
        s_t1 = (1.0 - p_weight)[:, None] * s_t1_p0 + p_weight[:, None] * s_t1_p1
        s_interp = (1.0 - t_weight)[:, None] * s_t0 + t_weight[:, None] * s_t1
        return 10.0 ** s_interp

    # Vectorize over all species: (n_species, nlay, nwl)
    sigma_interp_all = jax.vmap(_interp_one_species)(sigma_cube, temperature_grids)

    # Weight by volume mixing ratios: (n_species, nlay, 1) * (n_species, nlay, nwl)
    weighted_sigma_all = sigma_interp_all * mixing_ratios[:, :, None]

    # Sum over species dimension
    weighted_sigma = jnp.sum(weighted_sigma_all, axis=0)

    return weighted_sigma / (layer_mu[:, None] * amu)
