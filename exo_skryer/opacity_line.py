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
    log_p_grid = XS.line_log10_pressure_grid()
    log_temperature_grids = XS.line_log10_temperature_grids()

    # Direct lookup - species names must match VMR keys exactly
    # VMR values are already JAX arrays, no need to wrap
    mixing_ratios = jnp.stack(
        [jnp.broadcast_to(layer_vmr[name], (layer_count,)) for name in species_names],
        axis=0,
    )

    layer_pressures_bar = layer_pressures / 1e6
    log_p_layers = jnp.log10(layer_pressures_bar)
    log_t_layers = jnp.log10(layer_temperatures)

    p_idx = jnp.searchsorted(log_p_grid, log_p_layers) - 1
    p_idx = jnp.clip(p_idx, 0, log_p_grid.shape[0] - 2)
    p_weight = (log_p_layers - log_p_grid[p_idx]) / (log_p_grid[p_idx + 1] - log_p_grid[p_idx])
    p_weight = jnp.clip(p_weight, 0.0, 1.0)

    def _interp_one_species(sigma_3d, log_temp_grid):
        t_idx = jnp.searchsorted(log_temp_grid, log_t_layers) - 1
        t_idx = jnp.clip(t_idx, 0, log_temp_grid.shape[0] - 2)
        t_weight = (log_t_layers - log_temp_grid[t_idx]) / (log_temp_grid[t_idx + 1] - log_temp_grid[t_idx])
        t_weight = jnp.clip(t_weight, 0.0, 1.0)

        s_t0_p0 = sigma_3d[t_idx, p_idx, :]
        s_t0_p1 = sigma_3d[t_idx, p_idx + 1, :]
        s_t1_p0 = sigma_3d[t_idx + 1, p_idx, :]
        s_t1_p1 = sigma_3d[t_idx + 1, p_idx + 1, :]

        s_t0 = (1.0 - p_weight)[:, None] * s_t0_p0 + p_weight[:, None] * s_t0_p1
        s_t1 = (1.0 - p_weight)[:, None] * s_t1_p0 + p_weight[:, None] * s_t1_p1
        s_interp = (1.0 - t_weight)[:, None] * s_t0 + t_weight[:, None] * s_t1
        return 10.0 ** s_interp.astype(jnp.float64)

    # Vectorize over all species: (n_species, nlay, nwl)
    sigma_interp_all = jax.vmap(_interp_one_species)(sigma_cube, log_temperature_grids)

    # Weight by volume mixing ratios: (n_species, nlay, 1) * (n_species, nlay, nwl)
    weighted_sigma_all = sigma_interp_all * mixing_ratios[:, :, None]

    # Sum over species dimension
    weighted_sigma = jnp.sum(weighted_sigma_all, axis=0)

    return weighted_sigma / (layer_mu[:, None] * amu)
