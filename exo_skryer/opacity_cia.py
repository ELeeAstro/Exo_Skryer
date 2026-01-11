"""
opacity_cia.py
==============
"""

from typing import Dict

import jax.numpy as jnp
import jax

from . import build_opacities as XS

__all__ = [
    "compute_cia_opacity",
    "zero_cia_opacity"
]


def zero_cia_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Return a zero CIA opacity array.

    This function is used as a fallback when no CIA species are enabled or when
    all CIA pairs are filtered out (e.g., H- opacity handled separately).

    Parameters
    ----------
    state : dict[str, jnp.ndarray]
        State dictionary containing:

        - `nlay` : int-like
            Number of atmospheric layers.
        - `nwl` : int-like
            Number of wavelength points.

    params : dict[str, jnp.ndarray]
        Parameter dictionary (unused; kept for API compatibility with other
        opacity calculation functions).

    Returns
    -------
    zeros : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Zero-valued CIA opacity array in cm² g⁻¹.
    """
    # Use shape directly without jnp.size() for JIT compatibility
    shape = (state["nlay"], state["nwl"])
    return jnp.zeros(shape)


def _interpolate_sigma(layer_temperatures: jnp.ndarray) -> jnp.ndarray:
    """Interpolate CIA cross-sections to atmospheric layer temperatures.

    This function retrieves pre-loaded CIA cross-section tables from the opacity
    registry and interpolates them to the specified layer temperatures using
    log-log linear interpolation. The interpolation is performed in log₁₀ space
    for both temperature and cross-section to better capture the exponential
    temperature dependence typical of CIA processes.

    Parameters
    ----------
    layer_temperatures : `~jax.numpy.ndarray`, shape (nlay,)
        Atmospheric layer temperatures in Kelvin.

    Returns
    -------
    sigma_interp : `~jax.numpy.ndarray`, shape (nspecies, nlay, nwl)
        Interpolated CIA cross-sections in linear space with units of
        cm⁵ molecule⁻². The first axis corresponds to different CIA pairs
        (e.g., H2-He, H2-H2), the second to atmospheric layers, and the
        third to wavelength points.
    """
    sigma_cube = XS.cia_sigma_cube()
    log_temperature_grids = XS.cia_log10_temperature_grids()
    temperature_grids = XS.cia_temperature_grids()

    # Convert to log10 space for interpolation
    log_t_layers = jnp.log10(layer_temperatures)

    def _interp_one_species(sigma_2d, log_temp_grid, temp_grid):
        """Interpolate log10 cross-sections for a single CIA species.

        Parameters
        ----------
        sigma_2d : `~jax.numpy.ndarray`, shape (nT, nwl)
            Log10 CIA cross-sections.
        log_temp_grid : `~jax.numpy.ndarray`, shape (nT,)
            Log10 temperature grid.
        temp_grid : `~jax.numpy.ndarray`, shape (nT,)
            Temperature grid in Kelvin (for min temp check).

        Returns
        -------
        s_interp : `~jax.numpy.ndarray`, shape (nlay, nwl)
            Log10 cross-sections interpolated to `layer_temperatures`.
        """
        # Find temperature bracket indices and weights in log space
        t_idx = jnp.searchsorted(log_temp_grid, log_t_layers) - 1
        t_idx = jnp.clip(t_idx, 0, log_temp_grid.shape[0] - 2)
        t_weight = (log_t_layers - log_temp_grid[t_idx]) / (log_temp_grid[t_idx + 1] - log_temp_grid[t_idx])
        t_weight = jnp.clip(t_weight, 0.0, 1.0)

        # Linear interpolation between temperature brackets
        s_t0 = sigma_2d[t_idx, :]
        s_t1 = sigma_2d[t_idx + 1, :]
        s_interp = (1.0 - t_weight)[:, None] * s_t0 + t_weight[:, None] * s_t1

        # Set cross sections to very small value below minimum temperature
        min_temp = temp_grid[0]
        below_min = layer_temperatures < min_temp
        tiny = jnp.array(-199.0, dtype=s_interp.dtype)
        s_interp = jnp.where(below_min[:, None], tiny, s_interp)

        return s_interp

    # Vectorize over all CIA species
    sigma_log = jax.vmap(_interp_one_species)(sigma_cube, log_temperature_grids, temperature_grids)
    return 10.0 ** sigma_log.astype(jnp.float64)


def _compute_pair_weight(
    name: str,
    layer_count: int,
    layer_vmr: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Compute the volume mixing ratio product for a CIA molecular pair.

    This function calculates the per-layer weight f_A × f_B for a collision pair,
    where f_A and f_B are the volume mixing ratios of the two colliding species.
    This weight is essential for computing the CIA opacity contribution since the
    collision rate is proportional to the product of the two species' abundances.

    Parameters
    ----------
    name : str
        CIA pair name in 'A-B' format (e.g., 'H2-He', 'H2-H2').
    layer_count : int
        Number of atmospheric layers.
    layer_vmr : dict[str, `~jax.numpy.ndarray`]
        Mapping from species name to volume mixing ratio values. Each value
        can be a scalar (constant profile) or array with shape (nlay,).

    Returns
    -------
    pair_weight : `~jax.numpy.ndarray`, shape (nlay,)
        Product of volume mixing ratios for the pair: f_A × f_B.
        Scalar VMR values are automatically broadcast to (nlay,).

    Raises
    ------
    ValueError
        If `name` is not in 'A-B' format with exactly two hyphen-separated species.
    """
    name_clean = name.strip()

    # Normal CIA pair: "H2-He" -> product of H2 and He VMRs
    parts = name_clean.split("-")
    if len(parts) != 2:
        raise ValueError(f"CIA species '{name}' must be in 'A-B' format")

    species_a, species_b = parts[0], parts[1]
    # VMR values are already JAX arrays, no need to wrap
    ratio_a = layer_vmr[species_a]
    ratio_b = layer_vmr[species_b]
    ratio_a = jnp.broadcast_to(ratio_a, (layer_count,))
    ratio_b = jnp.broadcast_to(ratio_b, (layer_count,))
    return ratio_a * ratio_b


def compute_cia_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute collision-induced absorption (CIA) mass opacity for all molecular pairs.

    This function calculates the total CIA opacity by summing contributions from
    all enabled molecular pairs (e.g., H2-He, H2-H2). For each pair, it:
    1. Interpolates pre-loaded cross-sections to layer temperatures
    2. Computes the VMR pair product (f_A × f_B)
    3. Applies the opacity formula: κ = f_A × f_B × (n_d)² / ρ × σ(λ, T)
    4. Sums over all pairs to get total CIA opacity

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        Atmospheric state dictionary containing:

        - `nlay` : int
            Number of atmospheric layers.
        - `nwl` : int
            Number of wavelength points.
        - `wl` : `~jax.numpy.ndarray`, shape (nwl,)
            Wavelength grid in microns (must match CIA table wavelengths).
        - `T_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer temperatures in Kelvin.
        - `nd_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer number density in molecule cm⁻³.
        - `rho_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer mass density in g cm⁻³.
        - `vmr_lay` : dict[str, `~jax.numpy.ndarray`]
            Volume mixing ratios for each species. Values can be scalars
            or arrays with shape (nlay,).

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary (unused; kept for API compatibility with other
        opacity functions that may depend on retrieval parameters).

    Returns
    -------
    kappa_cia : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Total CIA mass opacity in cm² g⁻¹, summed over all molecular pairs.
        Returns zeros if no CIA pairs are enabled.

    Raises
    ------
    ValueError
        If the CIA wavelength grid does not match the forward model master grid.
    """
    # Use JAX array directly without int() for JIT compatibility
    layer_count = state["nlay"]
    wavelengths = state["wl"]
    layer_temperatures = state["T_lay"]
    number_density = state["nd_lay"]   # (nlay,)
    density = state["rho_lay"]         # (nlay,)
    layer_vmr = state["vmr_lay"]

    master_wavelength = XS.cia_master_wavelength()
    if master_wavelength.shape != wavelengths.shape:
        raise ValueError("CIA wavelength grid must match the forward-model master grid.")

    sigma_values = _interpolate_sigma(layer_temperatures)  # (nspecies, nlay, nwl)
    species_names = XS.cia_species_names()
    keep_indices = [i for i, name in enumerate(species_names) if name.strip() != "H-"]
    if not keep_indices:
        return zero_cia_opacity(state, params)

    # Compute pair weights for each CIA species (string ops happen once at trace time)
    pair_weights = jnp.stack(
        [_compute_pair_weight(species_names[i], layer_count, layer_vmr) for i in keep_indices],
        axis=0,
    )  # (nspecies_keep, nlay)

    normalization = pair_weights * (number_density**2 / density)[None, :]  # (nspecies_keep, nlay)
    sigma_keep = sigma_values[keep_indices, :, :]  # (nspecies_keep, nlay, nwl)
    return jnp.sum(normalization[:, :, None] * sigma_keep, axis=0)  # (nlay, nwl)
