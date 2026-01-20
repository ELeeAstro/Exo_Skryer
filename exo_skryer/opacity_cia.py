"""
opacity_cia.py
==============
"""

from typing import Dict

import jax.numpy as jnp

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

    species_names = XS.cia_species_names()
    keep_indices = [i for i, name in enumerate(species_names) if name.strip() != "H-"]
    if not keep_indices:
        return zero_cia_opacity(state, params)

    sigma_cube = XS.cia_sigma_cube()  # (nspecies, nT, nwl) in log10
    log_temperature_grids = XS.cia_log10_temperature_grids()
    temperature_grids = XS.cia_temperature_grids()
    log_t_layers = jnp.log10(layer_temperatures)

    weights_nd2_over_rho = (number_density**2 / density)  # (nlay,)
    out = jnp.zeros((layer_count, wavelengths.shape[0]), dtype=jnp.float64)

    # Accumulate species one-by-one to avoid materializing (n_species, nlay, nwl).
    # Use a Python loop so CIA pair parsing and vmr dict lookups happen at trace time.
    for spec_idx in keep_indices:
        sigma_log_table = sigma_cube[spec_idx]               # (nT, nwl)
        log_temp_grid = log_temperature_grids[spec_idx]      # (nT,)
        temp_grid = temperature_grids[spec_idx]              # (nT,)

        t_idx = jnp.searchsorted(log_temp_grid, log_t_layers) - 1
        t_idx = jnp.clip(t_idx, 0, log_temp_grid.shape[0] - 2)
        t_weight = (log_t_layers - log_temp_grid[t_idx]) / (log_temp_grid[t_idx + 1] - log_temp_grid[t_idx])
        t_weight = jnp.clip(t_weight, 0.0, 1.0)

        s_t0 = sigma_log_table[t_idx, :]
        s_t1 = sigma_log_table[t_idx + 1, :]
        s_interp = (1.0 - t_weight)[:, None] * s_t0 + t_weight[:, None] * s_t1

        min_temp = temp_grid[0]
        below_min = layer_temperatures < min_temp
        tiny = jnp.array(-199.0, dtype=s_interp.dtype)
        s_interp = jnp.where(below_min[:, None], tiny, s_interp)

        sigma_val = 10.0 ** s_interp.astype(jnp.float64)  # (nlay, nwl)
        pair_weight = _compute_pair_weight(species_names[spec_idx], layer_count, layer_vmr)  # (nlay,)
        normalization = pair_weight * weights_nd2_over_rho  # (nlay,)
        out = out + normalization[:, None] * sigma_val

    return out
