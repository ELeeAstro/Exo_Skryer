"""
opacity_cia.py
==============

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from typing import Dict

import jax.numpy as jnp
from jax import vmap

import build_opacities as XS


def zero_cia_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    layer_pressures = state["p_lay"]
    wavelengths = state["wl"]
    layer_count = jnp.size(layer_pressures)
    wavelength_count = jnp.size(wavelengths)
    return jnp.zeros((layer_count, wavelength_count))


def _interpolate_sigma(layer_temperatures: jnp.ndarray) -> jnp.ndarray:
    """
    Linear interpolation of CIA cross sections on log T grids.

    sigma_cube shape: (n_species, n_temp, n_wavelength)
    Returns: (n_species, n_layers, n_wavelength)
    """
    sigma_cube = XS.cia_sigma_cube()
    temperature_grids = XS.cia_temperature_grids()

    # Convert to log10 space for interpolation
    log_t_layers = jnp.log10(layer_temperatures)

    def _interp_one_species(sigma_2d, temp_grid):
        """Interpolate cross sections for one CIA species."""
        # sigma_2d: (n_temp, n_wavelength)
        # temp_grid: (n_temp,)

        # Convert temperature grid to log space
        log_t_grid = jnp.log10(temp_grid)

        # Find temperature bracket indices and weights in log space
        t_idx = jnp.searchsorted(log_t_grid, log_t_layers) - 1
        t_idx = jnp.clip(t_idx, 0, len(log_t_grid) - 2)
        t_weight = (log_t_layers - log_t_grid[t_idx]) / (log_t_grid[t_idx + 1] - log_t_grid[t_idx])
        t_weight = jnp.clip(t_weight, 0.0, 1.0)

        # Get lower and upper temperature brackets
        # Indexing: sigma_2d[temp, wavelength]
        s_t0 = sigma_2d[t_idx, :]          # shape: (n_layers, n_wavelength)
        s_t1 = sigma_2d[t_idx + 1, :]

        # Linear interpolation in temperature
        s_interp = (1.0 - t_weight)[:, None] * s_t0 + t_weight[:, None] * s_t1

        # Set cross sections to very small value below minimum temperature
        min_temp = temp_grid[0]
        below_min = layer_temperatures < min_temp
        tiny = jnp.array(-199.0, dtype=s_interp.dtype)
        s_interp = jnp.where(below_min[:, None], tiny, s_interp)

        return s_interp

    # Vectorize over all CIA species
    sigma_log = vmap(_interp_one_species)(sigma_cube, temperature_grids)
    return 10.0 ** sigma_log


def _compute_pair_weight(
    name: str,
    layer_count: int,
    layer_vmr: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """
    Compute CIA pair weight from VMR dictionary.

    Args:
        name: CIA species name (e.g., "H2-H2", "H2-He", "H-")
        layer_count: Number of atmospheric layers
        layer_vmr: VMR dictionary (species names must match exactly)

    Returns:
        Per-layer weight array of shape (nlay,)
    """
    name_clean = name.strip()

    # Special case: H- bound-free continuum uses single species
    if "-" not in name_clean or name_clean == "H-":
        # Single species (H- bound-free)
        w = jnp.asarray(layer_vmr[name_clean])
        return jnp.broadcast_to(w, (layer_count,))

    # Normal CIA pair: "H2-He" -> product of H2 and He VMRs
    parts = name_clean.split("-")
    if len(parts) != 2:
        raise ValueError(f"CIA species '{name}' must be 'A-B' or 'H-' format")

    species_a, species_b = parts[0], parts[1]
    ratio_a = jnp.asarray(layer_vmr[species_a])
    ratio_b = jnp.asarray(layer_vmr[species_b])
    ratio_a = jnp.broadcast_to(ratio_a, (layer_count,))
    ratio_b = jnp.broadcast_to(ratio_b, (layer_count,))
    return ratio_a * ratio_b



def compute_cia_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Compute collision-induced absorption (CIA) opacity.

    Args:
        state: State dictionary containing:
            - nlay: Number of layers
            - wl: Wavelengths
            - T_lay: Layer temperatures
            - nd_lay: Number density per layer
            - rho_lay: Mass density per layer
            - vmr_lay: VMR dictionary indexed by species name
        params: Parameter dictionary (kept for API compatibility)

    Returns:
        Opacity array of shape (n_layers, n_wavelength) in cm^2/g
    """
    layer_count = int(state["nlay"])
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

    # Compute pair weights for each CIA species (string ops happen once at trace time)
    pair_weights = jnp.stack(
        [_compute_pair_weight(name, layer_count, layer_vmr) for name in species_names],
        axis=0,
    )  # (nspecies, nlay)

    # Determine normalization based on whether species is H- or CIA pair
    # - CIA pairs:   nd^2 / rho
    # - H-:          nd / rho
    is_hm = jnp.asarray([n.strip() == "H-" for n in species_names])[:, None]  # (nspecies, 1) bool

    norm_cia = (number_density ** 2 / density)[None, :]   # (1, nlay)
    norm_hm  = (number_density / density)[None, :]        # (1, nlay)

    normalization = pair_weights * jnp.where(is_hm, norm_hm, norm_cia)  # (nspecies, nlay)

    return jnp.sum(normalization[:, :, None] * sigma_values, axis=0)  # (nlay, nwl)
