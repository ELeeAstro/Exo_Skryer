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


_CIA_SIGMA_CACHE: jnp.ndarray | None = None


def _load_cia_sigma() -> jnp.ndarray:
    global _CIA_SIGMA_CACHE
    if _CIA_SIGMA_CACHE is None:
        _CIA_SIGMA_CACHE = XS.cia_sigma_cube()
    return _CIA_SIGMA_CACHE


def _interpolate_sigma(layer_temperatures: jnp.ndarray) -> jnp.ndarray:
    """
    Linear interpolation of CIA cross sections on log T grids.

    sigma_cube shape: (n_species, n_temp, n_wavelength)
    Returns: (n_species, n_layers, n_wavelength)
    """
    sigma_cube = _load_cia_sigma()
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


def _is_hminus(name: str) -> bool:
    s = str(name).strip().lower().replace(" ", "")
    return s in {"h-", "h−", "hminus", "hm", "hminusion"}

def _compute_pair_weight(
    name: str,
    params: Dict[str, jnp.ndarray],
    layer_count: int,
    mixing_ratios: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    # --- special case: H- bound-free continuum uses only f_Hminus ---
    if _is_hminus(name):
        # accept a few common spellings for convenience
        for key in ("f_Hminus", "f_hminus", "f_H-", "f_h-", "f_Hm", "f_hm"):
            if key in params:
                w = jnp.asarray(params[key])
                return jnp.broadcast_to(w, (layer_count,))
        raise KeyError(
            "Missing H- abundance parameter. Provide one of: "
            "params['f_Hminus'] (preferred), or params['f_H-'], params['f_Hm']."
        )

    # --- normal CIA: requires 'A-B' and uses product of mixing ratios ---
    parts = name.split("-")
    if len(parts) != 2 or (parts[0] == "") or (parts[1] == ""):
        raise ValueError(f"CIA species name '{name}' must be of form 'A-B' (or 'H-' special case).")
    species_a, species_b = parts

    def _resolve_ratio(species: str) -> jnp.ndarray:
        if species in mixing_ratios:
            return jnp.asarray(mixing_ratios[species])
        key = f"f_{species}"
        if key in params:
            return jnp.asarray(params[key])
        raise KeyError(f"Missing CIA mixing parameter for '{key}'")

    ratio_a = jnp.broadcast_to(_resolve_ratio(species_a), (layer_count,))
    ratio_b = jnp.broadcast_to(_resolve_ratio(species_b), (layer_count,))
    return ratio_a * ratio_b



def compute_cia_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    layer_count = int(state["nlay"])
    wavelengths = state["wl"]
    layer_temperatures = state["T_lay"]
    number_density = state["nd"]   # (nlay,)
    density = state["rho"]         # (nlay,)

    master_wavelength = XS.cia_master_wavelength()
    if master_wavelength.shape != wavelengths.shape:
        raise ValueError("CIA wavelength grid must match the forward-model master grid.")

    sigma_values = _interpolate_sigma(layer_temperatures)  # (nspecies, nlay, nwl) presumably
    species_names = XS.cia_species_names()
    mixing_ratios = state.get("mixing_ratios", {})

    pair_weights = jnp.stack(
        [_compute_pair_weight(name, params, layer_count, mixing_ratios) for name in species_names],
        axis=0,
    )  # (nspecies, nlay)

    density = jnp.where(density == 0.0, jnp.inf, density)

    # - CIA pairs:   nd^2 / rho
    # - H-:          nd / rho
    is_hm = jnp.asarray([_is_hminus(n) for n in species_names])[:, None]  # (nspecies, 1) bool

    norm_cia = (number_density ** 2 / density)[None, :]   # (1, nlay)
    norm_hm  = (number_density / density)[None, :]        # (1, nlay)

    normalization = pair_weights * jnp.where(is_hm, norm_hm, norm_cia)  # (nspecies, nlay)

    return jnp.sum(normalization[:, :, None] * sigma_values, axis=0)  # (nlay, nwl)
