"""
opacity_ck.py
==============

Overview:
    Correlated-k opacity module for handling pre-banded opacity tables with
    Gauss quadrature integration over g-points.

    Similar to opacity_line.py but includes an extra g-dimension for the
    correlated-k method.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from typing import Dict

import jax
import jax.numpy as jnp
from jax import lax

import build_opacities as XS
from data_constants import amu


def _interpolate_sigma(layer_pressures_bar: jnp.ndarray, layer_temperatures: jnp.ndarray) -> jnp.ndarray:
    """
    Bilinear interpolation of correlated-k cross sections on (log T, log P) grids.

    sigma_cube shape: (n_species, n_temp, n_pressure, n_wavelength, n_g)
    Returns: (n_species, n_layers, n_wavelength, n_g)
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
    p_idx = jnp.clip(p_idx, 0, len(log_p_grid) - 2)
    p_weight = (log_p_layers - log_p_grid[p_idx]) / (log_p_grid[p_idx + 1] - log_p_grid[p_idx])
    p_weight = jnp.clip(p_weight, 0.0, 1.0)

    def _interp_one_species(sigma_4d, temp_grid):
        """Interpolate cross sections for one species."""
        # sigma_4d: (n_temp, n_pressure, n_wavelength, n_g)
        # temp_grid: (n_temp,)

        # Convert temperature grid to log space
        log_t_grid = jnp.log10(temp_grid)

        # Find temperature bracket indices and weights in log space
        t_idx = jnp.searchsorted(log_t_grid, log_t_layers) - 1
        t_idx = jnp.clip(t_idx, 0, len(log_t_grid) - 2)
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
    return 10.0 ** sigma_log


def _get_ck_quadrature(state: Dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return shared g-points and quadrature weights."""
    g_points_all = XS.ck_g_points()
    g_weights_all = state.get("g_weights")
    if g_weights_all is None:
        g_weights_all = XS.ck_g_weights()

    if g_points_all.ndim == 1:
        g_eval = jnp.asarray(g_points_all)
    else:
        g_eval = jnp.asarray(g_points_all[0])

    if g_weights_all.ndim == 1:
        weights = jnp.asarray(g_weights_all)
    else:
        weights = jnp.asarray(g_weights_all[0])

    return g_eval, weights


def _rom_mix_band(
    sigma_stack: jnp.ndarray,
    vmr_layer: jnp.ndarray,
    g_points: jnp.ndarray,
    base_weights: jnp.ndarray,
) -> jnp.ndarray:
    """
    Mix one wavelength band's species using RORR.

    Follows the algorithm from RT_opac.py:mix_k_table_RORR more directly:
    - Initialize with first species
    - Sequentially mix in remaining species
    """
    n_species = sigma_stack.shape[0]
    ng = sigma_stack.shape[-1]

    if n_species == 0:
        return jnp.zeros(ng, dtype=sigma_stack.dtype)

    # Initialize with first species (following original: cs_mix = kc_int[k,0,b,:] * VMR_tot)
    vmr_tot = vmr_layer[0]
    cs_mix = sigma_stack[0] * vmr_tot

    if n_species == 1:
        return cs_mix

    # Pre-compute the ROM weight matrix (same for all species since g-weights are identical)
    rom_weights = jnp.outer(base_weights, base_weights).reshape(-1)

    def body(carry, inputs):
        cs_mix_prev, vmr_tot_prev = carry
        sigma_spec, vmr_spec = inputs

        # Skip mixing if species has negligible cross-section (optimization)
        def skip_species(_):
            # Just update total VMR without mixing
            vmr_tot = vmr_tot_prev + vmr_spec
            cs_mix_new = cs_mix_prev * (vmr_tot / jnp.maximum(vmr_tot_prev, 1e-30))
            return (cs_mix_new, vmr_tot), None

        def mix_species(_):
            # Add to total VMR
            vmr_tot = vmr_tot_prev + vmr_spec

            # Create ROM matrix: k_rom_matrix[i,j] = (cs_mix[i] + vmr*sigma[j]) / vmr_tot
            k_rom_matrix = (cs_mix_prev[:, None] + vmr_spec * sigma_spec[None, :]) / vmr_tot

            # Flatten
            k_rom_flat = k_rom_matrix.ravel()

            # Sort by k-value
            sort_idx = jnp.argsort(k_rom_flat)
            k_rom_sorted = jnp.maximum(k_rom_flat[sort_idx], 1e-99)
            w_rom_sorted = rom_weights[sort_idx]

            # Compute cumulative g
            g_rom = jnp.cumsum(w_rom_sorted)
            g_rom = g_rom / g_rom[-1]

            # Interpolate to standard g-points
            cs_mix_new = jnp.power(10.0, jnp.interp(g_points, g_rom, jnp.log10(k_rom_sorted))) * vmr_tot

            return (cs_mix_new, vmr_tot), None

        # Skip if max cross-section is negligible (< 1e-50)
        return lax.cond(jnp.max(sigma_spec) < 1e-50, skip_species, mix_species, operand=None)

    # Scan over species 1 onwards
    (cs_mix_final, _), _ = lax.scan(
        body,
        (cs_mix, vmr_tot),
        (sigma_stack[1:], vmr_layer[1:])
    )

    return cs_mix_final


def _mix_k_tables_rorr(
    sigma_values: jnp.ndarray,
    mixing_ratios: jnp.ndarray,
    g_points: jnp.ndarray,
    base_weights: jnp.ndarray,
) -> jnp.ndarray:
    """
    Random overlap (RORR) mixing of correlated-k tables across species.

    sigma_values: (n_species, n_layers, n_wavelength, n_g)
    mixing_ratios: (n_species, n_layers)
    """
    n_species, n_layers, n_wl, n_g = sigma_values.shape
    dtype = sigma_values.dtype
    if n_species == 0:
        return jnp.zeros((n_layers, n_wl, n_g), dtype=dtype)

    if mixing_ratios.ndim == 1:
        mixing_ratios = jnp.broadcast_to(mixing_ratios[:, None], (n_species, n_layers))

    # Reorder and reshape for batched vmap: flatten (layers, wavelength) into single batch dimension
    # sigma_values: (n_species, n_layers, n_wavelength, n_g) -> (n_layers*n_wl, n_species, n_g)
    sigma_batch = jnp.transpose(sigma_values, (1, 2, 0, 3)).reshape(n_layers * n_wl, n_species, n_g)

    # mixing_ratios: (n_species, n_layers) -> (n_layers*n_wl, n_species)
    # Broadcast VMR across wavelengths since it's per-layer
    vmr_batch = jnp.transpose(mixing_ratios, (1, 0))  # (n_layers, n_species)
    vmr_batch = jnp.tile(vmr_batch[:, None, :], (1, n_wl, 1))  # (n_layers, n_wl, n_species)
    vmr_batch = vmr_batch.reshape(n_layers * n_wl, n_species)  # (n_layers*n_wl, n_species)

    # Single vmap over the combined batch dimension
    mixed_batch = jax.vmap(_rom_mix_band, in_axes=(0, 0, None, None))(
        sigma_batch, vmr_batch, g_points, base_weights
    )

    # Reshape back to (n_layers, n_wl, n_g)
    mixed = mixed_batch.reshape(n_layers, n_wl, n_g)
    return mixed


def zero_ck_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]):
    """
    Return zero opacity (placeholder for when ck opacities are not used).

    Args:
        state: State dictionary containing wavelengths and layer pressures
        params: Parameter dictionary

    Returns:
        Zero array of shape (n_layers, n_wavelength, n_g)
    """
    layer_pressures = state["p_lay"]
    wavelengths = state["wl"]
    layer_count = jnp.size(layer_pressures)
    wavelength_count = jnp.size(wavelengths)

    # Get number of g-points from loaded ck data
    g_weights = XS.ck_g_weights()
    n_g = jnp.size(g_weights)

    return jnp.zeros((layer_count, wavelength_count, n_g))


def compute_ck_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]):
    """
    Compute correlated-k opacity (interpolated and normalized, keeping g-dimension).

    Interpolates cross sections at layer conditions, applies normalization for mixing
    ratios, and returns opacity with g-dimension intact for later integration.

    Args:
        state: State dictionary containing:
            - p_lay: Layer pressures (microbar)
            - T_lay: Layer temperatures (K)
            - mu_lay: Mean molecular weight per layer
            - wl: Wavelengths
        params: Parameter dictionary containing mixing ratios (f_{species_name})

    Returns:
        Opacity array of shape (n_layers, n_wavelength, n_g) in cm^2/g
    """
    layer_pressures = state["p_lay"]
    layer_temperatures = state["T_lay"]
    layer_mu = state["mu_lay"]

    # Get species names and mixing ratios
    species_names = XS.ck_species_names()
    layer_vmr = state["vmr_lay"]

    # Direct lookup - species names must match VMR keys exactly
    mixing_arrays = []
    for name in species_names:
        arr = jnp.asarray(layer_vmr[name])
        if arr.ndim == 0:
            arr = jnp.full((layer_pressures.shape[0],), arr)
        mixing_arrays.append(arr)
    mixing_ratios = jnp.stack(mixing_arrays)

    # Interpolate cross sections for all species at layer conditions
    # sigma_values shape: (n_species, n_layers, n_wavelength, n_g)
    sigma_values = _interpolate_sigma(layer_pressures / 1e6, layer_temperatures)

    g_points, g_weights = _get_ck_quadrature(state)

    # Perform random-overlap mixing
    mixed_sigma = _mix_k_tables_rorr(sigma_values, mixing_ratios, g_points, g_weights)

    # Convert to mass opacity (cm^2 / g)
    total_opacity = mixed_sigma / (layer_mu[:, None, None] * amu)

    return total_opacity
