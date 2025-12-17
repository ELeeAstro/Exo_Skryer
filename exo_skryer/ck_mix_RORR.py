"""
ck_mix_RORR.py
==============
"""

import jax
import jax.numpy as jnp
from jax import lax


__all__ = ['mix_k_tables_rorr']

def _rom_mix_band(
    sigma_stack: jnp.ndarray,
    vmr_layer: jnp.ndarray,
    g_points: jnp.ndarray,
    base_weights: jnp.ndarray,
    rom_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Mix multiple species at one (layer, wavelength) using RORR.

    Implements random overlap with ranked reordering (RORR): sequentially add
    species by forming the ROM matrix, sorting by k, and interpolating back to
    the standard g-grid.

    Parameters
    ----------
    sigma_stack : `~jax.numpy.ndarray`
        Cross-sections for all species at this (layer, wavelength),
        shape `(n_species, n_g)`.
    vmr_layer : `~jax.numpy.ndarray`
        Volume mixing ratios for each species at this layer, shape `(n_species,)`.
    g_points : `~jax.numpy.ndarray`
        Standard g-points for interpolation, shape `(n_g,)`.
    base_weights : `~jax.numpy.ndarray`
        Quadrature weights for each g-point, shape `(n_g,)`.
    rom_weights : `~jax.numpy.ndarray`
        Pre-computed ROM weights (outer product of `base_weights`), shape `(n_g**2,)`.

    Returns
    -------
    `~jax.numpy.ndarray`
        Mixed cross-section at this (layer, wavelength), shape `(n_g,)`.
    """
    n_species = sigma_stack.shape[0]
    ng = sigma_stack.shape[-1]

    if n_species == 0:
        return jnp.zeros(ng, dtype=sigma_stack.dtype)

    # Initialize with first species
    vmr_tot = vmr_layer[0]
    cs_mix = sigma_stack[0] * vmr_tot

    if n_species == 1:
        return cs_mix

    # NOTE: rom_weights is pre-computed outside and passed in as parameter

    def body(carry, inputs):
        cs_mix_prev, vmr_tot_prev = carry
        sigma_spec, vmr_spec = inputs

        def skip_species(_):
            """Skip species with negligible cross-section."""
            vmr_tot = vmr_tot_prev + vmr_spec
            return (cs_mix_prev, vmr_tot), None

        def mix_species(_):
            """Perform RORR mixing for this species."""
            vmr_tot = vmr_tot_prev + vmr_spec

            # Create ROM matrix: k_rom_matrix[i,j] = (cs_mix[i] + vmr*sigma[j]) / vmr_tot
            k_rom_matrix = (cs_mix_prev[:, None] + vmr_spec * sigma_spec[None, :]) / vmr_tot

            # Flatten
            k_rom_flat = k_rom_matrix.ravel()

            # OPTIMIZATION: Sort pairs directly instead of argsort + fancy indexing
            k_rom_sorted, w_rom_sorted = lax.sort_key_val(k_rom_flat, rom_weights)
            k_rom_sorted = jnp.clip(k_rom_sorted, 1e-99, None)

            # Compute cumulative g with optimized normalization
            w_cumsum = jnp.cumsum(w_rom_sorted)
            g_rom = w_cumsum / w_cumsum[-1]

            # OPTIMIZATION: Cleaner log/exp operations using 10** for clarity
            log_k_interp = jnp.interp(g_points, g_rom, jnp.log10(k_rom_sorted))
            cs_mix_new = vmr_tot * (10.0 ** log_k_interp)

            return (cs_mix_new, vmr_tot), None

        # Skip if max cross-section is negligible (< 1e-50)
        return lax.cond(
            jnp.max(sigma_spec) < 1e-50,
            skip_species,
            mix_species,
            operand=None
        )

    # Scan over species 1 onwards
    (cs_mix_final, _), _ = lax.scan(
        body,
        (cs_mix, vmr_tot),
        (sigma_stack[1:], vmr_layer[1:])
    )

    return cs_mix_final


def mix_k_tables_rorr(
    sigma_values: jnp.ndarray,
    mixing_ratios: jnp.ndarray,
    g_points: jnp.ndarray,
    base_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Mix correlated-k tables across species using RORR.

    Parameters
    ----------
    sigma_values : `~jax.numpy.ndarray`
        Cross-sections for all species, shape `(n_species, n_layers, n_wavelength, n_g)`.
    mixing_ratios : `~jax.numpy.ndarray`
        Volume mixing ratios, shape `(n_species, n_layers)` or `(n_species,)`.
        If 1D, it is broadcast across layers.
    g_points : `~jax.numpy.ndarray`
        Standard g-points for interpolation, shape `(n_g,)`.
    base_weights : `~jax.numpy.ndarray`
        Quadrature weights, shape `(n_g,)`.

    Returns
    -------
    `~jax.numpy.ndarray`
        Mixed cross-sections, shape `(n_layers, n_wavelength, n_g)`.
    """
    n_species, n_layers, n_wl, n_g = sigma_values.shape
    dtype = sigma_values.dtype

    if n_species == 0:
        return jnp.zeros((n_layers, n_wl, n_g), dtype=dtype)

    if mixing_ratios.ndim == 1:
        mixing_ratios = jnp.broadcast_to(mixing_ratios[:, None], (n_species, n_layers))

    # OPTIMIZATION: Pre-compute ROM weights ONCE (not n_layers * n_wl times!)
    rom_weights = jnp.outer(base_weights, base_weights).reshape(-1)

    # sigma_values: (n_species, n_layers, n_wavelength, n_g) -> (n_layers, n_wl, n_species, n_g)
    sigma_reordered = sigma_values.transpose(1, 2, 0, 3)
    vmr_layers = mixing_ratios.T  # (n_layers, n_species)

    def _mix_one_layer(sigma_layer: jnp.ndarray, vmr_layer: jnp.ndarray) -> jnp.ndarray:
        # sigma_layer: (n_wl, n_species, n_g) -> vmap over wavelengths
        return jax.vmap(_rom_mix_band, in_axes=(0, None, None, None, None))(
            sigma_layer, vmr_layer, g_points, base_weights, rom_weights
        )

    return jax.vmap(_mix_one_layer, in_axes=(0, 0))(sigma_reordered, vmr_layers)
