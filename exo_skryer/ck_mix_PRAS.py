"""
ck_mix_PRAS.py
=================
"""

import jax
import jax.numpy as jnp

from .aux_funtions import latin_hypercube, pchip_1d

__all__ = [
    "mix_k_tables_pras",
]


def _pras_mix_band(
    sigma_stack_log: jnp.ndarray,
    vmr_layer: jnp.ndarray,
    g_points: jnp.ndarray,
    base_weights: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    del base_weights

    n_samples = g_points.shape[0] * g_points.shape[0]
    n_species = sigma_stack_log.shape[0]

    gs_mat, _ = latin_hypercube(key, n_samples, n_species, scramble=True, dtype=jnp.float32)
    u_by_species = gs_mat.T  # (n_species, n_samples)

    sigma_stack_log = jnp.maximum(sigma_stack_log, -99.0)

    def _eval_one(log_sigma_g: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return pchip_1d(u, g_points, log_sigma_g)

    log_samp = jax.vmap(_eval_one, in_axes=(0, 0))(sigma_stack_log, u_by_species)  # (n_species, n_samples)
    cs_samp = vmr_layer[:, None] * (10.0 ** log_samp)
    cs_tot = jnp.sum(cs_samp, axis=0)
    cs_tot = jnp.maximum(cs_tot, 1e-99)

    cs_sorted = jnp.sort(cs_tot)
    g_mid = (jnp.arange(n_samples, dtype=jnp.float32) + 0.5) / jnp.asarray(n_samples, dtype=jnp.float32)
    return jnp.interp(g_points, g_mid, cs_sorted)


def mix_k_tables_pras(
    sigma_values_log: jnp.ndarray,
    mixing_ratios: jnp.ndarray,
    g_points: jnp.ndarray,
    base_weights: jnp.ndarray,
) -> jnp.ndarray:
    n_species, n_layers, n_wl, n_g = sigma_values_log.shape
    dtype = sigma_values_log.dtype

    if n_species == 0:
        return jnp.zeros((n_layers, n_wl, n_g), dtype=dtype)

    if mixing_ratios.ndim == 1:
        mixing_ratios = jnp.broadcast_to(mixing_ratios[:, None], (n_species, n_layers))

    sigma_reordered = sigma_values_log.transpose(1, 2, 0, 3)  # (n_layers, n_wl, n_species, n_g)
    vmr_layers = mixing_ratios.T  # (n_layers, n_species)
    base_key = jax.random.PRNGKey(0)
    wl_indices = jnp.arange(n_wl)

    def _mix_one_layer(layer_index: jnp.ndarray, sigma_layer: jnp.ndarray, vmr_layer: jnp.ndarray) -> jnp.ndarray:
        key_layer = jax.random.fold_in(base_key, layer_index)

        def _mix_one_wl(wl_index: jnp.ndarray, sigma_band: jnp.ndarray) -> jnp.ndarray:
            key = jax.random.fold_in(key_layer, wl_index)
            return _pras_mix_band(sigma_band, vmr_layer, g_points, base_weights, key)

        return jax.vmap(_mix_one_wl, in_axes=(0, 0))(wl_indices, sigma_layer)

    layer_indices = jnp.arange(n_layers)
    return jax.vmap(_mix_one_layer, in_axes=(0, 0, 0))(layer_indices, sigma_reordered, vmr_layers)
