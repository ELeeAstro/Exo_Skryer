"""
vert_alt.py
===========

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp

from data_constants import amu, kb, R_jup, bar


def hypsometric(p_lev, T_lay, mu_lay, params):
    g = jnp.asarray(params["log_10_g"])
    H = (kb * T_lay) / (mu_lay * amu * g)
    dlnp = jnp.log(p_lev[:-1] / p_lev[1:])
    dz = H * dlnp
    z0 = jnp.zeros_like(p_lev[:1])

    z_lev = jnp.concatenate([z0, jnp.cumsum(dz)])
    z_lay = (z_lev[:-1] + z_lev[1:]) / 2.0

    return z_lev, z_lay, dz


def g_at_z(R0, z, g_ref):
    return g_ref * (R0 / (R0 + z)) ** 2

def hypsometric_variable_g(p_lev, T_lay, mu_lay, params):

    nlev = p_lev.shape[0]

    g_ref = 10.0**jnp.asarray(params["log_10_g"])
    R0 = jnp.asarray(params["R_p"]) * R_jup

    dlnp = jnp.log(p_lev[:-1] / p_lev[1:])

    z0 = jnp.zeros_like(p_lev)
    dz0 = jnp.zeros_like(p_lev[:-1])

    def body(i, carry):
        z, dz = carry

        # predictor using g at level i
        g_i = g_at_z(R0, z[i], g_ref)
        H_i = (kb * T_lay[i]) / (mu_lay[i] * amu * g_i)
        dz_pred = H_i * dlnp[i]

        # corrector using g at mid-layer altitude (predicted)
        z_mid = z[i] + 0.5 * dz_pred
        g_mid = g_at_z(R0, z_mid, g_ref)
        H_mid = (kb * T_lay[i]) / (mu_lay[i] * amu * g_mid)
        dz_i = H_mid * dlnp[i]

        z = z.at[i + 1].set(z[i] + dz_i)
        dz = dz.at[i].set(dz_i)
        return (z, dz)

    z_lev, dz = jax.lax.fori_loop(0, nlev - 1, body, (z0, dz0))
    z_lay = 0.5 * (z_lev[:-1] + z_lev[1:])
    return z_lev, z_lay, dz


def hypsometric_variable_g_pref(p_lev, T_lay, mu_lay, params):
    """
    Hypsometric integration that anchors R0 and g_ref at an arbitrary
    reference pressure p_ref (between p_bot and p_top).
    """
    g_ref = 10.0**jnp.asarray(params["log_10_g"])
    R0 = jnp.asarray(params["R_p"]) * R_jup
    p_ref = 10.0**jnp.asarray(params["log_10_p_ref"]) * bar

    nlev = p_lev.shape[0]
    dlnp = jnp.log(p_lev[:-1] / p_lev[1:])

    # Ensure p_ref lies within the grid bounds
    p_ref = jnp.clip(p_ref, p_lev[-1], p_lev[0])

    # Locate the layer whose bounds encompass p_ref
    mask = p_lev >= p_ref
    ref_layer = jnp.sum(mask.astype(jnp.int32)) - 1
    ref_layer = jnp.clip(ref_layer, 0, nlev - 2)

    z_lev = jnp.zeros_like(p_lev)

    def integrate_segment(layer_idx, z_start, delta_ln, direction):
        T = T_lay[layer_idx]
        mu = mu_lay[layer_idx]
        g_i = g_at_z(R0, z_start, g_ref)
        H_i = (kb * T) / (mu * amu * g_i)
        dz_pred = direction * H_i * delta_ln
        z_mid = z_start + 0.5 * dz_pred
        g_mid = g_at_z(R0, z_mid, g_ref)
        H_mid = (kb * T) / (mu * amu * g_mid)
        dz_val = direction * H_mid * delta_ln
        return z_start + dz_val

    # Partial step from p_ref to the bracketing levels (if needed)
    delta_down = jnp.maximum(jnp.log(p_lev[ref_layer] / p_ref), 0.0)
    z_lower = integrate_segment(ref_layer, 0.0, delta_down, -1.0)
    z_lev = z_lev.at[ref_layer].set(z_lower)

    has_upper = ref_layer + 1 < nlev
    delta_up = jnp.maximum(jnp.log(p_ref / p_lev[ref_layer + 1]), 0.0)

    def set_upper(z_vals):
        z_upper = integrate_segment(ref_layer, 0.0, delta_up, 1.0)
        return z_vals.at[ref_layer + 1].set(z_upper)

    z_lev = jax.lax.cond(has_upper, set_upper, lambda z_vals: z_vals, z_lev)

    # Integrate toward lower pressures (upward in altitude)
    def body_up(i, z_vals):
        use_layer = jnp.logical_and(i >= (ref_layer + 1), i < nlev - 1)

        def update(z_arr):
            z_start = z_arr[i]
            delta = dlnp[i]
            z_next = integrate_segment(i, z_start, delta, 1.0)
            return z_arr.at[i + 1].set(z_next)

        return jax.lax.cond(use_layer, update, lambda z_arr: z_arr, z_vals)

    z_lev = jax.lax.fori_loop(0, nlev - 1, body_up, z_lev)

    # Integrate toward higher pressures (downward in altitude)
    def body_down(i, z_vals):
        idx = ref_layer - 1 - i
        use_layer = idx >= 0

        def update(z_arr):
            z_start = z_arr[idx + 1]
            delta = dlnp[idx]
            z_next = integrate_segment(idx, z_start, delta, -1.0)
            return z_arr.at[idx].set(z_next)

        return jax.lax.cond(use_layer, update, lambda z_arr: z_arr, z_vals)

    z_lev = jax.lax.fori_loop(0, nlev - 1, body_down, z_lev)

    dz = z_lev[1:] - z_lev[:-1]
    z_lay = 0.5 * (z_lev[:-1] + z_lev[1:])
    return z_lev, z_lay, dz
