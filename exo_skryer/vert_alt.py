"""
[TODO: add documentation]
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from .data_constants import amu, kb, R_jup, bar

__all__ = [
    "hypsometric",
    "g_at_z",
    "hypsometric_variable_g",
    "hypsometric_variable_g_pref"
]


def hypsometric(
    p_lev: jnp.ndarray,
    T_lay: jnp.ndarray,
    mu_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes altitude profile using the hypsometric equation.

    Parameters
    ----------
    p_lev : `~jax.numpy.ndarray`
        Pressure at levels in bars.
    T_lay : `~jax.numpy.ndarray`
        Layer temperatures in Kelvin.
    mu_lay : `~jax.numpy.ndarray`
        Mean molecular weight at layers in amu.
    params : dict
        A dictionary containing the log10 of surface gravity.

    Returns
    -------
    z_lev : `~jax.numpy.ndarray`
        Altitude at levels in cm.
    z_lay : `~jax.numpy.ndarray`
        Altitude at layer midpoints in cm.
    dz : `~jax.numpy.ndarray`
        Layer thickness in cm.
    """
    # Parameter values are already JAX arrays, no need to wrap
    g_ref = 10.0**params["log_10_g"]
    H = (kb * T_lay) / (mu_lay * amu * g_ref)
    dlnp = jnp.log(p_lev[:-1] / p_lev[1:])
    dz = H * dlnp
    z0 = jnp.zeros_like(p_lev[:1])

    z_lev = jnp.concatenate([z0, jnp.cumsum(dz)])
    z_lay = (z_lev[:-1] + z_lev[1:]) / 2.0

    return z_lev, z_lay, dz


def g_at_z(R0: jnp.ndarray, z: jnp.ndarray, g_ref: jnp.ndarray) -> jnp.ndarray:
    """Computes gravity as a function of altitude.

    Parameters
    ----------
    R0 : float
        Reference radius of the planet in cm.
    z : `~jax.numpy.ndarray`
        Altitude above the reference level in cm.
    g_ref : float
        Reference surface gravity at R0 in cm/s^2.

    Returns
    -------
    `~jax.numpy.ndarray`
        Gravity at altitude z in cm/s^2.
    """
    return g_ref * (R0 / (R0 + z)) ** 2


def hypsometric_variable_g(
    p_lev: jnp.ndarray,
    T_lay: jnp.ndarray,
    mu_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes altitude profile with altitude-dependent gravity.

    Parameters
    ----------
    p_lev : `~jax.numpy.ndarray`
        Pressure at levels in bars.
    T_lay : `~jax.numpy.ndarray`
        Layer temperatures in Kelvin.
    mu_lay : `~jax.numpy.ndarray`
        Mean molecular weight at layers in amu.
    params : dict
        A dictionary containing the log10 of surface gravity and planetary radius.

    Returns
    -------
    z_lev : `~jax.numpy.ndarray`
        Altitude at levels in cm.
    z_lay : `~jax.numpy.ndarray`
        Altitude at layer midpoints in cm.
    dz : `~jax.numpy.ndarray`
        Layer thickness in cm.
    """

    # Parameter values are already JAX arrays, no need to wrap
    g_ref = 10.0**params["log_10_g"]
    R0 = params["R_p"] * R_jup

    dlnp = jnp.log(p_lev[:-1] / p_lev[1:])

    def step(z_current, inputs):
        T_i, mu_i, dlnp_i = inputs

        # Predictor using g at current level
        g_i = g_at_z(R0, z_current, g_ref)
        H_i = (kb * T_i) / (mu_i * amu * g_i)
        dz_pred = H_i * dlnp_i

        # Corrector using g at predicted mid-layer altitude
        z_mid = z_current + 0.5 * dz_pred
        g_mid = g_at_z(R0, z_mid, g_ref)
        H_mid = (kb * T_i) / (mu_i * amu * g_mid)
        dz_i = H_mid * dlnp_i

        return z_current + dz_i, dz_i

    z0 = jnp.zeros((), dtype=p_lev.dtype)
    _, dz = jax.lax.scan(step, z0, (T_lay, mu_lay, dlnp))
    z_lev = jnp.concatenate([jnp.zeros((1,), dtype=dz.dtype), jnp.cumsum(dz)])
    z_lay = 0.5 * (z_lev[:-1] + z_lev[1:])
    return z_lev, z_lay, dz


def hypsometric_variable_g_pref(
    p_lev: jnp.ndarray,
    T_lay: jnp.ndarray,
    mu_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes altitude profile with variable gravity anchored at reference pressure.

    Parameters
    ----------
    p_lev : `~jax.numpy.ndarray`
        Pressure at levels in bars.
    T_lay : `~jax.numpy.ndarray`
        Layer temperatures in Kelvin.
    mu_lay : `~jax.numpy.ndarray`
        Mean molecular weight at layers in amu.
    params : dict
        A dictionary containing the log10 of surface gravity, planetary radius, and log10 of reference pressure.

    Returns
    -------
    z_lev : `~jax.numpy.ndarray`
        Altitude at levels in cm.
    z_lay : `~jax.numpy.ndarray`
        Altitude at layer midpoints in cm.
    dz : `~jax.numpy.ndarray`
        Layer thickness in cm.
    """
    # Parameter values are already JAX arrays, no need to wrap
    g_ref = 10.0**params["log_10_g"]
    R0 = params["R_p"] * R_jup
    p_ref = 10.0**params["log_10_p_ref"] * bar

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
        z_start = jnp.asarray(z_start, dtype=p_lev.dtype)
        delta_ln = jnp.asarray(delta_ln, dtype=p_lev.dtype)
        T = T_lay[layer_idx]
        mu = mu_lay[layer_idx]
        g_i = g_at_z(R0, z_start, g_ref)
        H_i = (kb * T) / (mu * amu * g_i)
        direction = jnp.asarray(direction, dtype=z_start.dtype)
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

    delta_up = jnp.maximum(jnp.log(p_ref / p_lev[ref_layer + 1]), 0.0)

    def set_upper(z_vals):
        z_upper = integrate_segment(ref_layer, 0.0, delta_up, 1.0)
        return z_vals.at[ref_layer + 1].set(z_upper)

    z_lev = jax.lax.cond(ref_layer + 1 < nlev, set_upper, lambda z_vals: z_vals, z_lev)

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
