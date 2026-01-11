"""
vert_alt.py
===========
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
    """Compute an altitude profile using the hypsometric equation (constant gravity).

    Parameters
    ----------
    p_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Pressure at layer interfaces (levels). Units are arbitrary as long as
        consistent across the grid (in the forward model this is dyne cm⁻²).
    T_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Layer temperatures in Kelvin.
    mu_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Mean molecular weight per layer in g mol^-1.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `log_10_g` : float
            Log₁₀ surface gravity in cm s⁻².

    Returns
    -------
    z_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Altitude at levels in cm, with `z_lev[0] = 0`.
    z_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Altitude at layer midpoints in cm.
    dz : `~jax.numpy.ndarray`, shape (nlay,)
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
    """Compute gravity as a function of altitude assuming spherical geometry.

    Parameters
    ----------
    R0 : `~jax.numpy.ndarray`
        Reference planetary radius in cm.
    z : `~jax.numpy.ndarray`
        Altitude above the reference level in cm.
    g_ref : `~jax.numpy.ndarray`
        Reference gravity at `R0` in cm s⁻².

    Returns
    -------
    g_z : `~jax.numpy.ndarray`
        Gravity at altitude `z` in cm s⁻².
    """
    return g_ref * (R0 / (R0 + z)) ** 2


def hypsometric_variable_g(
    p_lev: jnp.ndarray,
    T_lay: jnp.ndarray,
    mu_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute an altitude profile with altitude-dependent gravity.

    Parameters
    ----------
    p_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Pressure at layer interfaces (levels), units consistent across the grid.
    T_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Layer temperatures in Kelvin.
    mu_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Mean molecular weight per layer in g mol^-1.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `log_10_g` : float
            Log₁₀ gravity at the reference radius in cm s⁻².
        - `R_p` : float
            Planet radius in Jupiter radii (used to form `R0 = R_p × R_jup`).

    Returns
    -------
    z_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Altitude at levels in cm, with `z_lev[0] = 0`.
    z_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Altitude at layer midpoints in cm.
    dz : `~jax.numpy.ndarray`, shape (nlay,)
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
    params,
):
    g_ref = 10.0**params["log_10_g"]
    R0 = params["R_p"] * R_jup
    p_ref = 10.0**params["log_10_p_ref"] * bar

    nlev = p_lev.shape[0]
    dlnp = jnp.log(p_lev[:-1] / p_lev[1:])  # positive for descending grid

    p_ref = jnp.clip(p_ref, p_lev[-1], p_lev[0])

    # bracket index k such that p[k] >= p_ref >= p[k+1]
    k = jnp.searchsorted(-p_lev, -p_ref, side="right") - 1
    k = jnp.clip(k, 0, nlev - 2)

    def g_at_z(z):
        r = R0 / (R0 + z)
        return g_ref * r * r

    def step_pc(layer_idx, z_start, delta_ln, direction):
        T = jnp.take(T_lay, layer_idx, mode="clip")
        mu = jnp.take(mu_lay, layer_idx, mode="clip")

        g0 = g_at_z(z_start)
        H0 = (kb * T) / (mu * amu * g0)
        dz_pred = direction * H0 * delta_ln

        z_mid = z_start + 0.5 * dz_pred
        g_mid = g_at_z(z_mid)
        H_mid = (kb * T) / (mu * amu * g_mid)
        dz = direction * H_mid * delta_ln
        return z_start + dz

    z_lev = jnp.zeros_like(p_lev)

    # partial steps from p_ref to the bracketing levels
    d_dn = jnp.log(p_lev[k] / p_ref)       # >= 0
    d_up = jnp.log(p_ref / p_lev[k + 1])   # >= 0

    z_lev = z_lev.at[k].set(step_pc(k, 0.0, d_dn, -1.0))
    z_lev = z_lev.at[k + 1].set(step_pc(k, 0.0, d_up, +1.0))

    # upward integration: i = k+1 .. nlev-2 updates level i+1
    def up_body(i, z):
        z_next = step_pc(i, z[i], dlnp[i], +1.0)
        do = i >= (k + 1)
        return z.at[i + 1].set(jnp.where(do, z_next, z[i + 1]))

    z_lev = jax.lax.fori_loop(0, nlev - 1, up_body, z_lev)

    # downward integration: i = k-1 .. 0 updates level i
    def down_body(ii, z):
        i = (k - 1) - ii
        z_next = step_pc(i, z[i + 1], dlnp[i], -1.0)
        do = i >= 0
        i_safe = jnp.maximum(i, 0)
        return z.at[i_safe].set(jnp.where(do, z_next, z[i_safe]))

    z_lev = jax.lax.fori_loop(0, nlev - 1, down_body, z_lev)

    dz = z_lev[1:] - z_lev[:-1]
    z_lay = 0.5 * (z_lev[:-1] + z_lev[1:])
    return z_lev, z_lay, dz

