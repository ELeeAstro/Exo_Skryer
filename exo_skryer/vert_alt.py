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

    This implementation uses an exact (per-layer) analytic update for spherical
    gravity, assuming layer-wise constant temperature and mean molecular weight:

        g(z) = g_ref * (R0 / (R0 + z))**2

    Under these assumptions, hydrostatic balance integrates to a closed-form
    mapping between pressure ratio and altitude increment, avoiding any
    predictor-corrector or iterative solve.

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
    g_ref = 10.0**params["log_10_g"]
    R0 = params["R_p"] * R_jup

    dlnp = jnp.log(p_lev[:-1] / p_lev[1:])

    def step(z_i, inputs):
        T_i, mu_i, dlnp_i = inputs

        # Let A = mu*amu*g_ref*R0^2/(kb*T). Then:
        #   dlnp = A * ( 1/(R0+z_i) - 1/(R0+z_{i+1}) )
        # => 1/(R0+z_{i+1}) = 1/(R0+z_i) - dlnp/A
        A = (mu_i * amu * g_ref * (R0 * R0)) / (kb * T_i)
        inv_i = 1.0 / (R0 + z_i)
        inv_next = inv_i - (dlnp_i / A)

        # Guard against pathological inputs that would make inv_next <= 0.
        inv_next = jnp.maximum(inv_next, jnp.finfo(p_lev.dtype).tiny)

        z_next = (1.0 / inv_next) - R0
        dz_i = z_next - z_i
        return z_next, dz_i

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
    """Compute an altitude profile with altitude-dependent gravity anchored at p_ref.

    This scheme defines ``z = 0`` at a reference pressure ``p_ref`` and integrates
    both upward and downward to fill the full level grid.

    Like :func:`hypsometric_variable_g`, this implementation uses an exact
    (per-layer) analytic update for spherical gravity assuming layer-wise constant
    temperature and mean molecular weight:

        g(z) = g_ref * (R0 / (R0 + z))**2

    Notes
    -----
    - Levels at pressures greater than ``p_ref`` end up with negative altitudes.
    - ``p_ref`` is clipped to lie within the provided pressure grid.
    """
    g_ref = 10.0**params["log_10_g"]
    R0 = params["R_p"] * R_jup
    p_ref = 10.0**params["log_10_p_ref"] * bar

    nlev = p_lev.shape[0]
    dlnp = jnp.log(p_lev[:-1] / p_lev[1:])  # positive for descending grid

    p_ref = jnp.clip(p_ref, p_lev[-1], p_lev[0])

    # bracket index k such that p[k] >= p_ref >= p[k+1]
    k = jnp.searchsorted(-p_lev, -p_ref, side="right") - 1
    k = jnp.clip(k, 0, nlev - 2)

    def step_exact(layer_idx, z_start, delta_ln, direction):
        T = jnp.take(T_lay, layer_idx, mode="clip")
        mu = jnp.take(mu_lay, layer_idx, mode="clip")

        # Let A = mu*amu*g_ref*R0^2/(kb*T). Then:
        #   dlnp = A * ( 1/(R0+z_start) - 1/(R0+z_end) )
        # For direction=+1 (upward): inv_end = inv_start - dlnp/A
        # For direction=-1 (downward): inv_end = inv_start + dlnp/A
        A = (mu * amu * g_ref * (R0 * R0)) / (kb * T)
        inv_start = 1.0 / (R0 + z_start)
        inv_end = inv_start - direction * (delta_ln / A)

        # Guard against pathological inputs that would make inv_end <= 0.
        inv_end = jnp.maximum(inv_end, jnp.finfo(p_lev.dtype).tiny)
        return (1.0 / inv_end) - R0

    z_lev = jnp.zeros_like(p_lev)

    # partial steps from p_ref to the bracketing levels
    d_dn = jnp.log(p_lev[k] / p_ref)       # >= 0
    d_up = jnp.log(p_ref / p_lev[k + 1])   # >= 0

    z_lev = z_lev.at[k].set(step_exact(k, 0.0, d_dn, -1.0))
    z_lev = z_lev.at[k + 1].set(step_exact(k, 0.0, d_up, +1.0))

    # upward integration: i = k+1 .. nlev-2 updates level i+1
    def up_body(i, z):
        z_next = step_exact(i, z[i], dlnp[i], +1.0)
        do = i >= (k + 1)
        return z.at[i + 1].set(jnp.where(do, z_next, z[i + 1]))

    z_lev = jax.lax.fori_loop(0, nlev - 1, up_body, z_lev)

    # downward integration: i = k-1 .. 0 updates level i
    def down_body(ii, z):
        i = (k - 1) - ii
        z_next = step_exact(i, z[i + 1], dlnp[i], -1.0)
        do = i >= 0
        i_safe = jnp.maximum(i, 0)
        return z.at[i_safe].set(jnp.where(do, z_next, z[i_safe]))

    z_lev = jax.lax.fori_loop(0, nlev - 1, down_body, z_lev)

    dz = z_lev[1:] - z_lev[:-1]
    z_lay = 0.5 * (z_lev[:-1] + z_lev[1:])
    return z_lev, z_lay, dz
