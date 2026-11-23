"""
vert_alt.py
===========
"""

from __future__ import annotations

from typing import Dict, Tuple

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

    # Per layer, with A_i = mu_i amu g_ref R0^2 / (kb T_i):
    #   dlnp_i = A_i * ( 1/(R0+z_i) - 1/(R0+z_{i+1}) )
    # The 1/r drops add up, so the profile is one cumulative sum anchored at the bottom level
    # (z = 0, r = R0), with no layer loop:  1/r_j = 1/R0 - C_j,  C_j = sum_{i<j} dlnp_i / A_i.
    inv_A = (kb * T_lay) / (mu_lay * amu * g_ref * R0 * R0)              # (nlay,)
    seg = dlnp * inv_A                                                   # 1/r drop across each layer
    C = jnp.concatenate([jnp.zeros((1,), dtype=seg.dtype), jnp.cumsum(seg)])   # (nlev,)
    inv_r = 1.0 / R0 - C
    # Guard against pathological (unbound) inputs that would make 1/r <= 0.
    inv_r = jnp.maximum(inv_r, jnp.finfo(p_lev.dtype).tiny)
    # Cancellation-free forms (not 1/inv_r - R0 and differences of z):
    #   z = r - R0 = R0 C / inv_r,   dz_i = r_{i+1} - r_i = seg_i / (inv_r_i inv_r_{i+1})
    z_lev = R0 * C / inv_r
    dz = seg / (inv_r[:-1] * inv_r[1:])
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
    k = jnp.searchsorted(-p_lev, -p_ref, side="right", method="compare_all") - 1
    k = jnp.clip(k, 0, nlev - 2)

    # Per layer, with g = g_ref (R0/r)^2 and constant T, mu, the hypsometric step is exact in 1/r:
    #   dlnp = A_i * (1/r_low - 1/r_up),  A_i = mu_i amu g_ref R0^2 / (kb T_i).
    # The 1/r drops therefore add up, so the whole profile is one cumulative sum (no level loop):
    #   1/r_j = 1/R0 - (C_j - C_ref),  C_j = sum_{i<j} dlnp_i / A_i,
    # with C_ref the partial sum up to p_ref inside the bracketing layer k (using layer k's T, mu).
    inv_A = (kb * T_lay) / (mu_lay * amu * g_ref * R0 * R0)              # (nlay,)
    seg = dlnp * inv_A                                                   # 1/r drop across each layer
    C = jnp.concatenate([jnp.zeros((1,), dtype=seg.dtype), jnp.cumsum(seg)])   # (nlev,), bottom level = 0
    C_ref = C[k] + jnp.log(p_lev[k] / p_ref) * inv_A[k]
    inv_r = 1.0 / R0 - (C - C_ref)
    # Guard against pathological (unbound) inputs that would make 1/r <= 0.
    inv_r = jnp.maximum(inv_r, jnp.finfo(p_lev.dtype).tiny)
    # Cancellation-free forms (not 1/inv_r - R0 and differences of z, which lose ~R0/dz digits):
    #   z = r - R0 = R0 (C - C_ref) / inv_r,   dz_i = r_{i+1} - r_i = seg_i / (inv_r_i inv_r_{i+1})
    z_lev = R0 * (C - C_ref) / inv_r
    dz = seg / (inv_r[:-1] * inv_r[1:])
    z_lay = 0.5 * (z_lev[:-1] + z_lev[1:])
    return z_lev, z_lay, dz
