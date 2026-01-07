"""
Vertical cloud profile kernels.

This module contains functions that compute the vertical distribution of cloud
mass mixing ratio (q_c_lay) as a function of pressure and atmospheric conditions.
"""

from typing import Dict
import jax.numpy as jnp

__all__ = [
    "no_cloud",
    "exponential_decay_profile",
    "slab_profile",
    "const_profile",
]


def no_cloud(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    mu_lay: jnp.ndarray,
    rho_lay: jnp.ndarray,
    nd_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Return zero cloud mass mixing ratio (no clouds).

    Parameters
    ----------
    p_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Pressure at layer centers in microbar.
    T_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Layer temperatures in K.
    mu_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Mean molecular weight per layer in amu.
    rho_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Mass density per layer in g cm⁻³.
    nd_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Number density per layer in cm⁻³.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary (unused).

    Returns
    -------
    q_c_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Cloud mass mixing ratio (all zeros).
    """
    nlay = T_lay.shape[0]
    return jnp.zeros(nlay)


def exponential_decay_profile(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    mu_lay: jnp.ndarray,
    rho_lay: jnp.ndarray,
    nd_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Exponential decay cloud profile with hard base cutoff.

    This profile follows:
        q_c(P) = q_c_0 * (P / P_base)^alpha  for P < P_base
        q_c(P) = 0                           for P >= P_base

    Parameters
    ----------
    p_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Pressure at layer centers in microbar.
    T_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Layer temperatures in K.
    mu_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Mean molecular weight per layer in amu.
    rho_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Mass density per layer in g cm⁻³.
    nd_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Number density per layer in cm⁻³.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `log_10_q_c` : float
            Log₁₀ cloud mass mixing ratio at the base pressure.
        - `log_10_alpha_cld` : float
            Log₁₀ cloud pressure power-law exponent.
        - `log_10_p_base` : float
            Log₁₀ base pressure in bar (converted to microbar internally).

    Returns
    -------
    q_c_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Cloud mass mixing ratio per layer.

    Notes
    -----
    A hard cutoff is applied at P_base: clouds are zero for P >= P_base.
    This ensures q_c = 0 in the deep atmosphere below the cloud base.
    """
    # Retrieved parameters
    q_c_0 = 10.0 ** params["log_10_q_c"]
    alpha = 10.0 ** params["log_10_alpha_cld"]

    p_base = 10.0 ** params["log_10_p_base"] * 1e6  # bar → microbar

    # Hard cutoff: clouds only exist for P < P_base
    cloud_mask = p_lay < p_base

    # Compute exponential profile
    q_c_profile = q_c_0 * (p_lay / jnp.maximum(p_base, 1e-30)) ** alpha

    # Apply hard cutoff
    q_c_lay = jnp.where(cloud_mask, q_c_profile, 0.0)

    return q_c_lay


def slab_profile(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    mu_lay: jnp.ndarray,
    rho_lay: jnp.ndarray,
    nd_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Uniform slab cloud profile with hard pressure cutoffs.

    The cloud is present with constant q_c between P_top and P_bot, and zero outside.

    Parameters
    ----------
    p_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Pressure at layer centers in microbar.
    T_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Layer temperatures in K.
    mu_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Mean molecular weight per layer in amu.
    rho_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Mass density per layer in g cm⁻³.
    nd_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Number density per layer in cm⁻³.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `log_10_q_c` : float
            Log₁₀ cloud mass mixing ratio inside the slab.
        - `log_10_p_top_slab` : float
            Log₁₀ pressure at the top of the slab in bar.
        - `log_10_dp_slab` : float
            Log₁₀ pressure extent of the slab (P_bot = P_top * 10^Δlog_P).

    Returns
    -------
    q_c_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Cloud mass mixing ratio per layer (q_c inside slab, 0 outside).

    Notes
    -----
    The slab extends from P_top to P_bot = P_top * 10^(Δlog_P).
    Hard cutoffs are used (no smooth edges).
    """
    # Retrieved parameters
    q_c_slab = 10.0 ** params["log_10_q_c"]

    # Slab boundaries in pressure (bars → microbar)
    log_P_top = params["log_10_p_top_slab"]
    Delta_log_P = params["log_10_dp_slab"]

    P_top = 10.0 ** log_P_top * 1e6  # bar → microbar
    P_bot = 10.0 ** (log_P_top + Delta_log_P) * 1e6  # bar → microbar

    # Hard slab cutoff: 1 inside [P_top, P_bot], 0 outside
    slab_mask = jnp.logical_and(p_lay >= P_top, p_lay <= P_bot)
    q_c_lay = q_c_slab * slab_mask

    return q_c_lay


def const_profile(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    mu_lay: jnp.ndarray,
    rho_lay: jnp.ndarray,
    nd_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Constant cloud mass mixing ratio throughout the entire atmosphere.

    Parameters
    ----------
    p_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Pressure at layer centers in microbar.
    T_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Layer temperatures in K.
    mu_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Mean molecular weight per layer in amu.
    rho_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Mass density per layer in g cm⁻³.
    nd_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Number density per layer in cm⁻³.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `log_10_q_c` : float
            Log₁₀ cloud mass mixing ratio (constant value throughout atmosphere).

    Returns
    -------
    q_c_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Cloud mass mixing ratio per layer (constant value everywhere).
    """
    # Retrieved parameter
    q_c = 10.0 ** params["log_10_q_c"]

    # Return constant profile
    nlay = T_lay.shape[0]
    return jnp.full(nlay, q_c)
