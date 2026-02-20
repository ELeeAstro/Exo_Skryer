"""
Vertical cloud profile kernels.

This module contains functions that compute the vertical distribution of cloud
mass mixing ratio (q_c_lay) as a function of pressure and atmospheric conditions.
"""

from typing import Dict
import jax.numpy as jnp

from .data_constants import bar

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
        Pressure at layer centers in dyne cm⁻².
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
        Pressure at layer centers in dyne cm⁻².
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
            Log₁₀ base pressure in bar (converted to dyne cm⁻² internally).

    Returns
    -------
    q_c_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Cloud mass mixing ratio per layer.
    """
    # Retrieved parameters
    q_c_0 = 10.0 ** params["log_10_q_c"]
    alpha = 10.0 ** params["log_10_alpha_cld"]

    p_base = 10.0 ** params["log_10_p_base"] * bar  # bar → dyne cm⁻²

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
        Pressure at layer centers in dyne cm⁻².
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
            Log₁₀ linear pressure width of the slab in bar (Δp = 10^log_10_dp_slab).

    Returns
    -------
    q_c_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Cloud mass mixing ratio per layer (q_c inside slab, 0 outside).
    """
    # Retrieved parameters
    q_c_slab = 10.0 ** params["log_10_q_c"]

    # Slab boundaries in pressure (bars → dyne cm⁻²)
    P_top = 10.0 ** params["log_10_p_top_slab"] * bar  # bar → dyne cm⁻²
    Delta_p = 10.0 ** params["log_10_dp_slab"] * bar   # bar → dyne cm⁻²
    P_bot = P_top + Delta_p                             # P_c,top + Δp

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
        Pressure at layer centers in dyne cm⁻².
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
