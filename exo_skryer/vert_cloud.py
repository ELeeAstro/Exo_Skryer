"""
Vertical cloud profile kernels.

This module contains functions that compute the vertical distribution of cloud
mass mixing ratio (q_c_lay) as a function of pressure and atmospheric conditions.
"""

from typing import Dict, Optional
import jax.numpy as jnp

from .data_constants import bar

__all__ = [
    "no_cloud",
    "exponential_decay_profile",
    "slab_profile",
    "const_profile",
]


def _layer_overlap(p_lev: jnp.ndarray, p_top: jnp.ndarray, p_bot: jnp.ndarray) -> jnp.ndarray:
    """Fraction of each layer's pressure range (i.e. mass) inside [p_top, p_bot].

    ``p_lev`` has shape (nlay+1,), ordered from the bottom (highest pressure) up.
    """
    p_hi = p_lev[:-1]
    p_lo = p_lev[1:]
    overlap = jnp.minimum(p_hi, p_bot) - jnp.maximum(p_lo, p_top)
    return jnp.clip(overlap / (p_hi - p_lo), 0.0, 1.0)


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
    p_lev: Optional[jnp.ndarray] = None,
    return_cover: bool = False,
):
    """Exponential decay cloud profile with a base cutoff.

    This profile follows:
        q_c(P) = q_c_0 * (P / P_base)^alpha  for P < P_base
        q_c(P) = 0                           for P >= P_base

    With ``p_lev`` the layer holding the base is filled in proportion to the
    fraction of its mass above P_base, so the base moves smoothly through the
    grid (finite gradients in ``log_10_p_base``). Without it the layer-centre
    cutoff is used.

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
    p_lev : `~jax.numpy.ndarray`, shape (nlay+1,), optional
        Level pressures in dyne cm⁻², bottom first.
    return_cover : bool, optional
        Also return the fraction of each layer inside the cloud.

    Returns
    -------
    q_c_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Cloud mass mixing ratio per layer.
    cover : `~jax.numpy.ndarray`, shape (nlay,)
        Only with ``return_cover``: fraction of each layer inside the cloud.
    """
    # Retrieved parameters
    q_c_0 = 10.0 ** params["log_10_q_c"]
    alpha = 10.0 ** params["log_10_alpha_cld"]

    p_base = 10.0 ** params["log_10_p_base"] * bar  # bar → dyne cm⁻²

    if p_lev is None:
        # Layer-centre cutoff: clouds only exist for P < P_base
        cover = (p_lay < p_base).astype(p_lay.dtype)
        p_eval = p_lay
    else:
        # Fraction of each layer's mass above the base
        cover = _layer_overlap(p_lev, jnp.zeros_like(p_base), p_base)
        p_eval = jnp.minimum(p_lay, p_base)

    q_c_lay = q_c_0 * (p_eval / jnp.maximum(p_base, 1e-30)) ** alpha * cover
    return (q_c_lay, cover) if return_cover else q_c_lay


def slab_profile(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    mu_lay: jnp.ndarray,
    rho_lay: jnp.ndarray,
    nd_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    p_lev: Optional[jnp.ndarray] = None,
    return_cover: bool = False,
):
    """Uniform slab cloud profile.

    The cloud is present with constant q_c between P_top and P_bot, and zero outside.
    With ``p_lev`` the layers holding the slab edges are filled in proportion to
    the fraction of their mass inside the slab, so the edges move smoothly through
    the grid (finite gradients in the slab pressures). Without it the
    layer-centre test is used.

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
    p_lev : `~jax.numpy.ndarray`, shape (nlay+1,), optional
        Level pressures in dyne cm⁻², bottom first.
    return_cover : bool, optional
        Also return the fraction of each layer inside the slab.

    Returns
    -------
    q_c_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Cloud mass mixing ratio per layer (q_c inside slab, 0 outside).
    cover : `~jax.numpy.ndarray`, shape (nlay,)
        Only with ``return_cover``: fraction of each layer inside the slab.
    """
    # Retrieved parameters
    q_c_slab = 10.0 ** params["log_10_q_c"]

    # Slab boundaries in pressure (bars → dyne cm⁻²)
    P_top = 10.0 ** params["log_10_p_top_slab"] * bar  # bar → dyne cm⁻²
    Delta_p = 10.0 ** params["log_10_dp_slab"] * bar   # bar → dyne cm⁻²
    P_bot = P_top + Delta_p                             # P_c,top + Δp

    if p_lev is None:
        # Layer-centre test: 1 inside [P_top, P_bot], 0 outside
        cover = jnp.logical_and(p_lay >= P_top, p_lay <= P_bot).astype(p_lay.dtype)
    else:
        # Fraction of each layer's mass inside [P_top, P_bot]
        cover = _layer_overlap(p_lev, P_top, P_bot)

    q_c_lay = q_c_slab * cover
    return (q_c_lay, cover) if return_cover else q_c_lay


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
