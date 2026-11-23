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
    "f_sed_profile",
    "f_sed_beta_profile",
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


def f_sed_profile(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    mu_lay: jnp.ndarray,
    rho_lay: jnp.ndarray,
    nd_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    p_lev: Optional[jnp.ndarray] = None,
    return_cover: bool = False,
):
    """Sedimentation-efficiency (f_sed) cloud profile with a base cutoff.

    Power-law decay of the cloud mass above the cloud base, with the
    sedimentation efficiency f_sed as the exponent (cf. Ackerman & Marley 2001)::

        q_c(P) = q_c_0 * (P / P_base)^f_sed  for P < P_base
        q_c(P) = 0                           for P >= P_base

    Larger f_sed gives thinner, more compact clouds; small f_sed gives
    vertically extended clouds.

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
        - `log_10_f_sed` : float
            Log₁₀ sedimentation efficiency (f_sed is the power-law exponent).
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
    f_sed = 10.0 ** params["log_10_f_sed"]

    p_base = 10.0 ** params["log_10_p_base"] * bar  # bar → dyne cm⁻²

    if p_lev is None:
        # Layer-centre cutoff: clouds only exist for P < P_base
        cover = (p_lay < p_base).astype(p_lay.dtype)
        p_eval = p_lay
    else:
        # Fraction of each layer's mass above the base
        cover = _layer_overlap(p_lev, jnp.zeros_like(p_base), p_base)
        p_eval = jnp.minimum(p_lay, p_base)

    q_c_lay = q_c_0 * (p_eval / jnp.maximum(p_base, 1e-30)) ** f_sed * cover
    return (q_c_lay, cover) if return_cover else q_c_lay


def f_sed_beta_profile(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    mu_lay: jnp.ndarray,
    rho_lay: jnp.ndarray,
    nd_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    p_lev: Optional[jnp.ndarray] = None,
    return_cover: bool = False,
):
    """Extended f_sed cloud profile: power law with an exponential drop-off above the base, steep tail below.

        q_c(P) = q_c,base (P/P_base)^alpha_top exp[beta_0 (P/P_base - 1)]   for P <= P_base
        q_c(P) = q_c,base (P/P_base)^(-alpha_bot)                             for P >  P_base

    The exponential term lets the cloud fall off from the base more flexibly than
    the pure power law (beta_0 -> 0 recovers the f_sed profile with
    alpha_top = f_sed; beta_0 <~ 0.01 is indistinguishable from it). Near the
    base only alpha_top + beta_0 matters, since ln(x) ~ x - 1 there; the two
    separate further up, where the exponential levels off at exp(-beta_0).
    Below the base the steep power law (alpha_bot ~ 20) models
    the virga region where the condensate evaporates. The profile is continuous at
    P_base, so it is evaluated at layer centres and the base still moves smoothly
    through the grid.

    Parameters
    ----------
    p_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Pressure at layer centers in dyne cm⁻².
    T_lay, mu_lay, rho_lay, nd_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Unused; kept for the vert_cloud kernel signature.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `log_10_q_c` : float
            Log₁₀ cloud mass mixing ratio at the base pressure.
        - `log_10_p_base` : float
            Log₁₀ base pressure in bar.
        - `log_10_alpha_top` : float
            Log₁₀ power-law index above the base.
        - `log_10_beta_0` : float
            Log₁₀ exponential drop-off coefficient above the base (the drop
            happens over a fractional pressure width ~1/beta_0 of P_base).
        - `alpha_bot` : float, optional
            Power-law index below the base (virga region); default 20.
    p_lev : `~jax.numpy.ndarray`, shape (nlay+1,), optional
        Unused (the profile is continuous at the base).
    return_cover : bool, optional
        Also return the fraction of each layer inside the cloud (1 everywhere,
        since the cloud extends through the whole column).

    Returns
    -------
    q_c_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Cloud mass mixing ratio per layer.
    cover : `~jax.numpy.ndarray`, shape (nlay,)
        Only with ``return_cover``.
    """
    del T_lay, mu_lay, rho_lay, nd_lay, p_lev
    q_c_base = 10.0 ** params["log_10_q_c"]
    p_base = 10.0 ** params["log_10_p_base"] * bar  # bar → dyne cm⁻²
    alpha_top = 10.0 ** params["log_10_alpha_top"]
    beta_0 = 10.0 ** params["log_10_beta_0"]
    alpha_bot = params.get("alpha_bot", 20.0)

    x = p_lay / jnp.maximum(p_base, 1e-30)
    # Clamp each branch to its own side so the unused branch cannot overflow.
    x_top = jnp.minimum(x, 1.0)
    x_bot = jnp.maximum(x, 1.0)
    q_top = x_top ** alpha_top * jnp.exp(beta_0 * (x_top - 1.0))
    q_bot = x_bot ** (-alpha_bot)
    q_c_lay = q_c_base * jnp.where(x <= 1.0, q_top, q_bot)
    cover = jnp.ones_like(p_lay)
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

        - `log_10_q_c` : float, optional
            Log₁₀ cloud mass mixing ratio inside the slab. Not needed with
            ``opac_cloud: grey_profile`` (q_c is then 1 inside the slab).
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
    # Retrieved parameters (q_c is optional: grey clouds only use the cover)
    q_c_slab = 10.0 ** params["log_10_q_c"] if "log_10_q_c" in params else 1.0

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
