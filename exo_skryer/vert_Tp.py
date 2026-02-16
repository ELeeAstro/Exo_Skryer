"""
vert_Tp.py
==========
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .data_constants import bar
from .aux_functions import pchip_1d

# ---------------- Hopf function ----------------
FIT_P = jnp.asarray([0.6162, -0.3799, 2.395, -2.041, 2.578])
FIT_Q = jnp.asarray([-0.9799, 3.917, -3.17, 3.69])

__all__ = [
    "hopf_function",
    "isothermal",
    "Barstow",
    "Milne",
    "Milne_modified",
    "Guillot",
    "Modified_Guillot",
    "MandS",
    "picket_fence",
    "dry_convective_adjustment"
]


def hopf_function(tau: jnp.ndarray) -> jnp.ndarray:
    """Compute the Hopf function for radiative transfer.

    This function provides a rational polynomial approximation for the Hopf
    function, used in analytical T-P profiles.

    Parameters
    ----------
    tau : `~jax.numpy.ndarray`
        Optical depth (dimensionless).

    Returns
    -------
    hopf : `~jax.numpy.ndarray`
        Hopf function value at the given optical depth.
    """
    tau = jnp.asarray(tau)
    tiny = jnp.finfo(tau.dtype).tiny
    tau_safe = jnp.maximum(tau, tiny)

    x = jnp.log10(tau_safe)

    # Rational fit in x via Horner
    p0, p1, p2, p3, p4 = FIT_P
    q0, q1, q2, q3 = FIT_Q
    num = ((((p0 * x + p1) * x + p2) * x + p3) * x + p4)
    den = ((((1.0 * x + q0) * x + q1) * x + q2) * x + q3)
    mid = num / den

    # Low-tau patch (linear in tau)
    low = 0.577351 + (tau_safe - 0.0) * (0.588236 - 0.577351) / (0.01 - 0.0)

    # High-tau patch (linear in log10(tau)) -- corrected denominator
    x0 = jnp.log10(5.0)
    x1 = jnp.log10(10000.0) 
    high = 0.710398 + (x - x0) * (0.710446 - 0.710398) / (x1 - x0)

    out = jnp.where(tau_safe < 0.01, low, mid)
    out = jnp.where(tau_safe > 5.0, high, out)
    return out

def isothermal(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate an isothermal temperature profile.

    Parameters
    ----------
    p_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Pressure at atmospheric levels.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `T_iso` : float
            Isothermal temperature in Kelvin.

    Returns
    -------
    T_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Temperature at levels in Kelvin.
    T_lay : `~jax.numpy.ndarray`, shape (nlev-1,)
        Temperature at layer midpoints in Kelvin.
    """
    nlev = jnp.size(p_lev)
    # Parameter values are already JAX arrays, no need to wrap
    T_iso = params["T_iso"]
    T_lev = jnp.full((nlev,), T_iso)
    T_lay = jnp.full((nlev-1,), T_iso)
    return T_lev, T_lay

def Barstow(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a Barstow et al. (2020) temperature profile.

    This profile is isothermal at low pressures, follows an adiabat in the
    middle, and becomes isothermal again at high pressures.

    Parameters
    ----------
    p_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Pressure at atmospheric levels.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `T_strat` : float
            Upper-atmosphere isothermal temperature in Kelvin.

    Returns
    -------
    T_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Temperature at levels in Kelvin.
    T_lay : `~jax.numpy.ndarray`, shape (nlev-1,)
        Temperature at layer midpoints in Kelvin.
    """
    # Parameter values are already JAX arrays, no need to wrap
    T_strat = params["T_strat"]
    kappa = 2.0 / 7.0
    p1 = 0.1 * bar
    p2 = 1.0 * bar
    p_for_adiabat = jnp.maximum(p_lev, p1)
    T_adiabat = T_strat * (p_for_adiabat / p1) ** kappa
    T_deep = T_strat * (p2 / p1) ** kappa
    T_lev = jnp.where(p_lev <= p1, T_strat, jnp.where(p_lev <= p2, T_adiabat, T_deep))
    T_lay = 0.5 * (T_lev[:-1] + T_lev[1:])
    return T_lev, T_lay

def Milne_modified(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a modified Milne temperature profile with stretched exponential transition.

    This profile uses a grey optical depth model with a stretched exponential
    transition from a skin temperature at low pressure to the standard Milne
    profile at high pressure.

    Parameters
    ----------
    p_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Pressure at atmospheric levels in dyne cm⁻².
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `T_int` : float
            Internal temperature in Kelvin.
        - `log_10_tau_ref` : float
            Log₁₀ infrared optical depth at reference pressure (1 bar, dimensionless).
        - `T_ratio` : float
            Skin-to-internal temperature ratio (T_skin / T_int, dimensionless).
        - `log_10_p_t` : float
            Log₁₀ transition pressure in bar.
        - `beta` : float
            Stretching exponent for transition (0 < beta <= 1, dimensionless).

    Returns
    -------
    T_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Temperature at levels in Kelvin.
    T_lay : `~jax.numpy.ndarray`, shape (nlev-1,)
        Temperature at layer midpoints in Kelvin.
    """

    g = 10.0**params["log_10_g"]
    T_int = params["T_int"]
    k_ir = 10.0**params["log_10_k_ir"]
    T_ratio = params["T_ratio"]
    p_t = (10.0**params["log_10_p_t"]) * bar
    beta = params["beta"]

    tau_ir = k_ir / g * p_lev

    q_inf = 0.710446
    q0 = (4.0 / 3.0) * (T_ratio**4)

    sig = jnp.exp(-((p_lev / p_t) ** beta))
    q = q_inf + (q0 - q_inf) * sig

    T_lev = (0.75 * T_int**4 * (q + tau_ir)) ** 0.25

    T_lay = 0.5 * (T_lev[:-1] + T_lev[1:])
    return T_lev, T_lay



def Milne(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a Milne temperature profile for an internally heated atmosphere.

    Parameters
    ----------
    p_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Pressure at atmospheric levels.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `log_10_g` : float
            Log₁₀ surface gravity in cm s⁻².
        - `T_int` : float
            Internal temperature in Kelvin.
        - `k_ir` : float
            Infrared opacity in cm² g⁻¹.

    Returns
    -------
    T_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Temperature at levels in Kelvin.
    T_lay : `~jax.numpy.ndarray`, shape (nlev-1,)
        Temperature at layer midpoints in Kelvin.
    """
    # Parameter values are already JAX arrays, no need to wrap
    g = 10.0**params["log_10_g"]
    T_int = params["T_int"]
    k_ir = 10.0**params["log_10_k_ir"]
    tau_ir = k_ir / g * p_lev
    T_lev = (0.75 * T_int**4 * (hopf_function(tau_ir) + tau_ir)) ** 0.25
    T_lay = 0.5 * (T_lev[:-1] + T_lev[1:])
    return T_lev, T_lay


def Guillot(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a Guillot (2010) analytical temperature profile.

    This profile combines internal heating and external irradiation using a
    two-stream approximation with separate visible and infrared opacities.

    Parameters
    ----------
    p_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Pressure at atmospheric levels.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `T_int` : float
            Internal temperature in Kelvin.
        - `T_eq` : float
            Equilibrium temperature in Kelvin.
        - `log_10_k_ir` : float
            Log₁₀ infrared opacity in cm² g⁻¹.
        - `log_10_gam_v` : float
            Log₁₀ visible-to-IR opacity ratio (dimensionless).
        - `log_10_g` : float
            Log₁₀ surface gravity in cm s⁻².
        - `f_hem` : float
            Hemispheric redistribution factor (dimensionless).

    Returns
    -------
    T_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Temperature at levels in Kelvin.
    T_lay : `~jax.numpy.ndarray`, shape (nlev-1,)
        Temperature at layer midpoints in Kelvin.
    """
    # Parameter values are already JAX arrays, no need to wrap
    T_int = params["T_int"]
    T_eq = params["T_eq"]
    k_ir = 10.0**params["log_10_k_ir"]
    gam = 10.0**params["log_10_gam_v"]
    g = 10.0**params["log_10_g"]
    f = params["f_hem"]
    tau_ir = k_ir / g * p_lev
    sqrt3 = jnp.sqrt(3.0)
    milne = 0.75 * T_int**4 * (2.0 / 3.0 + tau_ir)
    guillot = 0.75 * T_eq**4 * 4.0*f * (
        2.0 / 3.0
        + 1.0 / (gam * sqrt3)
        + (gam / sqrt3 - 1.0 / (gam * sqrt3)) * jnp.exp(-gam * tau_ir * sqrt3)
    )
    T_lev = (milne + guillot) ** 0.25
    T_lay = 0.5 * (T_lev[:-1] + T_lev[1:])
    return T_lev, T_lay


def Modified_Guillot(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a modified Guillot profile with a flexible irradiated Hopf term.

    This profile keeps the Guillot (2010) semi-grey structure but replaces the
    fixed 2/3 term in the irradiation component with a stretched-exponential
    Hopf-like transition in pressure.

    Parameters
    ----------
    p_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Pressure at atmospheric levels.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `T_int` : float
            Internal temperature in Kelvin.
        - `T_eq` : float
            Equilibrium temperature in Kelvin.
        - `log_10_k_ir` : float
            Log10 infrared opacity in cm^2 g^-1.
        - `log_10_gam_v` : float
            Log10 visible-to-IR opacity ratio.
        - `log_10_g` : float
            Log10 surface gravity in cm s^-2.
        - `f_hem` : float
            Hemispheric redistribution factor.
        - `q_irr_0` : float
            Irradiated Hopf value at low optical depth.
        - `log_10_p_t` : float
            Log10 transition pressure in bar.
        - `beta` : float
            Stretching exponent for the Hopf transition.

    Returns
    -------
    T_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Temperature at levels in Kelvin.
    T_lay : `~jax.numpy.ndarray`, shape (nlev-1,)
        Temperature at layer midpoints in Kelvin.
    """
    T_int = params["T_int"]
    T_eq = params["T_eq"]
    k_ir = 10.0 ** params["log_10_k_ir"]
    gam = 10.0 ** params["log_10_gam_v"]
    g = 10.0 ** params["log_10_g"]
    f = params["f_hem"]
    q_irr_0 = params["q_irr_0"]
    p_t = (10.0 ** params["log_10_p_t"]) * bar
    beta = params["beta"]

    tau_ir = (k_ir / g) * p_lev
    sqrt3 = jnp.sqrt(3.0)

    # Keep internal Hopf at the Eddington limit.
    q_int = 2.0 / 3.0

    # Flexible irradiated Hopf term: q_irr -> 2/3 at depth.
    q_inf = 2.0/3.0
    sig = jnp.exp(-((p_lev / p_t) ** beta))
    q_irr = q_inf + (q_irr_0 - q_inf) * sig

    internal = 0.75 * T_int**4 * (q_int + tau_ir)
    irradiated = 0.75 * T_eq**4 * 4.0 * f * (
        q_irr
        + 1.0 / (gam * sqrt3)
        + (gam / sqrt3 - 1.0 / (gam * sqrt3)) * jnp.exp(-gam * tau_ir * sqrt3)
    )

    T_lev = jnp.maximum(internal + irradiated, 0.0) ** 0.25
    T_lay = 0.5 * (T_lev[:-1] + T_lev[1:])
    return T_lev, T_lay


def MandS(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a Madhusudhan & Seager (2009) three-region T-P profile.

    This profile divides the atmosphere into three regions defined by
    pressure boundaries and slope parameters.

    Parameters
    ----------
    p_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Pressure at atmospheric levels.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `a1`, `a2` : float
            Shape/slope parameters controlling the inversion strength.
        - `log_10_P1`, `log_10_P2`, `log_10_P3` : float
            Transition pressures in bar (converted internally to linear pressure).
        - `T_ref` : float
            Reference temperature at the top of the atmosphere in Kelvin.

    Returns
    -------
    T_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Temperature at levels in Kelvin.
    T_lay : `~jax.numpy.ndarray`, shape (nlev-1,)
        Temperature at layer midpoints in Kelvin.
    """
    p_lev = jnp.asarray(p_lev)

    # Parameter values are already JAX arrays, no need to wrap
    a1 = params["a1"]
    a2 = params["a2"]
    P1 = 10.0 ** params["log_10_P1"] * bar
    P2 = 10.0 ** params["log_10_P2"] * bar
    P3 = 10.0 ** params["log_10_P3"] * bar
    T0 = params["T_ref"]

    # TOA pressure (p_lev is bottom->top)
    P0 = jnp.min(p_lev)

    def inv_sq(P, Pref, a):
        # avoid division-by-zero / NaNs if a is proposed extremely small
        a_safe = jnp.where(jnp.abs(a) > 1e-12, a, jnp.sign(a) * 1e-12 + (a == 0.0) * 1e-12)
        return (jnp.log(P / Pref) / a_safe) ** 2

    # Continuity
    T1 = T0 + inv_sq(P1, P0, a1)
    T2 = T1 - inv_sq(P1, P2, a2)
    T3 = T2 + inv_sq(P3, P2, a2)

    # Piecewise inversion T(P)
    T_reg1 = T0 + inv_sq(p_lev, P0, a1)   # P0 < P <= P1
    T_reg2 = T2 + inv_sq(p_lev, P2, a2)   # P1 < P <= P3

    in_reg1 = p_lev <= P1
    in_reg2 = (p_lev > P1) & (p_lev <= P3)

    T_lev = jnp.where(in_reg1, T_reg1, jnp.where(in_reg2, T_reg2, T3))
    T_lay = 0.5 * (T_lev[:-1] + T_lev[1:])
    return T_lev, T_lay

def Line(p_lev: jnp.ndarray,params: Dict[str, jnp.ndarray],) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Line et al. (2013) (two visible channels) analytic T(p) profile.

    Parameters
    ----------
    p_lev : jnp.ndarray, shape (nlev,)
        Pressure at atmospheric levels.
    params : dict[str, jnp.ndarray]
        Expected keys (all scalar JAX arrays):
        - "T_int" : internal temperature [K]
        - "T_eq"  : equilibrium temperature [K]
        - "f_hem" : redistribution factor
        - "log_10_k_ir" : log10 thermal/IR opacity [cm^2 g^-1]
        - "log_10_g"    : log10 gravity [cm s^-2]
        - "log_10_gam_v1": log10(γ1) visible/thermal opacity ratio (channel 1)
        - "log_10_gam_v2": log10(γ2) visible/thermal opacity ratio (channel 2)
        - "alpha" : partition between visible channels, α in [0, 1]

    Returns
    -------
    T_lev : jnp.ndarray, shape (nlev,)
        Temperature at levels [K]
    T_lay : jnp.ndarray, shape (nlev-1,)
        Temperature at layer midpoints [K]
    """
    T_int = params["T_int"]
    T_eq = params["T_eq"]
    f = params["f_hem"]

    k_ir = 10.0 ** params["log_10_k_ir"]
    g = 10.0 ** params["log_10_g"]

    gam1 = 10.0 ** params["log_10_gam_v1"]
    gam2 = 10.0 ** params["log_10_gam_v2"]
    alpha = params["alpha"]

    tau = (k_ir / g) * p_lev

    T_irr4 = (4.0 * f) * (T_eq**4)

    def xi_gamma(tau_: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
        x = gamma * tau_
        # E2(x) = exponential integral of order 2
        E2 = jsp.special.expn(2, x)

        term0 = 2.0 / 3.0
        term1 = (2.0 / (3.0 * gamma)) * (1.0 + (x / 2.0 - 1.0) * jnp.exp(-x))
        term2 = (2.0 * gamma / 3.0) * (1.0 - (tau_**2) / 2.0) * E2
        return term0 + term1 + term2

    # Eq. (13)
    T4 = (3.0 * T_int**4 / 4.0) * (2.0 / 3.0 + tau)
    T4 += (3.0 * T_irr4 / 4.0) * (1.0 - alpha) * xi_gamma(tau, gam1)
    T4 += (3.0 * T_irr4 / 4.0) * alpha * xi_gamma(tau, gam2)

    T_lev = T4 ** 0.25
    T_lay = 0.5 * (T_lev[:-1] + T_lev[1:])
    return T_lev, T_lay

def picket_fence(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a Parmentier & Guillot (2014,2015) picket fence T-P profile.

    This profile uses a picket fence approximation for radiative transfer,
    treating opacity as a combination of discrete spectral bins.

    Parameters
    ----------
    p_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Pressure at atmospheric levels.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `T_int`, `T_eq` : float
            Internal and equilibrium temperatures in Kelvin.
        - `log_10_k_ir`, `log_10_gam_v` : float
            Log₁₀ infrared opacity (cm² g⁻¹) and log₁₀ visible-to-IR ratio.
        - `log_10_R`, `Beta` : float
            Picket-fence parameters (dimensionless).
        - `log_10_g` : float
            Log₁₀ surface gravity in cm s⁻².
        - `f_hem` : float
            Hemispheric redistribution factor (dimensionless).

    Returns
    -------
    T_lev : `~jax.numpy.ndarray`, shape (nlev,)
        Temperature at levels in Kelvin.
    T_lay : `~jax.numpy.ndarray`, shape (nlev-1,)
        Temperature at layer midpoints in Kelvin.
    """

    # Parameter values are already JAX arrays, no need to wrap
    T_int = params["T_int"]
    T_eq = params["T_eq"]
    k_ir = 10.0**params["log_10_k_ir"]
    gam_v = 10.0**params["log_10_gam_v"]
    R = 10.0**params["log_10_R"]
    B = params["Beta"]
    g = 10.0**params["log_10_g"]
    f = params["f_hem"]

    tau_ir = k_ir / g * p_lev

    mu = 1.0/jnp.sqrt(3.0)

    gv = gam_v / mu 

    s = B + R - B * R
    gam_p = s + s / R - (s * s) / R     
    gam_1 = s                              
    gam_2 = s / R                             

    tau_lim = (jnp.sqrt(R) * jnp.sqrt(B * (1.0 - B) * (R - 1.0) ** 2 + R)) / (jnp.sqrt(3.0) * s ** 2)

    At1 = gam_1**2 * jnp.log(1.0 + 1.0 / (tau_lim * gam_1))
    At2 = gam_2**2 * jnp.log(1.0 + 1.0 / (tau_lim * gam_2))
    Av1 = gam_1**2 * jnp.log(1.0 + gv / gam_1)
    Av2 = gam_2**2 * jnp.log(1.0 + gv / gam_2)

    a0 = 1.0 / gam_1 + 1.0 / gam_2

    a1 = -(1.0 / (3.0 * tau_lim**2)) * (
        (gam_p / (1.0 - gam_p)) * ((gam_1 + gam_2 - 2.0) / (gam_1 + gam_2))
        + (gam_1 + gam_2) * tau_lim
        - (At1 + At2) * tau_lim**2
    )

    den_v = (1.0 - (gv**2) * (tau_lim**2))

    num_a2 = (
        (3.0 * gam_1**2 - gv**2) * (3.0 * gam_2**2 - gv**2) * (gam_1 + gam_2)
        - 3.0 * gv * (6.0 * gam_1**2 * gam_2**2 - gv**2 * (gam_1**2 + gam_2**2))
    )

    a2 = (tau_lim**2 / (gam_p * gv**2)) * (num_a2 / den_v)

    a3 = -(
        tau_lim**2
        * (3.0 * gam_1**2 - gv**2)
        * (3.0 * gam_2**2 - gv**2)
        * (Av2 + Av1)
    ) / (gam_p * gv**3 * den_v)

    term_b0 = (
        (gam_1 * gam_2 / (gam_1 - gam_2)) * (At1 - At2) / 3.0
        - (gam_1 * gam_2) ** 2 / jnp.sqrt(3.0 * gam_p)
        - (gam_1 * gam_2) ** 3 / ((1.0 - gam_1) * (1.0 - gam_2) * (gam_1 + gam_2))
    )
    b0 = 1.0 / term_b0

    b1 = (
        gam_1 * gam_2
        * (3.0 * gam_1**2 - gv**2)
        * (3.0 * gam_2**2 - gv**2)
        * tau_lim**2
    ) / (gam_p * gv**2 * (gv**2 * tau_lim**2 - 1.0))

    b2 = (3.0 * (gam_1 + gam_2) * gv**3) / (
        (3.0 * gam_1**2 - gv**2) * (3.0 * gam_2**2 - gv**2)
    )

    b3 = (Av2 - Av1) / (gv * (gam_1 - gam_2))

    # ---------- A..E (eqs 77-81) ----------
    A_pf = (a0 + a1 * b0) / 3.0
    B_pf = -(1.0 / 3.0) * ((gam_1 * gam_2) ** 2 / gam_p) * b0
    C_pf = -(1.0 / 3.0) * (b0 * b1 * (1.0 + b2 + b3) * a1 + a2 + a3)
    D_pf = (1.0 / 3.0) * ((gam_1 * gam_2) ** 2 / gam_p) * b0 * b1 * (1.0 + b2 + b3)
    E_pf = (
        (3.0 - (gv / gam_1) ** 2) * (3.0 - (gv / gam_2) ** 2)
    ) / (9.0 * gv * ((gv * tau_lim) ** 2 - 1.0))

    # ---------- Temperature profile (eq 76): returns T^4 ----------
    T_lev = (3.0 * T_int**4 / 4.0) * (tau_ir + A_pf + B_pf * jnp.exp(-tau_ir / tau_lim)) \
        + (3.0 * T_eq**4 / 4.0) * 4.0*f * (C_pf + D_pf * jnp.exp(-tau_ir / tau_lim) + E_pf * jnp.exp(-gv * tau_ir))
    T_lev = T_lev**0.25

    T_lay = 0.5 * (T_lev[:-1] + T_lev[1:])
    return T_lev, T_lay

def dry_convective_adjustment(T_lay: jnp.ndarray, p_lay: jnp.ndarray, p_lev: jnp.ndarray, kappa: float, max_iter: int = 10, tol: float = 1e-6) -> jnp.ndarray:
    """Apply dry convective adjustment to a temperature profile.

    This function iteratively adjusts a temperature profile to ensure it is
    convectively stable, preserving total enthalpy.

    Parameters
    ----------
    T_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Initial layer temperatures in Kelvin.
    p_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Layer pressures.
    p_lev : `~jax.numpy.ndarray`, shape (nlay+1,)
        Level pressures.
    kappa : float
        Adiabatic index (R/cp).
    max_iter : int, optional
        Maximum number of adjustment iterations.
    tol : float, optional
        Tolerance for the stability check.

    Returns
    -------
    T_lay_adj : `~jax.numpy.ndarray`, shape (nlay,)
        Convectively adjusted layer temperature profile in Kelvin.
    """
    nlay = T_lay.shape[0]

    # Calculate pressure differences (layer thicknesses)
    d_p = p_lev[1:] - p_lev[:-1]

    def adjust_pair(T_work, i1, i2):
        """Adjust a pair of layers if convectively unstable."""
        pfact = (p_lay[i1] / p_lay[i2]) ** kappa

        # Check convective stability: T(i) should be >= T(i+1) * pfact
        is_unstable = T_work[i1] < (T_work[i2] * pfact - tol)

        # Mass-weighted average temperature
        Tbar = (d_p[i1] * T_work[i1] + d_p[i2] * T_work[i2]) / (d_p[i1] + d_p[i2])

        # New temperatures after adjustment (conserves enthalpy)
        T_new_i2 = (d_p[i1] + d_p[i2]) * Tbar / (d_p[i2] + pfact * d_p[i1])
        T_new_i1 = T_new_i2 * pfact

        # Update only if unstable
        T_updated = jnp.where(
            is_unstable,
            T_work.at[i1].set(T_new_i1).at[i2].set(T_new_i2),
            T_work
        )

        return T_updated

    def single_iteration(T_curr, _):
        """One full iteration: downward pass + upward pass."""

        # Downward pass (from top to bottom: i=0 to nlay-2)
        def downward_body(i, T_work):
            return adjust_pair(T_work, i, i + 1)

        T_after_down = jax.lax.fori_loop(0, nlay - 1, downward_body, T_curr)

        # Upward pass (from bottom to top: i=nlay-2 to 0)
        def upward_body(i, T_work):
            idx = nlay - 2 - i
            return adjust_pair(T_work, idx, idx + 1)

        T_after_up = jax.lax.fori_loop(0, nlay - 1, upward_body, T_after_down)

        return T_after_up, None

    # Run max_iter iterations (no early exit in JAX scan)
    T_adjusted, _ = jax.lax.scan(single_iteration, T_lay, None, length=max_iter)

    return T_adjusted
