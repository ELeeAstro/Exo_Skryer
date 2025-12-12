"""
Temperature-Pressure Profile Module
====================================

This module provides various analytical and semi-analytical temperature-pressure (T-P)
profile parameterizations for exoplanet and brown dwarf atmospheres. These profiles are
commonly used in atmospheric retrieval and forward modeling.

Functions
---------
hopf_function : Compute the Hopf function for radiative transfer
isothermal : Simple isothermal temperature profile
Barstow : Temperature profile with isothermal upper layers and adiabatic deeper layers
Milne : Milne temperature profile using internal temperature and IR opacity
Guillot : Guillot (2010) analytical T-P profile including irradiation
MandS09 : Madhusudhan & Seager (2009) 3-region analytical T-P profile
picket_fence : Robinson & Catling (2012) picket fence radiative transfer T-P profile
dry_convective_adjustment : Adjust temperature profile for convective stability

Notes
-----
All temperature profiles return both level temperatures (T_lev) and layer temperatures (T_lay).
Pressures are typically in units of bar or Pa depending on the function.
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp

from data_constants import bar

# ---------------- Hopf function ----------------
FIT_P = jnp.asarray([0.6162, -0.3799, 2.395, -2.041, 2.578])
FIT_Q = jnp.asarray([-0.9799, 3.917, -3.17, 3.69])

def hopf_function(tau: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Hopf function for radiative transfer.

    The Hopf function is a rational polynomial approximation with special patches
    for low and high optical depth regimes. Used in analytical T-P profiles like
    the Milne approximation.

    Parameters
    ----------
    tau : jnp.ndarray
        Optical depth (dimensionless).

    Returns
    -------
    jnp.ndarray
        Hopf function value at the given optical depth.

    Notes
    -----
    The function uses three regimes:
    - Low tau (< 0.01): Linear interpolation
    - Mid tau (0.01 - 5.0): Rational polynomial fit
    - High tau (> 5.0): Linear fit in log10(tau) space
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

def isothermal(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]):
    """
    Generate an isothermal temperature profile.

    The simplest temperature profile where temperature is constant at all pressure levels.

    Parameters
    ----------
    p_lev : jnp.ndarray
        Pressure at levels (nlev,) [bar or Pa].
    params : Dict[str, jnp.ndarray]
        Dictionary containing:
        - 'T_iso' : Isothermal temperature [K].

    Returns
    -------
    T_lev : jnp.ndarray
        Temperature at levels (nlev,) [K].
    T_lay : jnp.ndarray
        Temperature at layer midpoints (nlev-1,) [K].
    """
    nlev = jnp.size(p_lev)
    T_iso = jnp.asarray(params["T_iso"])
    T_lev = jnp.full((nlev,), T_iso)
    T_lay = jnp.full((nlev-1,), T_iso)
    return T_lev, T_lay

def Barstow(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]):
    """
    Generate a temperature profile with isothermal upper atmosphere and adiabatic deeper layers.

    This profile is isothermal at low pressures (p < 0.1 bar), follows an adiabat
    between 0.1 and 1.0 bar, and becomes isothermal again at deeper pressures (p > 1 bar).
    Based on Barstow et al. (2020).

    Parameters
    ----------
    p_lev : jnp.ndarray
        Pressure at levels (nlev,) [bar].
    params : Dict[str, jnp.ndarray]
        Dictionary containing:
        - 'T_iso' : Temperature at the isothermal upper boundary [K].

    Returns
    -------
    T_lev : jnp.ndarray
        Temperature at levels (nlev,) [K].
    T_lay : jnp.ndarray
        Temperature at layer midpoints (nlev-1,) [K].

    Notes
    -----
    Uses kappa = 2/7 for the adiabatic index.
    Transition pressures: p1 = 0.1 bar, p2 = 1.0 bar.
    """
    T_iso = jnp.asarray(params["T_iso"])
    kappa = 2.0 / 7.0
    p1 = 0.1 * bar
    p2 = 1.0 * bar
    p_for_adiabat = jnp.maximum(p_lev, p1)
    T_adiabat = T_iso * (p_for_adiabat / p1) ** kappa
    T_deep = T_iso * (p2 / p1) ** kappa
    T_lev = jnp.where(p_lev <= p1, T_iso, jnp.where(p_lev <= p2, T_adiabat, T_deep))
    T_lay = 0.5 * (T_lev[:-1] + T_lev[1:])
    return T_lev, T_lay


def Milne(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]):
    """
    Generate a Milne temperature profile for internal heating.

    The Milne approximation is a radiative equilibrium solution for an atmosphere
    heated from below (internal heating only). Uses the Hopf function to model
    the temperature-optical depth relationship.

    Parameters
    ----------
    p_lev : jnp.ndarray
        Pressure at levels (nlev,) [bar or Pa].
    params : Dict[str, jnp.ndarray]
        Dictionary containing:
        - 'log_10_g' : Log10 of surface gravity [cm/s^2].
        - 'T_int' : Internal temperature [K].
        - 'k_ir' : Infrared opacity [cm^2/g].

    Returns
    -------
    T_lev : jnp.ndarray
        Temperature at levels (nlev,) [K].
    T_lay : jnp.ndarray
        Temperature at layer midpoints (nlev-1,) [K].

    Notes
    -----
    The optical depth is calculated as tau_ir = k_ir / g * p_lev.
    """
    g = 10.0**jnp.asarray(params["log_10_g"])
    T_int = jnp.asarray(params["T_int"])
    k_ir = jnp.asarray(params["k_ir"])
    tau_ir = k_ir / g * p_lev
    T_lev = (0.75 * T_int**4 * (hopf_function(tau_ir) + tau_ir)) ** 0.25
    T_lay = 0.5 * (T_lev[:-1] + T_lev[1:])
    return T_lev, T_lay


def Guillot(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]):
    """
    Generate a Guillot (2010) analytical temperature profile.

    This profile combines internal heating (Milne component) and external irradiation
    (Guillot component) for irradiated exoplanet atmospheres. Assumes a two-stream
    approximation with different opacities in the visible and infrared.

    Parameters
    ----------
    p_lev : jnp.ndarray
        Pressure at levels (nlev,) [bar or Pa].
    params : Dict[str, jnp.ndarray]
        Dictionary containing:
        - 'T_int' : Internal temperature [K].
        - 'T_eq' : Equilibrium temperature from stellar irradiation [K].
        - 'log_10_k_ir' : Log10 of infrared opacity [cm^2/g].
        - 'log_10_gam_v' : Log10 of gamma = kappa_v / kappa_ir (visible/IR opacity ratio).
        - 'log_10_g' : Log10 of surface gravity [cm/s^2].
        - 'f_hem' : Hemispheric redistribution factor (0.25-1.0).

    Returns
    -------
    T_lev : jnp.ndarray
        Temperature at levels (nlev,) [K].
    T_lay : jnp.ndarray
        Temperature at layer midpoints (nlev-1,) [K].

    Notes
    -----
    Based on Guillot (2010) A&A 520, A27.
    The hemispheric factor f_hem represents heat redistribution:
    - f_hem = 0.5: uniform redistribution
    - f_hem = 1.0: no redistribution (day-side only)
    - f_hem = 0.25: full redistribution over entire sphere
    """
    T_int = jnp.asarray(params["T_int"])
    T_eq = jnp.asarray(params["T_eq"])
    k_ir = 10.0**jnp.asarray(params["log_10_k_ir"])
    gam = 10.0**jnp.asarray(params["log_10_gam_v"])
    g = 10.0**jnp.asarray(params["log_10_g"])
    f = jnp.asarray(params["f_hem"])
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

from typing import Dict, Tuple
import jax.numpy as jnp


def MandS09(
    p_lev: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate Madhusudhan & Seager (2009) three-region analytical temperature profile.

    This profile divides the atmosphere into three regions with smooth transitions,
    commonly used for hot Jupiter retrievals. The profile is defined by three
    pressure boundaries and two slope parameters.

    Parameters
    ----------
    p_lev : jnp.ndarray
        Pressure at levels (nlev,) [bar].
    params : Dict[str, jnp.ndarray]
        Dictionary containing:
        - 'a1' : Slope parameter for upper region (dimensionless).
        - 'a2' : Slope parameter for middle region (dimensionless).
        - 'log_10_P1' : Log10 of first transition pressure [bar].
        - 'log_10_P2' : Log10 of second transition pressure [bar].
        - 'log_10_P3' : Log10 of third transition pressure [bar].
        - 'T_ref' : Reference temperature at top-of-atmosphere P0 [K].

    Returns
    -------
    T_lev : jnp.ndarray
        Temperature at levels (nlev,) [K].
    T_lay : jnp.ndarray
        Temperature at layer midpoints (nlev-1,) [K].

    Notes
    -----
    Based on Madhusudhan & Seager (2009) ApJ 707, 24.
    P0 is automatically determined as min(p_lev).
    The three regions are:
    - Region 1: P0 < P <= P1 (upper atmosphere)
    - Region 2: P1 < P <= P3 (middle atmosphere)
    - Region 3: P > P3 (lower atmosphere, isothermal at T3)
    Temperature continuity is enforced at the boundaries.
    """
    p_lev = jnp.asarray(p_lev)

    a1 = jnp.asarray(params["a1"])
    a2 = jnp.asarray(params["a2"])
    P1 = 10.0 ** jnp.asarray(params["log_10_P1"])
    P2 = 10.0 ** jnp.asarray(params["log_10_P2"])
    P3 = 10.0 ** jnp.asarray(params["log_10_P3"])
    T0 = jnp.asarray(params["T_ref"])

    # TOA pressure (since your p_lev is bottom->top)
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


def picket_fence(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]):
    """
    Generate a picket fence radiative transfer temperature profile.

    This profile uses the Robinson & Catling (2012) picket fence approximation
    for radiative transfer, which treats the opacity as a combination of discrete
    spectral bins (pickets) to better capture wavelength-dependent absorption.

    Parameters
    ----------
    p_lev : jnp.ndarray
        Pressure at levels (nlev,) [bar or Pa].
    params : Dict[str, jnp.ndarray]
        Dictionary containing:
        - 'T_int' : Internal temperature [K].
        - 'T_eq' : Equilibrium temperature from stellar irradiation [K].
        - 'log_10_k_ir' : Log10 of infrared opacity [cm^2/g].
        - 'log_10_gam_v' : Log10 of gamma = kappa_v / kappa_ir (visible/IR opacity ratio).
        - 'log_10_R' : Log10 of R parameter (picket fence opacity ratio).
        - 'Beta' : Beta parameter (picket fence opacity distribution).
        - 'log_10_g' : Log10 of surface gravity [cm/s^2].
        - 'f_hem' : Hemispheric redistribution factor (0.25-1.0).

    Returns
    -------
    T_lev : jnp.ndarray
        Temperature at levels (nlev,) [K].
    T_lay : jnp.ndarray
        Temperature at layer midpoints (nlev-1,) [K].

    Notes
    -----
    Based on Robinson & Catling (2012) ApJ 757, 104.
    This is a more sophisticated approximation than Guillot, accounting for
    discrete opacity bins rather than gray atmosphere assumptions.
    """

    T_int = jnp.asarray(params["T_int"])
    T_eq = jnp.asarray(params["T_eq"])
    k_ir = 10.0**jnp.asarray(params["log_10_k_ir"])
    gam_v = 10.0**jnp.asarray(params["log_10_gam_v"])
    R = 10.0**jnp.asarray(params["log_10_R"])
    B = jnp.asarray(params["Beta"])
    g = 10.0**jnp.asarray(params["log_10_g"])
    f = jnp.asarray(params["f_hem"])

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

def dry_convective_adjustment(T_lay, p_lay, p_lev, kappa, max_iter=10, tol=1e-6):
    """
    Apply dry convective adjustment to enforce convective stability.

    Adjusts temperature profile to be convectively stable by ensuring that
    T(i) >= T(i+1) * (p(i)/p(i+1))^kappa for all adjacent layers.

    This implements the algorithm from dry_conv_adj_mod.f90, performing iterative
    downward and upward passes through the atmosphere until the profile is stable
    or max_iter is reached. When an unstable pair is found, both layers are adjusted
    using a mass-weighted average temperature that preserves total enthalpy.

    Parameters
    ----------
    T_lay : jnp.ndarray
        Layer temperatures (nlay,) [K].
    p_lay : jnp.ndarray
        Layer pressures (nlay,) [bar or Pa].
    p_lev : jnp.ndarray
        Level/edge pressures (nlay+1,) [bar or Pa].
    kappa : float
        R/cp ratio (dimensionless), typically ~0.286 for ideal gas.
    max_iter : int, optional
        Maximum number of adjustment iterations (default: 10).
    tol : float, optional
        Tolerance for stability check (default: 1e-6).

    Returns
    -------
    T_adjusted : jnp.ndarray
        Convectively adjusted temperature profile (nlay,) [K].
        Assumes instant adjustment (returns final state directly).

    Notes
    -----
    Based on Ray Pierrehumbert's dry convective adjustment implementation.
    The adjustment preserves total enthalpy through mass-weighted averaging.
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
