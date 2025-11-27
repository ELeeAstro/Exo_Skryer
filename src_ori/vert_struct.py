"""
vert_struct.py
==============

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
from jax.scipy.special import expn

from data_constants import bar

# ---------------- Hopf function ----------------
FIT_P = jnp.asarray([0.6162, -0.3799, 2.395, -2.041, 2.578])
FIT_Q = jnp.asarray([-0.9799, 3.917, -3.17, 3.69])

def hopf_function(tau: jnp.ndarray) -> jnp.ndarray:
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
    nlev = jnp.size(p_lev)
    T_iso = jnp.asarray(params["T_iso"])
    return jnp.full((nlev - 1,), T_iso)


def Milne(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]):
    log_g = jnp.asarray(params["log_g"])
    T_int = jnp.asarray(params["T_int"])
    k_ir = jnp.asarray(params["k_ir"])
    g = 10.0 ** log_g
    tau_ir = k_ir / g * p_lev
    T_lev = (0.75 * T_int**4 * (hopf_function(tau_ir) + tau_ir)) ** 0.25
    return 0.5 * (T_lev[:-1] + T_lev[1:])


def Guillot(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]):
    T_int = jnp.asarray(params["T_int"])
    T_eq = jnp.asarray(params["T_eq"])
    k_ir = jnp.asarray(params["k_ir"])
    gam = jnp.asarray(params["gam_v"])
    log_g = jnp.asarray(params["log_g"])
    f = jnp.asarray(params["f_hem"])
    T_irr = T_eq * jnp.sqrt(2.0)
    g = 10.0 ** log_g
    tau_ir = k_ir / g * p_lev
    sqrt3 = jnp.sqrt(3.0)
    milne = 0.75 * T_int**4 * (2.0 / 3.0 + tau_ir)
    guillot = 0.75 * T_irr**4 * f * (
        2.0 / 3.0
        + 1.0 / (gam * sqrt3)
        + (gam / sqrt3 - 1.0 / (gam * sqrt3)) * jnp.exp(-gam * tau_ir * sqrt3)
    )
    T_lev = (milne + guillot) ** 0.25
    return 0.5 * (T_lev[:-1] + T_lev[1:])


def Line(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]):
    T_int = jnp.asarray(params["T_int"])
    k_ir = jnp.asarray(params["k_ir"])
    gam_1 = jnp.asarray(params["gam_1"])
    gam_2 = jnp.asarray(params["gam_2"])
    alpha = jnp.asarray(params["alpha"])
    beta = jnp.asarray(params["beta"])
    T_eq = jnp.asarray(params["T_eq"])
    log_g = jnp.asarray(params["log_g"])
    T_irr = beta * T_eq
    g = 10.0 ** log_g
    tau_ir = k_ir / g * p_lev
    eps1 = (
        2.0 / 3.0
        + 2.0 / (3.0 * gam_1) * (1.0 + (gam_1 * tau_ir / 2.0 - 1.0) * jnp.exp(-tau_ir * gam_1))
        + 2.0 * gam_1 / 3.0 * (1.0 - tau_ir**2 / 2.0) * expn(2, gam_1 * tau_ir)
    )
    eps2 = (
        2.0 / 3.0
        + 2.0 / (3.0 * gam_2) * (1.0 + (gam_2 * tau_ir / 2.0 - 1.0) * jnp.exp(-tau_ir * gam_2))
        + 2.0 * gam_2 / 3.0 * (1.0 - tau_ir**2 / 2.0) * expn(2, gam_2 * tau_ir)
    )
    milne = 0.75 * T_int**4 * (2.0 / 3.0 + tau_ir)
    line = 0.75 * T_irr**4 * ((1.0 - alpha) * eps1 + alpha * eps2)
    T_lev = (milne + line) ** 0.25
    return 0.5 * (T_lev[:-1] + T_lev[1:])


def Barstow(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]):
    T_iso = jnp.asarray(params["T_iso"])
    kappa = 2.0 / 7.0
    p1 = 0.1 * bar
    p2 = 1.0 * bar
    p_for_adiabat = jnp.maximum(p_lev, p1)
    T_adiabat = T_iso * (p_for_adiabat / p1) ** kappa
    T_deep = T_iso * (p2 / p1) ** kappa
    T_lev = jnp.where(p_lev <= p1, T_iso, jnp.where(p_lev <= p2, T_adiabat, T_deep))
    return 0.5 * (T_lev[:-1] + T_lev[1:])


def MS09(p_lev: jnp.ndarray, params: Dict[str, jnp.ndarray]):
    a1 = jnp.asarray(params["a1"])
    a2 = jnp.asarray(params["a2"])
    p1 = jnp.asarray(params["p1"]) * bar
    p2 = jnp.asarray(params["p2"]) * bar
    p3 = jnp.asarray(params["p3"]) * bar
    T_ref = jnp.asarray(params["T_ref"])
    p_ref = jnp.asarray(params["p_ref"])

    p_min = p_lev[0]
    idx = jnp.argmin(jnp.abs(p_lev - p_ref))
    p_ref_i = p_lev[idx]

    def _region_deep():
        T3 = T_ref
        T2 = T3 - ((1.0 / a2) * (jnp.log(p3 / p2))) ** 2
        T1 = T2 + ((1.0 / a2) * (jnp.log(p1 / p2))) ** 2
        T0 = T1 - ((1.0 / a1) * (jnp.log(p1 / p_min))) ** 2
        return T0, T1, T2, T3

    def _region_mid():
        T2 = T_ref - ((1.0 / a2) * (jnp.log(p_ref_i / p2))) ** 2
        T1 = T2 + ((1.0 / a2) * (jnp.log(p1 / p2))) ** 2
        T3 = T2 + ((1.0 / a2) * (jnp.log(p3 / p2))) ** 2
        T0 = T1 - ((1.0 / a1) * (jnp.log(p1 / p_min))) ** 2
        return T0, T1, T2, T3

    def _region_upper():
        T0 = T_ref - ((1.0 / a1) * (jnp.log(p_ref_i / p_min))) ** 2
        T1 = T0 + ((1.0 / a1) * (jnp.log(p1 / p_min))) ** 2
        T2 = T1 - ((1.0 / a2) * (jnp.log(p1 / p2))) ** 2
        T3 = T2 + ((1.0 / a2) * (jnp.log(p3 / p2))) ** 2
        return T0, T1, T2, T3

    T0, T1, T2, T3 = jax.lax.cond(
        p_ref_i >= p3,
        lambda _: _region_deep(),
        lambda _: jax.lax.cond(p_ref_i >= p1, lambda __: _region_mid(), lambda __: _region_upper(), None),
        None,
    )

    def _temperature_level(p_value):
        return jax.lax.cond(
            p_value >= p3,
            lambda _: T3,
            lambda _: jax.lax.cond(
                (p_value < p3) & (p_value > p1),
                lambda __: T2 + ((1.0 / a2) * (jnp.log(p_value / p2))) ** 2,
                lambda __: T0 + ((1.0 / a1) * (jnp.log(p_value / p_min))) ** 2,
                None,
            ),
            None,
        )

    T_lev = jax.vmap(_temperature_level)(p_lev)
    return 0.5 * (T_lev[:-1] + T_lev[1:])
