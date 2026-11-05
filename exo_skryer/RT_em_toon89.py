"""
RT_em_toon89.py
===============

Toon et al. (1989) two-stream source-function method, following Exo-FMS
lw_Toon89. Delta-Eddington scaling, with the 4-node Gauss-Jacobi-5 angles
for the source integration. Can be unstable for combined high ssa and g.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from jax import lax

from .RT_em_common import _MU_NODES, _MU_WEIGHTS, _sweep, _layer_contrib

__all__ = ["solve_toon89"]


def _dtridgl(af: jnp.ndarray, bf: jnp.ndarray, cf: jnp.ndarray, df: jnp.ndarray) -> jnp.ndarray:
    """Tridiagonal solve ``af x[i-1] + bf x[i] + cf x[i+1] = df`` (Exo-FMS dtridgl).

    Upward elimination then downward substitution; arrays are (l, nwl).
    """
    as_l = af[-1] / bf[-1]
    ds_l = df[-1] / bf[-1]

    def elim(carry, x):
        as_n, ds_n = carry
        a_i, b_i, c_i, d_i = x
        xi = 1.0 / (b_i - c_i * as_n)
        as_i = a_i * xi
        ds_i = (d_i - c_i * ds_n) * xi
        return (as_i, ds_i), (as_i, ds_i)

    _, (as_, ds) = lax.scan(elim, (as_l, ds_l), (af[:-1], bf[:-1], cf[:-1], df[:-1]), reverse=True)
    as_ = jnp.concatenate([as_, as_l[None]], axis=0)
    ds = jnp.concatenate([ds, ds_l[None]], axis=0)

    def subst(x_prev, x):
        x_i = x[1] - x[0] * x_prev
        return x_i, x_i

    _, xk = lax.scan(subst, ds[0], (as_[1:], ds[1:]))
    return jnp.concatenate([ds[:1], xk], axis=0)


def solve_toon89(
    be_levels: jnp.ndarray,
    dtau_layers: jnp.ndarray,
    ssa: jnp.ndarray,
    g_phase: jnp.ndarray,
    be_internal: jnp.ndarray,
    return_layer_contrib: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Toon et al. (1989) two-stream source-function method (Exo-FMS lw_Toon89).

    Inputs are bottom-first, ``be_levels`` (nlev, nwl) and the rest (nlay, nwl).
    Returns the TOA upward flux ``olr`` (nwl,) and a bottom-first layer
    contribution (nlay, nwl), zeros unless ``return_layer_contrib``. The lower
    boundary is the bottom-level Planck intensity, so ``be_internal`` is unused
    (as in Exo-FMS).
    """
    nlay = dtau_layers.shape[0]
    be = jnp.asarray(be_levels)[::-1]
    dtau_in = jnp.asarray(dtau_layers)[::-1]
    w_in = jnp.asarray(ssa)[::-1]
    g_in = jnp.asarray(g_phase)[::-1]
    dtype = be.dtype
    ubari = 0.5
    twopi = 2.0 * jnp.pi

    # Delta-Eddington scaling
    w0 = (1.0 - g_in**2) * w_in / (1.0 - w_in * g_in**2)
    dtau = (1.0 - w_in * g_in**2) * dtau_in
    hg = g_in / (1.0 + g_in)

    alp = jnp.sqrt((1.0 - w0) / (1.0 - w0 * hg))
    lam = alp * (1.0 - w0 * hg) / ubari
    gam = (1.0 - alp) / (1.0 + alp)
    term = ubari / (1.0 - w0 * hg)

    # Isothermal layers at low optical depth, else linear in tau
    iso = dtau <= 1.0e-6
    dtau_s = jnp.where(iso, 1.0, dtau)
    B1 = jnp.where(iso, 0.0, (be[1:] - be[:-1]) / dtau_s)
    B0 = jnp.where(iso, 0.5 * (be[1:] + be[:-1]), be[:-1])

    # C+/C- at the top (Cpm1, Cmm1) and bottom (Cp, Cm) of each layer
    Cpm1 = B0 + B1 * term
    Cmm1 = B0 - B1 * term
    Cp = B0 + B1 * dtau + B1 * term
    Cm = B0 + B1 * dtau - B1 * term

    tautop = dtau[0] * jnp.exp(-1.0)
    Btop = (1.0 - jnp.exp(-tautop / ubari)) * be[0]
    Bsurf = be[-1]

    exptrm = jnp.minimum(lam * dtau, 35.0)
    Ep = jnp.exp(exptrm)
    Em = 1.0 / Ep
    E1 = Ep + gam * Em
    E2 = Ep - gam * Em
    E3 = gam * Ep + Em
    E4 = gam * Ep - Em

    # Tridiagonal system (a_surf = 0); rows interleave even/odd layer interfaces
    gm1, gp1 = gam[1:] - 1.0, gam[1:] + 1.0
    A_e = (E1[:-1] + E3[:-1]) * gm1
    B_e = (E2[:-1] + E4[:-1]) * gm1
    C_e = 2.0 * (1.0 - gam[1:] ** 2)
    D_e = gm1 * (Cpm1[1:] - Cp[:-1]) + (1.0 - gam[1:]) * (Cm[:-1] - Cmm1[1:])
    A_o = 2.0 * (1.0 - gam[:-1] ** 2)
    B_o = (E1[:-1] - E3[:-1]) * gp1
    C_o = (E1[:-1] + E3[:-1]) * gm1
    D_o = E3[:-1] * (Cpm1[1:] - Cp[:-1]) + E1[:-1] * (Cm[:-1] - Cmm1[1:])

    def rows(first, even, odd, last):
        mid = jnp.stack([even, odd], axis=1).reshape(-1, even.shape[-1])
        return jnp.concatenate([first[None], mid, last[None]], axis=0)

    zero = jnp.zeros_like(be[0])
    Af = rows(zero, A_e, A_o, E1[-1])
    Bf = rows(gam[0] + 1.0, B_e, B_o, E2[-1])
    Cf = rows(gam[0] - 1.0, C_e, C_o, zero)
    Df = rows(Btop - Cmm1[0], D_e, D_o, Bsurf - Cp[-1])

    xkk = _dtridgl(Af, Bf, Cf, Df)
    xk1 = xkk[0::2] + xkk[1::2]
    xk2 = xkk[0::2] - xkk[1::2]
    xk2 = jnp.where(jnp.abs(xk2) < 1.0e-30 * jnp.abs(xkk[0::2]), 0.0, xk2)

    # Source-function coefficients; pure absorption below w0 = 1e-4
    scat = w0 > 1.0e-4
    fac = twopi * w0 / (1.0 + alp)
    g = jnp.where(scat, fac * xk1 * (1.0 + hg * alp), 0.0)
    h = jnp.where(scat, fac * xk2 * (1.0 - hg * alp), 0.0)
    xj = jnp.where(scat, fac * xk1 * (1.0 - hg * alp), 0.0)
    xk = jnp.where(scat, fac * xk2 * (1.0 + hg * alp), 0.0)
    b_corr = jnp.where(scat, B1 * (ubari * w0 * hg / (1.0 - w0 * hg)), 0.0)
    alpha1 = twopi * (B0 + b_corr)
    sigma1 = twopi * (B0 - b_corr)
    alpha2 = sigma2 = twopi * B1

    def _safe(x):
        # lam*mu = 1 is a removable singularity; avoid 0/0 at exactly that point
        return jnp.where(jnp.abs(x) > 1.0e-12, x, 1.0e-12)

    # Upward sweep only (OLR); the downward sweep does not feed back.
    # Layer axis leads, angles are an inner batch axis: (nlay, nmu, nwl).
    mu = jnp.asarray(_MU_NODES, dtype=dtype)[None, :, None]
    wuarr = 0.5 * jnp.asarray(_MU_WEIGHTS, dtype=dtype)
    c = lambda x: x[:, None]
    em2 = jnp.exp(-c(dtau) / mu)
    lp, lm = c(lam) * mu + 1.0, _safe(c(lam) * mu - 1.0)
    src_up = ((c(g) / lm) * (c(Ep) * em2 - 1.0) + (c(h) / lp) * (1.0 - c(Em) * em2)
              + c(alpha1) * (1.0 - em2) + c(alpha2) * (mu - (c(dtau) + mu) * em2))
    I_bot = twopi * (Bsurf + B1[-1] * mu[0])
    olr = wuarr @ _sweep(I_bot, em2, src_up, reverse=True)

    if return_layer_contrib:
        return olr, _layer_contrib(wuarr, src_up, dtau, mu)[::-1]
    return olr, jnp.zeros((nlay, be.shape[1]), dtype=dtype)
