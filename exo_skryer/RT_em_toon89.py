"""
RT_em_toon89.py
===============

Toon et al. (1989) two-stream source-function method, following Exo-FMS
lw_Toon89. Delta-Eddington scaling, with the 4-node Gauss-Jacobi-5 angles
for the source integration. Can be unstable for combined high ssa and g.

Memory layout: every per-layer quantity comes from ``_layer_terms``, which is
elementwise, so it is evaluated inside the scans one layer (or pair of layers)
at a time. The tridiagonal rows are built inside the elimination scan, and only
the elimination coefficients (needed by the back-substitution) and the
solution are kept for all layers.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from .RT_em_common import _MU_NODES, _MU_WEIGHTS, _sweep, _fused_sweep, _layer_contrib

__all__ = ["solve_toon89"]

_UBARI = 0.5
_TWOPI = 2.0 * jnp.pi


class _Terms(NamedTuple):
    """Delta-Eddington scaled two-stream quantities of a layer (or of all layers)."""
    w0: jnp.ndarray
    dtau: jnp.ndarray
    hg: jnp.ndarray
    alp: jnp.ndarray
    lam: jnp.ndarray
    gam: jnp.ndarray
    B0: jnp.ndarray
    B1: jnp.ndarray
    Cpm1: jnp.ndarray   # C+ at the layer top
    Cmm1: jnp.ndarray   # C- at the layer top
    Cp: jnp.ndarray     # C+ at the layer bottom
    Cm: jnp.ndarray     # C- at the layer bottom
    Ep: jnp.ndarray
    Em: jnp.ndarray
    E1: jnp.ndarray
    E2: jnp.ndarray
    E3: jnp.ndarray
    E4: jnp.ndarray


def _layer_terms(dtau_in, w_in, g_in, b_top, b_bot) -> _Terms:
    """Elementwise layer quantities; ``b_top``/``b_bot`` are the Planck values at the layer edges."""
    # Delta-Eddington scaling
    w0 = (1.0 - g_in**2) * w_in / (1.0 - w_in * g_in**2)
    dtau = (1.0 - w_in * g_in**2) * dtau_in
    hg = g_in / (1.0 + g_in)

    alp = jnp.sqrt((1.0 - w0) / (1.0 - w0 * hg))
    lam = alp * (1.0 - w0 * hg) / _UBARI
    gam = (1.0 - alp) / (1.0 + alp)
    term = _UBARI / (1.0 - w0 * hg)

    # Isothermal layers at low optical depth, else linear in tau
    iso = dtau <= 1.0e-6
    dtau_s = jnp.where(iso, 1.0, dtau)
    B1 = jnp.where(iso, 0.0, (b_bot - b_top) / dtau_s)
    B0 = jnp.where(iso, 0.5 * (b_bot + b_top), b_top)

    Cpm1 = B0 + B1 * term
    Cmm1 = B0 - B1 * term
    Cp = B0 + B1 * dtau + B1 * term
    Cm = B0 + B1 * dtau - B1 * term

    exptrm = jnp.minimum(lam * dtau, 35.0)
    Ep = jnp.exp(exptrm)
    Em = 1.0 / Ep
    return _Terms(w0, dtau, hg, alp, lam, gam, B0, B1, Cpm1, Cmm1, Cp, Cm,
                  Ep, Em, Ep + gam * Em, Ep - gam * Em, gam * Ep + Em, gam * Ep - Em)


def _pair_rows(u: _Terms, l: _Terms):
    """Tridiagonal rows (A, B, C, D) joining layer ``u`` (above) and ``l`` (below): even, odd."""
    gm1, gp1 = l.gam - 1.0, l.gam + 1.0
    even = ((u.E1 + u.E3) * gm1, (u.E2 + u.E4) * gm1, 2.0 * (1.0 - l.gam ** 2),
            gm1 * (l.Cpm1 - u.Cp) + (1.0 - l.gam) * (u.Cm - l.Cmm1))
    odd = (2.0 * (1.0 - u.gam ** 2), (u.E1 - u.E3) * gp1, (u.E1 + u.E3) * gm1,
           u.E3 * (l.Cpm1 - u.Cp) + u.E1 * (u.Cm - l.Cmm1))
    return even, odd


def _src_up(t: _Terms, xk1, xk2, mu, ex):
    """Upward source and transmission of a layer at the angles ``mu``; ``ex`` adds the angle axis."""
    def _safe(x):
        # lam*mu = 1 is a removable singularity; avoid 0/0 at exactly that point
        return jnp.where(jnp.abs(x) > 1.0e-12, x, 1.0e-12)

    # Source-function coefficients; pure absorption below w0 = 1e-4
    scat = t.w0 > 1.0e-4
    fac = _TWOPI * t.w0 / (1.0 + t.alp)
    g = jnp.where(scat, fac * xk1 * (1.0 + t.hg * t.alp), 0.0)
    h = jnp.where(scat, fac * xk2 * (1.0 - t.hg * t.alp), 0.0)
    b_corr = jnp.where(scat, t.B1 * (_UBARI * t.w0 * t.hg / (1.0 - t.w0 * t.hg)), 0.0)
    alpha1 = _TWOPI * (t.B0 + b_corr)
    alpha2 = _TWOPI * t.B1

    em2 = jnp.exp(-ex(t.dtau) / mu)
    lp, lm = ex(t.lam) * mu + 1.0, _safe(ex(t.lam) * mu - 1.0)
    src = ((ex(g) / lm) * (ex(t.Ep) * em2 - 1.0) + (ex(h) / lp) * (1.0 - ex(t.Em) * em2)
           + ex(alpha1) * (1.0 - em2) + ex(alpha2) * (mu - (ex(t.dtau) + mu) * em2))
    return em2, src


def _solve_two_stream(dtau_in, w_in, g_in, be):
    """Tridiagonal two-stream solution, top-first inputs; returns (xk1, xk2), (nlay, nwl).

    Exo-FMS dtridgl: upward elimination then downward substitution. Rows
    interleave even/odd layer interfaces: [first, e0, o0, e1, o1, ..., last].
    The rows of each interface are built inside the elimination step.
    """
    top = _layer_terms(dtau_in[0], w_in[0], g_in[0], be[0], be[1])
    bot = _layer_terms(dtau_in[-1], w_in[-1], g_in[-1], be[-2], be[-1])

    # Boundary rows (a_surf = 0)
    tautop = top.dtau * jnp.exp(-1.0)
    Btop = (1.0 - jnp.exp(-tautop / _UBARI)) * be[0]
    Bsurf = be[-1]
    zero = jnp.zeros_like(be[0])
    first = (zero, top.gam + 1.0, top.gam - 1.0, Btop - top.Cmm1)
    last = (bot.E1, bot.E2, zero, Bsurf - bot.Cp)

    def eliminate(carry, row):
        as_n, ds_n = carry
        a_i, b_i, c_i, d_i = row
        xi = 1.0 / (b_i - c_i * as_n)
        return (a_i * xi, (d_i - c_i * ds_n) * xi)

    @jax.checkpoint
    def elim_pair(carry, x):
        u = _layer_terms(*x[0])
        l = _layer_terms(*x[1])
        even, odd = _pair_rows(u, l)
        c_o = eliminate(carry, odd)
        c_e = eliminate(c_o, even)
        return c_e, (c_e, c_o)

    pairs = ((dtau_in[:-1], w_in[:-1], g_in[:-1], be[:-2], be[1:-1]),
             (dtau_in[1:], w_in[1:], g_in[1:], be[1:-1], be[2:]))
    c_last = (last[0] / last[1], last[3] / last[1])
    c_e0, ((as_e, ds_e), (as_o, ds_o)) = lax.scan(elim_pair, c_last, pairs, reverse=True)
    as_f, ds_f = eliminate(c_e0, first)
    del as_f

    def subst(x_prev, c):
        as_e_k, ds_e_k, as_o_k, ds_o_k = c
        x_e = ds_e_k - as_e_k * x_prev
        x_o = ds_o_k - as_o_k * x_e
        return x_o, (x_e, x_o)

    x_o_end, (x_e, x_o) = lax.scan(subst, ds_f, (as_e, ds_e, as_o, ds_o))
    x_last = c_last[1] - c_last[0] * x_o_end

    x_top = jnp.concatenate([ds_f[None], x_o], axis=0)        # rows 0, 2, 4, ... (layer tops)
    x_bot = jnp.concatenate([x_e, x_last[None]], axis=0)      # rows 1, 3, 5, ...
    xk1 = x_top + x_bot
    xk2 = x_top - x_bot
    xk2 = jnp.where(jnp.abs(xk2) < 1.0e-30 * jnp.abs(x_top), 0.0, xk2)
    return xk1, xk2, bot


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
    del be_internal
    nlay = dtau_layers.shape[0]
    be = jnp.asarray(be_levels)[::-1]
    dtau_in = jnp.asarray(dtau_layers)[::-1]
    w_in = jnp.asarray(ssa)[::-1]
    g_in = jnp.asarray(g_phase)[::-1]
    dtype = be.dtype

    xk1, xk2, bot = _solve_two_stream(dtau_in, w_in, g_in, be)

    # Upward sweep only (OLR); the downward sweep does not feed back.
    wuarr = 0.5 * jnp.asarray(_MU_WEIGHTS, dtype=dtype)
    mu_s = jnp.asarray(_MU_NODES, dtype=dtype)[:, None]          # (nmu, 1)
    I_bot = _TWOPI * (be[-1] + bot.B1 * mu_s)

    if not return_layer_contrib:
        def up(dt, w, g, b_top, b_bot, x1, x2):
            return _src_up(_layer_terms(dt, w, g, b_top, b_bot), x1, x2, mu_s, lambda x: x[None])

        xs = (dtau_in, w_in, g_in, be[:-1], be[1:], xk1, xk2)
        olr = wuarr @ _fused_sweep(I_bot, xs, up, reverse=True)
        return olr, jnp.zeros((nlay, be.shape[1]), dtype=dtype)

    # Contribution functions need every layer's source: (nlay, nmu, nwl).
    t = _layer_terms(dtau_in, w_in, g_in, be[:-1], be[1:])
    mu = mu_s[None]
    em2, src_up = _src_up(t, xk1, xk2, mu, lambda x: x[:, None])
    olr = wuarr @ _sweep(I_bot, em2, src_up, reverse=True)
    return olr, _layer_contrib(wuarr, src_up, t.dtau, mu)[::-1]
