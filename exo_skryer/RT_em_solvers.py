"""
RT_em_solvers.py
================

Emission radiative transfer solvers, selected with ``physics.em_scheme``:

- ``eaa`` (:func:`solve_alpha_eaa`): extended absorption approximation, linear in tau
  (Exo-FMS lw_AA_L). Fastest; approximate scattering.
- ``toon89`` (:func:`solve_toon89`): Toon et al. (1989) two-stream source-function method
  (Exo-FMS lw_Toon89).
- ``dda`` (:func:`solve_dda`): four-stream discrete ordinates adding method (Zhang et al.
  2013; Exo-FMS lw_DDA). Most accurate for multiple scattering.

All solvers share the signature
``(be_levels, dtau_layers, ssa, g_phase, be_internal, return_layer_contrib) -> (olr, layer_contrib)``
with bottom-first inputs, and return only the upward TOA flux (OLR) needed for emission
spectra. The entry points that call them are in ``RT_em_core_os`` and ``RT_em_core_ck``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import jax.numpy as jnp
from jax import lax

__all__ = ["solve_alpha_eaa", "solve_toon89", "solve_dda", "get_emission_solver"]



# ---------------------------------------------------------------------------
# Shared: Gauss-Jacobi-5 angular quadrature (Hogan 2024) and the layer intensity recursion
# ---------------------------------------------------------------------------

# Gauss-Jacobi-5 quadrature, 4 nodes (8 streams); weights for the measure 2 mu dmu (F = pi sum w I)
_MU_NODES = jnp.array([0.0454586727, 0.2322334416, 0.5740198775, 0.9030775973])
_MU_WEIGHTS = jnp.array([0.0092068785, 0.1285704278, 0.4323381850, 0.4298845087])


def _sweep(I0: jnp.ndarray, T: jnp.ndarray, S: jnp.ndarray, reverse: bool = False) -> jnp.ndarray:
    """Layer recursion ``I <- I * T + S`` over the leading (layer) axis; returns the end value only."""
    def step(I, x):
        return I * x[0] + x[1], None
    return lax.scan(step, I0, (T, S), reverse=reverse)[0]


def _layer_contrib(w_q, src_up: jnp.ndarray, dtau: jnp.ndarray, mu) -> jnp.ndarray:
    """Upward layer sources attenuated to TOA, quadrature summed: (nlay, nwl), top-first."""
    tau_above = jnp.cumsum(dtau, axis=0) - dtau
    return jnp.einsum("m,lmw->lw", w_q, src_up * jnp.exp(-tau_above[:, None] / mu))


# ---------------------------------------------------------------------------
# EAA: extended absorption approximation (eaa)
# ---------------------------------------------------------------------------

_SMALL_TAU = 1.0e-3


def _eaa_layer_coeffs(dtau_a: jnp.ndarray, mu) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Linear-in-tau layer transmission and source weights (Exo-FMS lw_AA_L).

    Returns ``T, psi_0, psi_u``; Taylor expansions are used at small slant
    absorption optical depth.
    """
    d = dtau_a / mu
    small = d < _SMALL_TAU
    d_s = jnp.where(small, 1.0, d)  # keeps the unused branch finite for grads
    T_exact = jnp.exp(-d_s)
    psi_0_exact = (d_s - (1.0 - T_exact)) / d_s
    psi_0_small = d * (0.5 + d * (-1.0 / 6.0 + d / 24.0))
    psi_u_small = d * (0.5 + d * (-1.0 / 3.0 + d / 8.0))
    psi_0 = jnp.where(small, psi_0_small, psi_0_exact)
    psi_u = jnp.where(small, psi_u_small, (1.0 - T_exact) - psi_0_exact)
    T = jnp.where(small, 1.0 - psi_u_small - psi_0_small, T_exact)
    return T, psi_0, psi_u


def solve_alpha_eaa(
    be_levels: jnp.ndarray,
    dtau_layers: jnp.ndarray,
    ssa: jnp.ndarray,
    g_phase: jnp.ndarray,
    be_internal: jnp.ndarray,
    return_layer_contrib: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Extended absorption approximation, linear in tau (Li 2002; Exo-FMS lw_AA_L).

    Inputs are bottom-first, ``be_levels`` (nlev, nwl) and the rest (nlay, nwl).
    Returns the TOA upward flux ``olr`` (nwl,) and a bottom-first layer
    contribution (nlay, nwl), zeros unless ``return_layer_contrib``.
    """
    nlay = dtau_layers.shape[0]
    be_levels = jnp.asarray(be_levels)[::-1]
    dtau = jnp.asarray(dtau_layers)[::-1]
    w_in = jnp.clip(jnp.asarray(ssa)[::-1], 0.0, 1.0)
    g_in = jnp.asarray(g_phase)[::-1]

    # EAA modified co-albedo; no delta scaling needed (invariant under a consistent delta transform).
    eps = jnp.sqrt(jnp.maximum((1.0 - w_in) * (1.0 - g_in * w_in), 0.0))
    dtau_a = eps * dtau

    w_q = jnp.asarray(_MU_WEIGHTS, dtype=be_levels.dtype)

    if not return_layer_contrib:
        # One bottom-up scan with the exact lower boundary I_up = I_down + B_int (net flux = F_int):
        #   I_top = U + (I_down_bot + B_int) * Q
        # U is the upward sweep started from zero; I_down_bot = sum_i S_down_i * prod_{j>i} T_j is
        # accumulated with Q = transmission of the layers already passed, and Q at the end is the
        # column transmission. Each step forms its layer's coefficients from (dtau_a, B_top, B_bot),
        # so no (nlay, nmu, nwl) array is stored (~3.4x less temporary memory than precomputing them).
        mu1 = jnp.asarray(_MU_NODES, dtype=be_levels.dtype)[:, None]          # (nmu, 1)

        def step(carry, x):
            I_dn, Q, U = carry
            d_i, B_top, B_bot = x                                              # (nwl,) each
            T_i, p0, pu = _eaa_layer_coeffs(d_i[None, :], mu1)                 # (nmu, nwl)
            S_dn = pu * B_top + p0 * B_bot
            S_up = p0 * B_top + pu * B_bot
            return (I_dn + S_dn * Q, Q * T_i, U * T_i + S_up), None

        z = jnp.zeros((mu1.shape[0], be_levels.shape[1]), dtype=be_levels.dtype)
        (I_dn, Q, U), _ = lax.scan(step, (z, jnp.ones_like(z), z),
                                   (dtau_a, be_levels[:-1], be_levels[1:]), reverse=True)
        olr = jnp.pi * (w_q @ (U + (I_dn + be_internal) * Q))
        return olr, jnp.zeros((nlay, be_levels.shape[1]), dtype=be_levels.dtype)

    # Contribution functions need every layer's upward source, so form the coefficients for all
    # layers at once: layer axis leads, angles are an inner batch axis: (nlay, nmu, nwl).
    mu = jnp.asarray(_MU_NODES, dtype=be_levels.dtype)[None, :, None]
    T, psi_0, psi_u = _eaa_layer_coeffs(dtau_a[:, None], mu)
    src_down = psi_u * be_levels[:-1, None] + psi_0 * be_levels[1:, None]
    src_up = psi_0 * be_levels[:-1, None] + psi_u * be_levels[1:, None]

    # Downward sweep (only its bottom value is needed), then the lower
    # boundary F_int = F_up - F_down, then the upward sweep to TOA.
    I_bot = _sweep(jnp.zeros_like(src_down[0]), T, src_down)
    olr = jnp.pi * (w_q @ _sweep(I_bot + be_internal, T, src_up, reverse=True))
    return olr, jnp.pi * _layer_contrib(w_q, src_up, dtau_a, mu)[::-1]


# ---------------------------------------------------------------------------
# Toon et al. (1989) two-stream source-function method (toon89)
# ---------------------------------------------------------------------------

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
    contribution (nlay, nwl), zeros unless ``return_layer_contrib``.

    Lower boundary: a reflecting surface emitting the internal flux,
    ``F_up = F_down + pi * B_int``, so the bottom net flux equals ``F_int`` for the
    given ``T_int`` (as in ``eaa`` and ``dda``). This replaces Exo-FMS's black
    surface at the bottom-level temperature, which ignored ``be_internal``. The
    upward source-function sweep starts from the matching isotropic intensity
    ``F_down(bottom) / pi + B_int``; ``F_down(bottom)`` comes from the two-stream
    solution, so no extra sweep is needed.
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
    B_int = jnp.broadcast_to(jnp.asarray(be_internal, dtype=dtype), be[0].shape)

    exptrm = jnp.minimum(lam * dtau, 35.0)
    Ep = jnp.exp(exptrm)
    Em = 1.0 / Ep
    E1 = Ep + gam * Em
    E2 = Ep - gam * Em
    E3 = gam * Ep + Em
    E4 = gam * Ep - Em

    # Tridiagonal system; rows interleave even/odd layer interfaces. Last row is the
    # lower boundary with surface reflectivity 1 and emission B_int (Toon et al. 1989,
    # A = e1 - R e3, B = e2 - R e4, D = S - C+ + R C-):  F_up = F_down + B_int (flux/pi units)
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
    Af = rows(zero, A_e, A_o, E1[-1] - E3[-1])
    Bf = rows(gam[0] + 1.0, B_e, B_o, E2[-1] - E4[-1])
    Cf = rows(gam[0] - 1.0, C_e, C_o, zero)
    Df = rows(Btop - Cmm1[0], D_e, D_o, B_int - Cp[-1] + Cm[-1])

    xkk = _dtridgl(Af, Bf, Cf, Df)
    # Two-stream downward flux (flux/pi units) at the bottom of the lowest layer
    F_down_bot = xkk[-2] * E3[-1] + xkk[-1] * E4[-1] + Cm[-1]
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
    # Isotropic upward intensity leaving the reflecting, internally heated lower boundary
    I_bot = twopi * jnp.broadcast_to(F_down_bot + B_int, src_up[0].shape)
    olr = wuarr @ _sweep(I_bot, em2, src_up, reverse=True)

    if return_layer_contrib:
        return olr, _layer_contrib(wuarr, src_up, dtau, mu)[::-1]
    return olr, jnp.zeros((nlay, be.shape[1]), dtype=dtype)


# ---------------------------------------------------------------------------
# Four-stream discrete ordinates adding method (dda)
# ---------------------------------------------------------------------------

# Double-Gauss quadrature (Gauss-Legendre on [0,1]), sum(w) = 1; flux = 2 pi sum(w mu I)
_MU = (0.2113248654051871, 0.7886751345948129)
_W = (0.5, 0.5)
_NMOM = 4
_W_MAX = 1.0 - 1.0e-6
_DTAU_MIN = 1.0e-10


def _phase_products() -> Tuple[np.ndarray, np.ndarray]:
    """PP[l, i, j] = w_j (2l+1) P_l(mu_i) P_l(mu_j) / 2, and the opposite-hemisphere version."""
    mu, w = np.array(_MU), np.array(_W)
    P = np.stack([np.ones(2), mu, 0.5 * (3.0 * mu**2 - 1.0), 0.5 * (5.0 * mu**3 - 3.0 * mu)])
    c2l = 2.0 * np.arange(_NMOM) + 1.0
    PP = 0.5 * w[None, None, :] * c2l[:, None, None] * P[:, :, None] * P[:, None, :]
    return PP, PP * ((-1.0) ** np.arange(_NMOM))[:, None, None]  # P_l(-mu) = (-1)^l P_l(mu)


_PP_S, _PP_D = _phase_products()


# 2x2 algebra with each element its own array: M = (m00, m01, m10, m11), v = (v0, v1).
# Keeps every operation a large elementwise op that XLA fuses.
def _mm(A, B):
    return (A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3],
            A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3])


def _mv(A, v):
    return (A[0] * v[0] + A[1] * v[1], A[2] * v[0] + A[3] * v[1])


def _inv(A):
    rdet = 1.0 / (A[0] * A[3] - A[1] * A[2])
    return (A[3] * rdet, -A[1] * rdet, -A[2] * rdet, A[0] * rdet)


def _add(A, B):
    return tuple(a + b for a, b in zip(A, B))


def _sub(A, B):
    return tuple(a - b for a, b in zip(A, B))


def _phase_matrix(chi, PP):
    """Quadrature-weighted phase matrix sum_l chi_l PP[l, i, j]."""
    return tuple(sum(chi[l] * PP[l, i, j] for l in range(_NMOM)) for i in range(2) for j in range(2))


def _layer_rt(dtau, w0, B_top, B_bot, chi):
    """Analytic 4-stream solution of each homogeneous layer (Exo-FMS layer_RT).

    Returns R, T (2x2), equal from either side, and the thermal sources Eu
    (emitted up at the top) and Ed (emitted down at the bottom), (2,).
    """
    tiny = jnp.finfo(dtau.dtype).tiny
    thin = dtau <= _DTAU_MIN
    dtau = jnp.where(thin, 1.0, dtau)
    r0, r1 = 1.0 / _MU[0], 1.0 / _MU[1]
    Ws = _phase_matrix(chi, _PP_S)
    Wd = _phase_matrix(chi, _PP_D)

    # mu dI+/dtau = alpha I+ - beta I-,  dI-/dtau = beta I+ - alpha I-
    al = ((1.0 - w0 * Ws[0]) * r0, -w0 * Ws[1] * r0, -w0 * Ws[2] * r1, (1.0 - w0 * Ws[3]) * r1)
    bt = (w0 * Wd[0] * r0, w0 * Wd[1] * r0, w0 * Wd[2] * r1, w0 * Wd[3] * r1)
    amb, apb = _sub(al, bt), _add(al, bt)
    Q = _mm(apb, amb)

    # Eigenvalues k^2 of Q; smaller root via det/lam1 to avoid cancellation
    tr = Q[0] + Q[3]
    det = Q[0] * Q[3] - Q[1] * Q[2]
    lam1 = 0.5 * (tr + jnp.sqrt(jnp.maximum(tr**2 - 4.0 * det, tiny)))
    k1 = jnp.sqrt(lam1)
    k2 = jnp.sqrt(jnp.maximum(det / lam1, tiny))

    # Decaying modes: I+ = Gp e^{-k tau}, I- = Gm e^{-k tau}, Gp/Gm = (X -/+ Y)/2, Y = (alpha-beta) X/k
    def mode(k):
        lam = k * k
        v1 = (Q[1], lam - Q[0])
        v2 = (lam - Q[3], Q[2])
        n1, n2 = v1[0]**2 + v1[1]**2, v2[0]**2 + v2[1]**2
        use1 = n1 >= n2
        s1 = 1.0 / jnp.sqrt(jnp.maximum(n1, tiny))
        s2 = 1.0 / jnp.sqrt(jnp.maximum(n2, tiny))
        X = (jnp.where(use1, v1[0] * s1, v2[0] * s2), jnp.where(use1, v1[1] * s1, v2[1] * s2))
        Y = _mv(amb, X)
        return ((X[0] - Y[0] / k) * 0.5, (X[1] - Y[1] / k) * 0.5), ((X[0] + Y[0] / k) * 0.5, (X[1] + Y[1] / k) * 0.5)

    (gp1, gm1), (gp2, gm2) = mode(k1), mode(k2)
    Gp = (gp1[0], gp2[0], gp1[1], gp2[1])  # columns are the modes
    Gm = (gm1[0], gm2[0], gm1[1], gm2[1])
    e1, e2 = jnp.exp(-k1 * dtau), jnp.exp(-k2 * dtau)
    GpE = (Gp[0] * e1, Gp[1] * e2, Gp[2] * e1, Gp[3] * e2)
    GmE = (Gm[0] * e1, Gm[1] * e2, Gm[2] * e1, Gm[3] * e2)

    # R + T = (Gp + Gm E)(Gm + Gp E)^-1,  R - T = (Gp - Gm E)(Gm - Gp E)^-1
    P1 = _mm(_add(Gp, GmE), _inv(_add(Gm, GpE)))
    P2 = _mm(_sub(Gp, GmE), _inv(_sub(Gm, GpE)))
    R = tuple(0.5 * (a + b) for a, b in zip(P1, P2))
    T = tuple(0.5 * (a - b) for a, b in zip(P1, P2))

    # Thermal particular solution, B linear in tau: I+/- = B(tau) +/- d B1, (E - w (Ws - Wd)) d = mu
    B1 = (B_bot - B_top) / dtau
    Wm = _sub(Ws, Wd)
    d = _mv(_inv((1.0 - w0 * Wm[0], -w0 * Wm[1], -w0 * Wm[2], 1.0 - w0 * Wm[3])), _MU)
    P2d = _mv(P2, d)
    Eu, Ed = [], []
    for i in range(2):
        sum_o = (B_top + B_bot) * (1.0 - P1[2 * i] - P1[2 * i + 1])
        dif_o = (B_top - B_bot) * (1.0 - P2[2 * i] - P2[2 * i + 1]) + 2.0 * B1 * (d[i] + P2d[i])
        Eu.append(0.5 * (sum_o + dif_o))
        Ed.append(0.5 * (sum_o - dif_o))

    # Optically negligible layers are transparent
    R = tuple(jnp.where(thin, 0.0, x) for x in R)
    T = tuple(jnp.where(thin, i, x) for i, x in zip((1.0, 0.0, 0.0, 1.0), T))
    Eu = tuple(jnp.where(thin, 0.0, x) for x in Eu)
    Ed = tuple(jnp.where(thin, 0.0, x) for x in Ed)
    return R, T, Eu, Ed


def solve_dda(
    be_levels: jnp.ndarray,
    dtau_layers: jnp.ndarray,
    ssa: jnp.ndarray,
    g_phase: jnp.ndarray,
    be_internal: jnp.ndarray,
    return_layer_contrib: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Four-stream discrete ordinates adding method (Zhang et al. 2013; Exo-FMS lw_DDA).

    Inputs are bottom-first, ``be_levels`` (nlev, nwl) and the rest (nlay, nwl).
    Returns the TOA upward flux ``olr`` (nwl,) and a bottom-first layer
    contribution (nlay, nwl), zeros unless ``return_layer_contrib``.
    """
    be = jnp.asarray(be_levels)
    dtype = be.dtype
    dtau = jnp.asarray(dtau_layers, dtype=dtype)
    g = jnp.asarray(g_phase, dtype=dtype)
    w0 = jnp.clip(jnp.asarray(ssa, dtype=dtype), 0.0, _W_MAX)

    # Henyey-Greenstein moments g^l, delta-M scaled with the first truncated moment f = g^4
    f = (g * g) ** 2
    f = jnp.where((w0 > 0.0) & (f > 1.0e-12) & (f < 1.0 - 1.0e-12), f, 0.0)
    rf = 1.0 / (1.0 - f)
    chi = (jnp.ones_like(g), (g - f) * rf, (g * g - f) * rf, (g * g * g - f) * rf)
    dtau = (1.0 - w0 * f) * dtau
    w0 = w0 * (1.0 - f) / (1.0 - w0 * f)

    # Bottom-first: the top of layer k is level k + 1
    R, T, Eu, Ed = _layer_rt(dtau, w0, be[1:], be[:-1], chi)

    # Upward adding from the bottom; lower boundary I_up = I_down + B_int in every stream.
    # Rb: reflection of the slab below, Su: upward intensity it emits.
    def add(carry, x):
        Rb, Su = carry
        R_k, T_k, Eu_k, Ed_k = x
        I_RR = _mm(Rb, R_k)
        A = _mm(T_k, _inv((1.0 - I_RR[0], -I_RR[1], -I_RR[2], 1.0 - I_RR[3])))
        src = _add(Eu_k, _mv(A, _mv(Rb, Ed_k)))
        carry = (_add(R_k, _mm(A, _mm(Rb, T_k))), _add(src, _mv(A, Su)))
        return carry, ((A, src) if return_layer_contrib else None)

    one, zero = jnp.ones_like(be[0]), jnp.zeros_like(be[0])
    b_int = jnp.broadcast_to(jnp.asarray(be_internal, dtype=dtype), be[0].shape)
    (_, Su_top), per_layer = lax.scan(add, ((one, zero, zero, one), (b_int, b_int)), (R, T, Eu, Ed))

    # No incident intensity at the top, so I_up(TOA) = Su_top
    wmu = tuple(2.0 * np.pi * w * m for w, m in zip(_W, _MU))
    olr = wmu[0] * Su_top[0] + wmu[1] * Su_top[1]

    if not return_layer_contrib:
        return olr, jnp.zeros_like(dtau)

    # Each layer's emission carried to TOA through the composite transfer of the layers above
    def to_top(u, x):
        A_k, s_k = x
        return (u[0] * A_k[0] + u[1] * A_k[2], u[0] * A_k[1] + u[1] * A_k[3]), u[0] * s_k[0] + u[1] * s_k[1]

    _, layer_contrib = lax.scan(to_top, (wmu[0] * one, wmu[1] * one), per_layer, reverse=True)
    return olr, layer_contrib


# ---------------------------------------------------------------------------
# Solver selection
# ---------------------------------------------------------------------------

solve_toon89_picaso = solve_toon89  # former name


def get_emission_solver(name: str):
    """Get emission RT solver function by name.

    Parameters
    ----------
    name : str
        Scheme name. Supported values:
        - "eaa", "alpha_eaa": Extended absorption approximation (linear in tau)
        - "toon89", "toon89_picaso": Toon et al. (1989) two-stream source-function method
        - "dda", "4dda": Four-stream discrete ordinates adding method

    Returns
    -------
    solver : callable
        Emission solver function with signature:
        (be_levels, dtau_layers, ssa, g_phase, be_internal, return_layer_contrib)
        -> (olr, layer_contrib_flux)

    Raises
    ------
    NotImplementedError
        If the scheme name is not recognized.
    """
    name = str(name).lower().strip()
    if name in ("eaa", "alpha_eaa"):
        return solve_alpha_eaa
    elif name in ("toon89", "toon89_picaso"):
        return solve_toon89
    elif name in ("dda", "4dda"):
        return solve_dda
    raise NotImplementedError(f"Unknown emission scheme '{name}'")
