"""
RT_em_dda.py
============

Four-stream discrete ordinates adding method (4DDA; Zhang et al. 2013),
thermal emission only, following Exo-FMS lw_DDA (without the source function
technique). Each homogeneous layer is solved analytically for its 2x2
reflection/transmission and thermal sources, and layers are added from the
bottom up. Double-Gauss streams; delta-M scaled Henyey-Greenstein phase
function with f = g**4. Four-stream accuracy for multiple scattering, only
2x2 solves.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import jax.numpy as jnp
from jax import lax

__all__ = ["solve_dda"]

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
