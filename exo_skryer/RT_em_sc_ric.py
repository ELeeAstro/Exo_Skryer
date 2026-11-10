"""
RT_em_sc_ric.py
===============

Linear short characteristics with anisotropic multiple scattering, solved by
Riccati invariant imbedding (Exo-FMS semi-grey lw_sc_scat_ric), with the
source function technique flux integration along the Hogan (2024)
Gauss-Jacobi angles. Thermal emission only.

Discrete problem (as the Fortran): double-Gauss discrete ordinates (``nmu``
per hemisphere), delta-M scaled Henyey-Greenstein phase function with
f = g**(2 nmu) (applied for g >= 1e-6), single scattering properties averaged
from layers onto levels (extinction weighted), source linear in tau along
each stream between levels.

At level k (top-first in the notation below), d_k / u_k are the down / up
stream intensities, q_k = (1 - ssa_k) B_k, F_k / G_k the forward / backward
scattering blocks, and per layer E = exp(-dtau/mu), P = psi_u, Q = psi_0.
With u_k = R_k d_k + s_k (R = I, s = B_int at the lower boundary):

    W = I - Q (F_{k+1} + G_{k+1} R_{k+1}),   c1 = P q_k + Q q_{k+1} + Q G_{k+1} s_{k+1}
    V = (E + P F_{k+1}) R_{k+1} + P G_{k+1}, c2 = (E + P F_{k+1}) s_{k+1} + P q_{k+1} + Q q_k
    Z = V W^-1,  Aa = E + P F_k,  Bg = P G_k
    X = I - Q F_k - Z Bg,  Y = Q G_k + Z Aa
    R_k = X^-1 Y,  s_k = X^-1 (Z c1 + c2)
    T_k = W^-1 (Aa + Bg R_k),  t_k = W^-1 (Bg s_k + c1)

then d_1 = 0, d_{k+1} = T_k d_k + t_k. The stream intensities give the level
source function at the Hogan angles, which is integrated once more with the
same linear short characteristics to give the TOA flux.

Differences from the Fortran, for the JAX layout:

- The clear-layer shortcut is not taken: with F = G = 0 the general formulas
  reduce to it exactly, and everything stays elementwise on (nlev, nwl) arrays.
- The final Hogan-angle sweeps are written in closed form with cumulative
  optical depths (no scan), which also gives each layer's contribution to the
  TOA flux directly.
- Only the TOA flux is returned. The lower boundary is I_up = I_down + B_int,
  no incident intensity at the top (as the Fortran and RT_em_dda).

Known limitation (Fortran notes): optically thick, near-conservative layers
(ssa -> 1, dtau >> 1) are under-resolved by the linear-in-tau source and the
OLR is too low; it converges with vertical resolution.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["solve_sc_ric", "make_sc_ric_solver"]

_NMU = 4
_PSI_SMALL = 1.0e-6

# Hogan (2024) Gauss-Jacobi-5 angles, weights for the measure 2 mu dmu (F = pi sum(wh I))
_UH = (0.0454586727, 0.2322334416, 0.5740198775, 0.9030775973)
_WH = (0.0092068785, 0.1285704278, 0.4323381850, 0.4298845087)

Mat = List[List[jnp.ndarray]]
Vec = List[jnp.ndarray]


# ---------------------------------------------------------------------------
# Unrolled small-matrix algebra on nested lists of arrays
# ---------------------------------------------------------------------------
def _eye(n: int) -> Mat:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _mm(A: Mat, B: Mat) -> Mat:
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def _mv(A: Mat, v: Vec) -> Vec:
    n = len(A)
    return [sum(A[i][k] * v[k] for k in range(n)) for i in range(n)]


def _inv(A: Mat) -> Mat:
    """Gauss-Jordan inverse without pivoting (W and X are diagonally dominant M-matrices)."""
    n = len(A)
    M = [list(r) for r in A]
    X = _eye(n)
    for k in range(n):
        r = 1.0 / M[k][k]
        M[k] = [x * r for x in M[k]]
        X[k] = [x * r for x in X[k]]
        for i in range(n):
            if i == k:
                continue
            f = M[i][k]
            M[i] = [a - f * b for a, b in zip(M[i], M[k])]
            X[i] = [a - f * b for a, b in zip(X[i], X[k])]
    return X


def _legendre(nmom: int, x: np.ndarray) -> np.ndarray:
    P = np.zeros((nmom, len(x)))
    P[0] = 1.0
    if nmom > 1:
        P[1] = x
    for l in range(1, nmom - 1):
        P[l + 1] = ((2 * l + 1) * x * P[l] - l * P[l - 1]) / (l + 1)
    return P


@lru_cache(maxsize=None)
def _tables(nmu: int):
    """Double-Gauss nodes/weights (sum(w) = 1), P_l at the nodes and Hogan angles, and
    PPw[l, i, j] = (2l+1)/2 P_l(u_i) P_l(u_j) w_j."""
    z, wz = np.polynomial.legendre.leggauss(nmu)
    mu = 0.5 * (z + 1.0)
    w = 0.5 * wz
    nstr = 2 * nmu
    Pl = _legendre(nstr, mu)
    Ph = _legendre(nstr, np.array(_UH))
    c = 0.5 * (2.0 * np.arange(nstr) + 1.0)
    PPw = c[:, None, None] * Pl[:, :, None] * Pl[:, None, :] * w[None, None, :]
    return mu, w, Pl, Ph, PPw


def _psi(dtau, mu: float):
    """Linear short characteristic coefficients: E = exp(-del), psi_u, psi_0 (series for small del)."""
    d = dtau / mu
    e = jnp.exp(-d)
    small = d < _PSI_SMALL
    ds = jnp.where(small, 1.0, d)
    es = jnp.exp(-ds)
    pu_ex = (1.0 - (1.0 + ds) * es) / ds
    p0_ex = -jnp.expm1(-ds) - pu_ex
    pu = jnp.where(small, d * (0.5 - d * (1.0 / 3.0 - d * 0.125)), pu_ex)
    p0 = jnp.where(small, d * (0.5 - d * (1.0 / 6.0 - d / 24.0)), p0_ex)
    return e, pu, p0


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def make_sc_ric_solver(nmu: int = _NMU) -> Callable:
    """Build a Riccati short characteristics emission solver with ``nmu`` streams per hemisphere."""
    mu, wq, Pl, Ph, PPw = _tables(nmu)
    nstr = 2 * nmu
    sgn = [1.0 - 2.0 * (l % 2) for l in range(nstr)]
    I = _eye(nmu)

    def solve_sc_ric(
        be_levels: jnp.ndarray,
        dtau_layers: jnp.ndarray,
        ssa: jnp.ndarray,
        g_phase: jnp.ndarray,
        be_internal: jnp.ndarray,
        return_layer_contrib: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Riccati short characteristics with scattering (Exo-FMS lw_sc_scat_ric, use_sft).

        Inputs are bottom-first, ``be_levels`` (nlev, nwl) and the rest (nlay, nwl).
        Returns the TOA upward flux ``olr`` (nwl,) and a bottom-first layer
        contribution (nlay, nwl), zeros unless ``return_layer_contrib``.
        """
        be = jnp.asarray(be_levels)
        dtype = be.dtype
        dtau_in = jnp.asarray(dtau_layers, dtype=dtype)
        g = jnp.asarray(g_phase, dtype=dtype)
        w_in = jnp.clip(jnp.asarray(ssa, dtype=dtype), 0.0, 1.0)
        b_int = jnp.broadcast_to(jnp.asarray(be_internal, dtype=dtype), be[0].shape)

        # Delta-M scaled Henyey-Greenstein, f = g**nstr for g >= 1e-6 (as the Fortran)
        def layer_props(dt_in, w, gg):
            """Scaled dtau and scattering moments ssa * dtau * chi_l of a layer (elementwise)."""
            f = gg ** nstr
            f = jnp.where((gg >= 1.0e-6) & (f < 1.0 - 1.0e-12), f, 0.0)
            rf = 1.0 / (1.0 - f)
            sw = w * (1.0 - f) * dt_in               # = ssa' * dtau' after scaling
            return (1.0 - w * f) * dt_in, [sw] + [sw * ((gg ** l - f) * rf) for l in range(1, nstr)]

        def level(lay_b, lay_a, b_lev):
            """Level moments ssa * chi_l and q = (1 - ssa) B, extinction weighted from the
            layers below and above (ssa * chi_l = sum(sw chi_l) / sum(dtau))."""
            (dt_b, m_b), (dt_a, m_a) = lay_b, lay_a
            ext = dt_b + dt_a
            pos = ext > 0.0
            r = 1.0 / jnp.where(pos, ext, 1.0)
            wc = [jnp.where(pos, (x + y) * r, 0.0) for x, y in zip(m_b, m_a)]
            return wc, (1.0 - wc[0]) * b_lev

        def levels(rb, rk, ra, b_lo, b_hi):
            """Both level moments of layer k from the raw inputs of layers k - 1, k, k + 1."""
            lb, lk, la = (layer_props(*r) for r in (rb, rk, ra))
            wc_lo, q_lo = level(lb, lk, b_lo)
            wc_hi, q_hi = level(lk, la, b_hi)
            return wc_lo, wc_hi, lk[0], q_lo, q_hi

        # The level moments (and the F/G blocks built from them) are recomputed inside each
        # sweep step from the raw layer inputs, so nothing per level is stored but R and s.
        # Bottom-first: level j is bounded by layers j - 1 (below) and j (above); zero
        # layers pad both ends.
        raw = jnp.stack([dtau_in, w_in, g], axis=1)                  # (nlay, 3, nwl)
        rp = jnp.pad(raw, ((1, 1), (0, 0), (0, 0)))
        lay3 = (rp[:-2], rp[1:-1], rp[2:], be[:-1], be[1:])
        dtau = layer_props(dtau_in, w_in, g)[0]

        def blocks(wc):
            """Forward (F) and backward (G) scattering blocks from one level's moments."""
            F = [[sum(float(PPw[l, i, j]) * wc[l] for l in range(nstr)) for j in range(nmu)]
                 for i in range(nmu)]
            G = [[sum(sgn[l] * float(PPw[l, i, j]) * wc[l] for l in range(nstr)) for j in range(nmu)]
                 for i in range(nmu)]
            return F, G

        def layer_ops(wc_lo, wc_hi, dt, q_lo, q_hi, R_lo, s_lo):
            """Per-layer pieces shared by both sweeps: W^-1, Aa, Bg, c1 (and blocks, coefficients)."""
            n = range(nmu)
            F_lo, G_lo = blocks(wc_lo)
            F_hi, G_hi = blocks(wc_hi)
            e, pu, p0 = zip(*[_psi(dt, float(m)) for m in mu])
            GR = _mm(G_lo, R_lo)
            Gs = _mv(G_lo, s_lo)
            W = [[I[i][j] - p0[i] * (F_lo[i][j] + GR[i][j]) for j in n] for i in n]
            c1 = [pu[i] * q_hi + p0[i] * (q_lo + Gs[i]) for i in n]
            Aa = [[I[i][j] * e[i] + pu[i] * F_hi[i][j] for j in n] for i in n]
            Bg = [[pu[i] * G_hi[i][j] for j in n] for i in n]
            return _inv(W), Aa, Bg, c1, (F_lo, G_lo, F_hi, G_hi, e, pu, p0)

        # ---- Upward Riccati sweep, bottom to top: u = R d + s at each level ----
        # Each step outputs R, s at its lower level (the only per-level values kept);
        # T and t are recomputed from them in the downward sweep.
        @jax.checkpoint
        def ric(carry, x):
            R, s = carry
            wc_lo, wc_hi, dt, q_lo, q_hi = levels(*x)
            n = range(nmu)
            Wi, Aa, Bg, c1, (F_lo, G_lo, F_hi, G_hi, e, pu, p0) = layer_ops(
                wc_lo, wc_hi, dt, q_lo, q_hi, R, s)
            FR = _mm(F_lo, R)
            Fs = _mv(F_lo, s)
            V = [[e[i] * R[i][j] + pu[i] * (FR[i][j] + G_lo[i][j]) for j in n] for i in n]
            c2 = [e[i] * s[i] + pu[i] * (Fs[i] + q_lo) + p0[i] * q_hi for i in n]
            Z = _mm(V, Wi)
            ZB, ZA = _mm(Z, Bg), _mm(Z, Aa)
            X = [[I[i][j] - p0[i] * F_hi[i][j] - ZB[i][j] for j in n] for i in n]
            Y = [[p0[i] * G_hi[i][j] + ZA[i][j] for j in n] for i in n]
            Xi = _inv(X)
            R_new = _mm(Xi, Y)
            s_new = _mv(Xi, [a + b for a, b in zip(_mv(Z, c1), c2)])
            return (R_new, s_new), (R, s)

        one = jnp.ones_like(b_int)
        zero = jnp.zeros_like(b_int)
        R0 = [[one * I[i][j] for j in range(nmu)] for i in range(nmu)]
        (R_top, s_top), (R_l, s_l) = lax.scan(ric, (R0, [b_int] * nmu), lay3)

        # Optical depth below / above each layer (each summed directly, no cancellation)
        tau_below = jnp.cumsum(dtau, axis=0) - dtau
        tau_above = jnp.cumsum(dtau[::-1], axis=0)[::-1] - dtau
        tau_tot = jnp.sum(dtau, axis=0)

        def source_functions(U, D, wc, q_lev):
            """Level source function at the Hogan angles from the stream intensities."""
            ml = []
            for l in range(nstr):
                m_l = sum(float(wq[j] * Pl[l, j]) * (U[j] + sgn[l] * D[j]) for j in range(nmu))
                ml.append(wc[l] * (0.5 * (2 * l + 1)) * m_l)
            S_up = [q_lev + sum(float(Ph[l, h]) * ml[l] for l in range(nstr)) for h in range(len(_UH))]
            S_dn = [q_lev + sum(sgn[l] * float(Ph[l, h]) * ml[l] for l in range(nstr)) for h in range(len(_UH))]
            return S_up, S_dn

        # ---- Downward sweep, top to bottom, with the source function technique ----
        # d_lo = T d_hi + t (d = 0 at the top), u = R d + s; the level source functions
        # are formed as each level is reached and each layer's Hogan-angle emission is
        # added to the TOA flux, so no per-level intensities are stored.
        @jax.checkpoint
        def down(carry, x):
            d_hi, R_hi, s_hi, S_up_hi, S_dn_hi, acc = carry
            *lay, R_lo, s_lo, ta, tb = x
            wc_lo_k, wc_hi_k, dt, q_lo, q_hi = levels(*lay)
            Wi, Aa, Bg, c1, _ = layer_ops(wc_lo_k, wc_hi_k, dt, q_lo, q_hi, R_lo, s_lo)
            BR = _mm(Bg, R_hi)
            T = _mm(Wi, [[Aa[i][j] + BR[i][j] for j in range(nmu)] for i in range(nmu)])
            t = _mv(Wi, [a + b for a, b in zip(_mv(Bg, s_hi), c1)])
            d_lo = [a + b for a, b in zip(_mv(T, d_hi), t)]
            U_lo = [a + b for a, b in zip(_mv(R_lo, d_lo), s_lo)]
            S_up_lo, S_dn_lo = source_functions(U_lo, d_lo, wc_lo_k, q_lo)

            layer = jnp.zeros_like(dt)
            for h, (uh, wh) in enumerate(zip(_UH, _WH)):
                _, phu, ph0 = _psi(dt, uh)
                src_up = phu * S_up_lo[h] + ph0 * S_up_hi[h]    # emitted up from the layer top
                src_dn = phu * S_dn_hi[h] + ph0 * S_dn_lo[h]    # emitted down from the layer bottom
                t_top = jnp.exp(-ta / uh) * src_up
                t_bot = jnp.exp(-(tb + tau_tot) / uh) * src_dn  # down, reflected, up to TOA
                layer = layer + (np.pi * wh) * (t_top + t_bot)
            return (d_lo, R_lo, s_lo, S_up_lo, S_dn_lo, acc + layer), (layer if return_layer_contrib else None)

        wc_top, q_top = level(layer_props(dtau_in[-1], w_in[-1], g[-1]),
                              (zero, [zero] * nstr), be[-1])
        S_up_top, S_dn_top = source_functions(s_top, [zero] * nmu, wc_top, q_top)
        xs = (*lay3, R_l, s_l, tau_above, tau_below)
        carry0 = ([zero] * nmu, R_top, s_top, S_up_top, S_dn_top, zero)
        (*_, acc), contrib = lax.scan(down, carry0, xs, reverse=True)

        olr = acc + sum((np.pi * wh) * jnp.exp(-tau_tot / uh) * b_int for uh, wh in zip(_UH, _WH))
        if not return_layer_contrib:
            contrib = jnp.zeros_like(dtau)
        return olr, contrib

    return solve_sc_ric


solve_sc_ric = make_sc_ric_solver()
