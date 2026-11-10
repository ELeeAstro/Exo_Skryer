"""
RT_em_ad.py
===========

Adding-doubling (matrix operator) method for thermal emission (Plass, Kattawar
& Catchings 1973; Hansen 1971), following the reference AddingDoublingRT
solver (src_jax/doubling.py, adding.py, phase_matrix.py) and Exo-FMS lw_AD.
Gauss-Legendre streams (``nmu`` per hemisphere, 2*nmu-stream method); delta-M
scaled Henyey-Greenstein phase function with f = g**(2 nmu) and
Hansen-normalised phase matrices; Planck function linear in tau inside each
layer.

Each homogeneous layer's reflection R, transmission T and thermal sources are
built by doubling from an optically thin sub-layer, then layers are added from
the bottom up. As in the reference:

- Non-scattering layers (w = 0) use the exact absorption/emission solution.
- Scattering layers start from first-order (single scattering) sub-layer
  R = tau0 Gpm, T = I - tau0 Gpp, y = (1 - w) tau0 / mu, z = 0, doubled
  nn = int(log2(tau)) + ipow0(w) times, with ipow0 = 4, 10, 16 for
  w < 0.01, < 0.1, >= 0.1.

Differences from the reference, for the JAX layout:

- Only the upward adding pass is done: Exo Skryer needs the TOA flux (and the
  per-layer contributions to it), not the flux profile. There is no incident
  intensity at the top, and the lower boundary is I_up = I_down + B_int in
  every stream (as RT_em_dda).
- The doubling count must be static under ``jit``/``grad``, so every layer
  runs ``n_max`` doubling steps, masked off once it reaches its own ``nn``.
  ``nn`` is clipped to ``n_max``; a deeper layer then starts from a thicker
  sub-layer (no opaque cap: the layer is always doubled to its full depth).
- Layers with tau <= 1e-10 are treated as transparent (gradient safety).
- The (2 nmu x 2 nmu) algebra is unrolled into elementwise operations on
  (nlay, nwl) arrays, which XLA fuses into large vector loops; the small
  systems are diagonally dominant (I - R R' with ||R|| < 1), so no pivoting
  is needed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["solve_ad", "make_ad_solver"]

_DTAU_MIN = 1.0e-10

# Defaults: 8 streams (4 per hemisphere). n_max = 32 doublings covers the reference
# doubling count for scattering layers up to tau = 2**16 (w >= 0.1), 2**22 (w >= 0.01)
# and 2**28 (w < 0.01); deeper layers start from a thicker sub-layer.
_NMU = 4
_N_MAX = 32

Mat = List[List[jnp.ndarray]]
Vec = List[jnp.ndarray]


# ---------------------------------------------------------------------------
# Unrolled small-matrix algebra on nested lists of arrays. Entries may be
# Python floats (structural zeros/ones), which XLA then folds away.
# ---------------------------------------------------------------------------
def _eye(n: int) -> Mat:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _mm(A: Mat, B: Mat) -> Mat:
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def _mv(A: Mat, v: Vec) -> Vec:
    n = len(A)
    return [sum(A[i][k] * v[k] for k in range(n)) for i in range(n)]


def _madd(A: Mat, B: Mat, a: float = 1.0, b: float = 1.0) -> Mat:
    return [[a * x + b * y for x, y in zip(ra, rb)] for ra, rb in zip(A, B)]


def _vadd(u: Vec, v: Vec) -> Vec:
    return [x + y for x, y in zip(u, v)]


def _inv(A: Mat) -> Mat:
    """Gauss-Jordan inverse without pivoting (diagonally dominant systems only)."""
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


def _where(c, a, b):
    """Elementwise select over matching pytrees (lists of arrays)."""
    return jax.tree_util.tree_map(lambda x, y: jnp.where(c, x, y), a, b)


# ---------------------------------------------------------------------------
# Static quadrature and phase-function tables
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _tables(nmu: int):
    """Gauss-Legendre nodes/weights on [0, 1] (sum(wt) = 1) and Legendre products.

    PP[l, i, j] = (2l+1) P_l(mu_i) P_l(mu_j); PM = PP (-1)^l (opposite hemisphere).
    The 1/(2 pi) of the azimuthal average cancels in the Hansen normalisation.
    """
    z, wz = np.polynomial.legendre.leggauss(nmu)
    mu = 0.5 * (z + 1.0)
    wt = 0.5 * wz
    nmom = 2 * nmu
    P = np.zeros((nmom, nmu))
    P[0] = 1.0
    P[1] = mu
    for l in range(2, nmom):
        P[l] = ((2 * l - 1) * mu * P[l - 1] - (l - 1) * P[l - 2]) / l
    c2l = 2.0 * np.arange(nmom) + 1.0
    PP = c2l[:, None, None] * P[:, :, None] * P[:, None, :]
    PM = PP * ((-1.0) ** np.arange(nmom))[:, None, None]
    return mu, wt, PP, PM


def _scattering_operators(w0, chi, nmu: int) -> Tuple[Mat, Mat]:
    """Single-scattering operators per unit tau, Gpp = (I - w Qpp C)/mu, Gpm = w Qpm C/mu.

    Q are the azimuth-averaged phase matrices, Hansen-normalised so each incident
    stream j scatters exactly w of its energy: sum_i wt_i (Qpp + Qpm)_ij = 1
    (reference compute_phase_matrices with con = 2 pi w).
    """
    mu, wt, PP, PM = _tables(nmu)
    nmom = 2 * nmu

    def phase(tab, i, j):
        return sum(float(tab[l, i, j]) * chi[l] for l in range(nmom))

    Ppp = [[phase(PP, i, j) for j in range(nmu)] for i in range(nmu)]
    Ppm = [[phase(PM, i, j) for j in range(nmu)] for i in range(nmu)]

    Gpp = [[None] * nmu for _ in range(nmu)]
    Gpm = [[None] * nmu for _ in range(nmu)]
    for j in range(nmu):
        colsum = sum(float(wt[i]) * (Ppp[i][j] + Ppm[i][j]) for i in range(nmu))
        ok = colsum > 0.0
        scale = w0 * float(wt[j]) / jnp.where(ok, colsum, 1.0)
        for i in range(nmu):
            r = 1.0 / float(mu[i])
            Gpp[i][j] = ((1.0 if i == j else 0.0) - scale * Ppp[i][j]) * r
            Gpm[i][j] = scale * Ppm[i][j] * r
    return Gpp, Gpm


# ---------------------------------------------------------------------------
# Doubling: R, T and thermal source basis (y, z) of each homogeneous layer.
# A layer emits y B_bar + z B_d up from its top and y B_bar - z B_d down from its
# bottom, for B linear in tau with mean B_bar and slope B_d (per unit tau, downward).
# ---------------------------------------------------------------------------
def _double(R, T, y, z, g, nmu: int):
    """Two identical layers of half-depth g combined into one of half-depth 2g."""
    I = _eye(nmu)
    TG = _mm(T, _inv(_madd(I, _mm(R, R), 1.0, -1.0)))   # T (I - R^2)^-1
    TGR = _mm(TG, R)
    zpgy = [zi + g * yi for zi, yi in zip(z, y)]
    z_new = [a + zi - g * yi for a, zi, yi in zip(_mv(_madd(TG, TGR, 1.0, -1.0), zpgy), z, y)]
    y_new = _vadd(_mv(_madd(TG, TGR), y), y)
    R_new = _madd(R, _mm(TGR, T))
    T_new = _mm(TG, T)
    return R_new, T_new, y_new, z_new


def _layer_doubling(dtau, w0, chi, nmu: int, n_max: int):
    """R, T, y, z of each scattering layer (reference doubling())."""
    mu, _, _, _ = _tables(nmu)
    I = _eye(nmu)

    # nn = int(log2(tau)) + ipow0(w), at least 1 (reference). Integer choices carry no
    # gradient; int() truncates toward zero, as in the reference.
    tau_ng = lax.stop_gradient(dtau)
    w_ng = lax.stop_gradient(w0)
    ipow0 = jnp.where(w_ng < 0.01, 4.0, jnp.where(w_ng < 0.1, 10.0, 16.0))
    nn = jnp.clip(jnp.trunc(jnp.log2(tau_ng)) + ipow0, 1, n_max)
    tau0 = dtau * jnp.exp2(-nn)

    Gpp, Gpm = _scattering_operators(w0, chi, nmu)

    # First-order (single scattering) starting sub-layer
    R = [[tau0 * Gpm[i][j] for j in range(nmu)] for i in range(nmu)]
    T = [[I[i][j] - tau0 * Gpp[i][j] for j in range(nmu)] for i in range(nmu)]
    y = [(1.0 - w0) * tau0 / float(mu[i]) for i in range(nmu)]
    z = [jnp.zeros_like(dtau) for _ in range(nmu)]

    def step(carry, n):
        R, T, y, z, g = carry
        new = (*_double(R, T, y, z, g, nmu), 2.0 * g)
        return _where(n < nn, new, carry), None

    init = (R, T, y, z, 0.5 * tau0)
    (R, T, y, z, _), _ = lax.scan(step, init, jnp.arange(n_max, dtype=nn.dtype))
    return R, T, y, z


def _layer_rt(dtau, w0, B_top, B_bot, chi, nmu, n_max):
    """R, T and the thermal sources emitted up from the top (sup) and down from the bottom (sdn)."""
    mu, _, _, _ = _tables(nmu)
    I = _eye(nmu)
    zero = jnp.zeros_like(dtau)

    thin = dtau <= _DTAU_MIN
    dtau = jnp.where(thin, 1.0, dtau)
    absorb = w0 <= 0.0

    B_bar = 0.5 * (B_top + B_bot)
    B_d = (B_bot - B_top) / dtau

    # Scattering layers: doubling (absorbing layers get a dummy depth, overwritten below)
    R, T, y, z = _layer_doubling(jnp.where(absorb, 1.0, dtau), w0, chi, nmu, n_max)
    sup = [yi * B_bar + zi * B_d for yi, zi in zip(y, z)]
    sdn = [yi * B_bar - zi * B_d for yi, zi in zip(y, z)]

    # Non-scattering layers: exact (reference pure-absorption branch; 1 - t via expm1)
    t_a, sup_a, sdn_a = [], [], []
    for i in range(nmu):
        m = float(mu[i])
        tex = jnp.maximum(-dtau / m, -200.0)
        t = jnp.where(-dtau / m > -200.0, jnp.exp(tex), 0.0)
        omt = -jnp.expm1(tex)
        slope = m * omt - 0.5 * dtau * (1.0 + t)
        t_a.append(t)
        sup_a.append(B_bar * omt + B_d * slope)
        sdn_a.append(B_bar * omt - B_d * slope)

    R = [[jnp.where(absorb, 0.0, x) for x in r] for r in R]
    T = [[jnp.where(absorb, t_a[i] if i == j else 0.0, x) for j, x in enumerate(r)]
         for i, r in enumerate(T)]
    sup = [jnp.where(absorb, a, x) for a, x in zip(sup_a, sup)]
    sdn = [jnp.where(absorb, a, x) for a, x in zip(sdn_a, sdn)]

    # Optically negligible layers are transparent
    R = [[jnp.where(thin, 0.0, x) for x in r] for r in R]
    T = [[jnp.where(thin, I[i][j], x) for j, x in enumerate(r)] for i, r in enumerate(T)]
    sup = [jnp.where(thin, zero, x) for x in sup]
    sdn = [jnp.where(thin, zero, x) for x in sdn]
    return R, T, sup, sdn


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def make_ad_solver(nmu: int = _NMU, n_max: int = _N_MAX) -> Callable:
    """Build an adding-doubling emission solver.

    Parameters
    ----------
    nmu : int
        Gauss-Legendre streams per hemisphere (2*nmu-stream method).
    n_max : int
        Static number of doubling steps (cost); each layer uses
        min(int(log2(tau)) + ipow0(w), n_max) of them.
    """
    if n_max < 1:
        raise ValueError("make_ad_solver: n_max must be >= 1.")
    mu, wt, _, _ = _tables(nmu)
    nmom = 2 * nmu
    wmu = [float(2.0 * np.pi * w * m) for w, m in zip(wt, mu)]


    def solve_ad(
        be_levels: jnp.ndarray,
        dtau_layers: jnp.ndarray,
        ssa: jnp.ndarray,
        g_phase: jnp.ndarray,
        be_internal: jnp.ndarray,
        return_layer_contrib: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Adding-doubling method (reference AddingDoublingRT / Exo-FMS lw_AD).

        Inputs are bottom-first, ``be_levels`` (nlev, nwl) and the rest (nlay, nwl).
        Returns the TOA upward flux ``olr`` (nwl,) and a bottom-first layer
        contribution (nlay, nwl), zeros unless ``return_layer_contrib``.
        """
        be = jnp.asarray(be_levels)
        dtype = be.dtype
        dtau = jnp.asarray(dtau_layers, dtype=dtype)
        g = jnp.asarray(g_phase, dtype=dtype)
        w0 = jnp.clip(jnp.asarray(ssa, dtype=dtype), 0.0, 1.0)

        # Henyey-Greenstein moments g^l, delta-M scaled with the first truncated moment g^(2 nmu)
        f = g ** nmom
        f = jnp.where((w0 > 0.0) & (f > 1.0e-12) & (f < 1.0 - 1.0e-12), f, 0.0)
        rf = 1.0 / (1.0 - f)
        chi = [jnp.ones_like(g)] + [(g ** l - f) * rf for l in range(1, nmom)]
        dtau = (1.0 - w0 * f) * dtau
        w0 = w0 * (1.0 - f) / (1.0 - w0 * f)

        # Upward adding from the bottom; lower boundary I_up = I_down + B_int in every stream.
        # Rb: reflection of the slab below (seen from above), Su: upward intensity it emits.
        # Each layer is doubled inside its own step (bottom-first: the top of layer k is
        # level k + 1), so only one layer's R, T and sources exist at a time. The step is
        # rematerialised under reverse-mode AD: only the carry and the layer inputs are kept
        # as residuals, not every doubling step.
        I = _eye(nmu)

        @jax.checkpoint
        def add(carry, x):
            Rb, Su = carry
            R_k, T_k, su_k, sd_k = _layer_rt(*x, nmu, n_max)
            A = _mm(T_k, _inv(_madd(I, _mm(Rb, R_k), 1.0, -1.0)))   # T (I - Rb R)^-1
            src = _vadd(su_k, _mv(A, _mv(Rb, sd_k)))
            carry = (_madd(R_k, _mm(A, _mm(Rb, T_k))), _vadd(src, _mv(A, Su)))
            return carry, ((A, src) if return_layer_contrib else None)

        one = jnp.ones_like(be[0])
        b_int = jnp.broadcast_to(jnp.asarray(be_internal, dtype=dtype), be[0].shape)
        Rb0 = [[one * I[i][j] for j in range(nmu)] for i in range(nmu)]
        xs = (dtau, w0, be[1:], be[:-1], chi)
        (_, Su_top), per_layer = lax.scan(add, (Rb0, [b_int] * nmu), xs)

        # No incident intensity at the top, so I_up(TOA) = Su_top
        olr = sum(c * s for c, s in zip(wmu, Su_top))

        if not return_layer_contrib:
            return olr, jnp.zeros_like(dtau)

        # Each layer's emission carried to TOA through the composite transfer of the layers above
        def to_top(u, x):
            A_k, s_k = x
            u_new = [sum(u[i] * A_k[i][j] for i in range(nmu)) for j in range(nmu)]
            return u_new, sum(ui * si for ui, si in zip(u, s_k))

        _, layer_contrib = lax.scan(to_top, [c * one for c in wmu], per_layer, reverse=True)
        return olr, layer_contrib

    return solve_ad


solve_ad = make_ad_solver()
