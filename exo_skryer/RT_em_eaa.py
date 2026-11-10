"""
RT_em_eaa.py
============

Extended absorption approximation, linear in tau (Li 2002), following
Exo-FMS lw_AA_L. Fast, with approximate scattering (less accurate at
moderate-high albedo).
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from .RT_em_common import _MU_NODES, _MU_WEIGHTS, _sweep, _fused_sweep, _layer_contrib

__all__ = ["solve_alpha_eaa"]

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
        # Layer coefficients built inside the sweeps, one layer at a time: (nmu, nwl).
        mu_s = jnp.asarray(_MU_NODES, dtype=be_levels.dtype)[:, None]

        def down(d, b_top, b_bot):
            T, psi_0, psi_u = _eaa_layer_coeffs(d[None], mu_s)
            return T, psi_u * b_top + psi_0 * b_bot

        def up(d, b_top, b_bot):
            T, psi_0, psi_u = _eaa_layer_coeffs(d[None], mu_s)
            return T, psi_0 * b_top + psi_u * b_bot

        xs = (dtau_a, be_levels[:-1], be_levels[1:])
        I0 = jnp.zeros((mu_s.shape[0], be_levels.shape[1]), dtype=be_levels.dtype)
        I_bot = _fused_sweep(I0, xs, down)
        olr = jnp.pi * (w_q @ _fused_sweep(I_bot + be_internal, xs, up, reverse=True))
        return olr, jnp.zeros((nlay, be_levels.shape[1]), dtype=be_levels.dtype)

    # Contribution functions need every layer's source: layer axis leads, angles
    # are an inner batch axis, (nlay, nmu, nwl).
    mu = jnp.asarray(_MU_NODES, dtype=be_levels.dtype)[None, :, None]
    T, psi_0, psi_u = _eaa_layer_coeffs(dtau_a[:, None], mu)
    src_down = psi_u * be_levels[:-1, None] + psi_0 * be_levels[1:, None]
    src_up = psi_0 * be_levels[:-1, None] + psi_u * be_levels[1:, None]

    # Downward sweep (only its bottom value is needed), then the lower
    # boundary F_int = F_up - F_down, then the upward sweep to TOA.
    I_bot = _sweep(jnp.zeros_like(src_down[0]), T, src_down)
    olr = jnp.pi * (w_q @ _sweep(I_bot + be_internal, T, src_up, reverse=True))
    return olr, jnp.pi * _layer_contrib(w_q, src_up, dtau_a, mu)[::-1]
