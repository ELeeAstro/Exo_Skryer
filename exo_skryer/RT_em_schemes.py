"""
RT_em_schemes.py
================

Emission radiative transfer solvers for thermal emission calculations.

This module provides multiple methods for solving the radiative transfer
equation in thermal emission mode:

1. **Alpha-EAA** (solve_alpha_eaa):
   - Single-angle approximation with alpha-EAA scaling
   - Fast and accurate for most cases
   - Supports contribution function calculation

2. **Toon89** (solve_toon89_picaso):
   - Full multi-stream Toon et al. (1989) method
   - 8-stream Gaussian quadrature integration
   - More accurate for optically thick, scattering atmospheres
   - Higher computational cost than EAA

All solvers use the same interface and can be selected via configuration
using the `em_scheme` parameter in the physics section.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax


_MU_NODES = jnp.array([0.0454586727, 0.2322334416, 0.5740198775, 0.9030775973])
_MU_WEIGHTS = jnp.array([0.0092068785, 0.1285704278, 0.4323381850, 0.4298845087])
_N_STREAMS_HALF = 4
nstreams = _N_STREAMS_HALF * 2
_DT_THRESHOLD = 1.0e-4
_DT_SAFE = 1.0e-12

__all__ = ["solve_alpha_eaa", "solve_toon89_picaso", "get_emission_solver"]


@jax.jit
def _tridiag_solve_batched_jax(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """
    Batched Thomas algorithm.

    Solve for X in tri-diagonal systems (per wavelength column):
      a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]

    Shapes
    ------
    a,b,c,d : (L, nwl)
    returns : (L, nwl)
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)
    d = jnp.asarray(d, dtype=jnp.float64)

    tiny = jnp.asarray(1.0e-300, dtype=jnp.float64)

    denom0 = jnp.where(jnp.abs(b[0]) > 0.0, b[0], tiny)
    c0 = c[0] / denom0
    d0 = d[0] / denom0

    def fwd_step(carry, inp):
        c_prev, d_prev = carry
        a_i, b_i, c_i, d_i = inp
        denom = b_i - a_i * c_prev
        denom = jnp.where(jnp.abs(denom) > 0.0, denom, tiny)
        c_i = c_i / denom
        d_i = (d_i - a_i * d_prev) / denom
        return (c_i, d_i), (c_i, d_i)

    inp = (a[1:], b[1:], c[1:], d[1:])
    (_, d_last), (c_hist, d_hist) = lax.scan(fwd_step, (c0, d0), inp)

    c_all = jnp.concatenate([c0[None, :], c_hist], axis=0)
    d_all = jnp.concatenate([d0[None, :], d_hist], axis=0)

    x_last = d_last

    def back_step(x_next, inp2):
        c_i, d_i = inp2
        x_i = d_i - c_i * x_next
        return x_i, x_i

    (_, x_rev_hist) = lax.scan(back_step, x_last, (c_all[:-1][::-1], d_all[:-1][::-1]))
    x = jnp.concatenate([x_rev_hist[::-1], x_last[None, :]], axis=0)
    return x


def _compute_stream_no_contrib(
    mu: jnp.ndarray,
    dtau_a: jnp.ndarray,
    be_levels: jnp.ndarray,
    al: jnp.ndarray,
    be_internal: jnp.ndarray,
    nlev: int,
    nlay: int,
    nwl: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute downward and upward intensities for a single angular stream.

    Parameters
    ----------
    mu : scalar
        Cosine of the zenith angle for this stream.
    dtau_a : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Adjusted optical depth per layer.
    be_levels : `~jax.numpy.ndarray`, shape (nlev, nwl)
        Planck function at level interfaces.
    al : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Planck function differences between levels.
    be_internal : `~jax.numpy.ndarray`, shape (nwl,)
        Internal emission at the bottom boundary.
    nlev, nlay, nwl : int
        Grid dimensions.

    Returns
    -------
    lw_down : `~jax.numpy.ndarray`, shape (nlev, nwl)
        Downward intensity at each level.
    lw_up : `~jax.numpy.ndarray`, shape (nlev, nwl)
        Upward intensity at each level.
    """
    T_trans = jnp.exp(-dtau_a / mu)
    mu_over_dtau = mu / jnp.maximum(dtau_a, _DT_SAFE)

    def down_body(k, lw):
        linear = (
            lw[k] * T_trans[k]
            + be_levels[k + 1]
            - al[k] * mu_over_dtau[k]
            - (be_levels[k] - al[k] * mu_over_dtau[k]) * T_trans[k]
        )
        iso = (
            lw[k] * T_trans[k]
            + 0.5 * (be_levels[k] + be_levels[k + 1]) * (1.0 - T_trans[k])
        )
        next_val = jnp.where(dtau_a[k] > _DT_THRESHOLD, linear, iso)
        return lw.at[k + 1].set(next_val)

    lw_init = jnp.zeros((nlev, nwl), dtype=be_levels.dtype)
    lw_down = lax.fori_loop(0, nlay, down_body, lw_init)

    lw_up_init = jnp.zeros((nlev, nwl), dtype=be_levels.dtype).at[-1].set(
        lw_down[-1] + be_internal
    )

    def up_body(idx, lw):
        k = nlay - 1 - idx
        linear = (
            lw[k + 1] * T_trans[k]
            + be_levels[k]
            + al[k] * mu_over_dtau[k]
            - (be_levels[k + 1] + al[k] * mu_over_dtau[k]) * T_trans[k]
        )
        iso = (
            lw[k + 1] * T_trans[k]
            + 0.5 * (be_levels[k] + be_levels[k + 1]) * (1.0 - T_trans[k])
        )
        I_top = jnp.where(dtau_a[k] > _DT_THRESHOLD, linear, iso)
        return lw.at[k].set(I_top)

    lw_up = lax.fori_loop(0, nlay, up_body, lw_up_init)

    return lw_down, lw_up


def _compute_stream_with_contrib(
    mu: jnp.ndarray,
    weight: jnp.ndarray,
    dtau_a: jnp.ndarray,
    be_levels: jnp.ndarray,
    al: jnp.ndarray,
    be_internal: jnp.ndarray,
    tau_top_layer: jnp.ndarray,
    nlev: int,
    nlay: int,
    nwl: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute intensities and layer contributions for a single angular stream.

    Same as `_compute_stream_no_contrib` but also computes the contribution
    function for each layer.
    """
    T_trans = jnp.exp(-dtau_a / mu)
    mu_over_dtau = mu / jnp.maximum(dtau_a, _DT_SAFE)
    T_toa = jnp.exp(-tau_top_layer / mu)

    def down_body(k, lw):
        linear = (
            lw[k] * T_trans[k]
            + be_levels[k + 1]
            - al[k] * mu_over_dtau[k]
            - (be_levels[k] - al[k] * mu_over_dtau[k]) * T_trans[k]
        )
        iso = (
            lw[k] * T_trans[k]
            + 0.5 * (be_levels[k] + be_levels[k + 1]) * (1.0 - T_trans[k])
        )
        next_val = jnp.where(dtau_a[k] > _DT_THRESHOLD, linear, iso)
        return lw.at[k + 1].set(next_val)

    lw_init = jnp.zeros((nlev, nwl), dtype=be_levels.dtype)
    lw_down = lax.fori_loop(0, nlay, down_body, lw_init)

    lw_up_init = jnp.zeros((nlev, nwl), dtype=be_levels.dtype).at[-1].set(
        lw_down[-1] + be_internal
    )
    layer_contrib_init = jnp.zeros((nlay, nwl), dtype=be_levels.dtype)

    def up_body(idx, carry):
        lw, layer_acc = carry
        k = nlay - 1 - idx

        linear = (
            lw[k + 1] * T_trans[k]
            + be_levels[k]
            + al[k] * mu_over_dtau[k]
            - (be_levels[k + 1] + al[k] * mu_over_dtau[k]) * T_trans[k]
        )
        iso = (
            lw[k + 1] * T_trans[k]
            + 0.5 * (be_levels[k] + be_levels[k + 1]) * (1.0 - T_trans[k])
        )
        I_top = jnp.where(dtau_a[k] > _DT_THRESHOLD, linear, iso)

        source = I_top - lw[k + 1] * T_trans[k]
        layer_acc = layer_acc.at[k].set(weight * source * T_toa[k])
        lw = lw.at[k].set(I_top)
        return (lw, layer_acc)

    lw_up, layer_contrib = lax.fori_loop(0, nlay, up_body, (lw_up_init, layer_contrib_init))

    return lw_down, lw_up, layer_contrib


@jax.jit
def _solve_toon89_picaso_jax(
    be_levels: jnp.ndarray,
    dtau_layers: jnp.ndarray,
    ssa: jnp.ndarray,
    g_phase: jnp.ndarray,
    be_internal: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JAX port of the picaso Toon89 thermal source-function solver.

    This routine integrates over µ using the module-level `_MU_NODES/_MU_WEIGHTS`
    convention in the same way as the original driver.
    """
    be_levels = jnp.asarray(be_levels, dtype=jnp.float64)
    dtau = jnp.asarray(dtau_layers, dtype=jnp.float64)
    w0_in = jnp.asarray(ssa, dtype=jnp.float64)
    g_in = jnp.asarray(g_phase, dtype=jnp.float64)
    be_internal = jnp.asarray(be_internal, dtype=jnp.float64)

    nlev, nwl = be_levels.shape
    nlay = nlev - 1
    mu1 = jnp.asarray(0.5, dtype=jnp.float64)
    pi = jnp.asarray(jnp.pi, dtype=jnp.float64)
    twopi = 2.0 * pi

    b0 = be_levels[:-1, :]
    b1 = (be_levels[1:, :] - b0) / jnp.maximum(dtau, 1.0e-30)

    g1 = 2.0 - w0_in * (1.0 + g_in)
    g2 = w0_in * (1.0 - g_in)
    disc = jnp.maximum(g1**2 - g2**2, 0.0)
    lam = jnp.sqrt(disc)

    eps = 1.0e-30
    g2_safe = jnp.where(jnp.abs(g2) > eps, g2, 1.0)
    gamma_raw = (g1 - lam) / g2_safe
    gamma = jnp.where(jnp.abs(g2) > eps, gamma_raw, 0.0)

    denom_sum = g1 + g2
    denom_sum_safe = jnp.where(jnp.abs(denom_sum) > eps, denom_sum, 1.0)
    g1_plus_g2_raw = 1.0 / denom_sum_safe
    g1_plus_g2 = jnp.where(jnp.abs(denom_sum) > eps, g1_plus_g2_raw, 0.0)

    c_plus_up = twopi * mu1 * (b0 + b1 * g1_plus_g2)
    c_minus_up = twopi * mu1 * (b0 - b1 * g1_plus_g2)
    c_plus_down = twopi * mu1 * (b0 + b1 * dtau + b1 * g1_plus_g2)
    c_minus_down = twopi * mu1 * (b0 + b1 * dtau - b1 * g1_plus_g2)

    exptrm = jnp.minimum(lam * dtau, 35.0)
    exp_pos = jnp.exp(exptrm)
    exp_neg = 1.0 / exp_pos

    tau_top = dtau[0, :] * jnp.exp(-1.0)
    b_top = (1.0 - jnp.exp(-tau_top / mu1)) * be_levels[0, :] * pi
    b_surface = (be_levels[-1, :] + b1[-1, :] * mu1) * pi

    L = 2 * nlay
    e1 = exp_pos + gamma * exp_neg
    e2 = exp_pos - gamma * exp_neg
    e3 = gamma * exp_pos + exp_neg
    e4 = gamma * exp_pos - exp_neg

    A = jnp.zeros((L, nwl), dtype=jnp.float64)
    B = jnp.zeros((L, nwl), dtype=jnp.float64)
    C = jnp.zeros((L, nwl), dtype=jnp.float64)
    D = jnp.zeros((L, nwl), dtype=jnp.float64)

    A = A.at[0].set(0.0)
    B = B.at[0].set(gamma[0] + 1.0)
    C = C.at[0].set(gamma[0] - 1.0)
    D = D.at[0].set(b_top - c_minus_up[0])

    idx = jnp.arange(nlay - 1)
    idx_even = 2 * idx + 1
    idx_odd = 2 * idx + 2

    A_even = (e1[:-1] + e3[:-1]) * (gamma[1:] - 1.0)
    B_even = (e2[:-1] + e4[:-1]) * (gamma[1:] - 1.0)
    C_even = 2.0 * (1.0 - gamma[1:] ** 2)
    D_even = (gamma[1:] - 1.0) * (c_plus_up[1:] - c_plus_down[:-1]) + (1.0 - gamma[1:]) * (
        c_minus_down[:-1] - c_minus_up[1:])

    A = A.at[idx_even].set(A_even)
    B = B.at[idx_even].set(B_even)
    C = C.at[idx_even].set(C_even)
    D = D.at[idx_even].set(D_even)

    A_odd = 2.0 * (1.0 - gamma[:-1] ** 2)
    B_odd = (e1[:-1] - e3[:-1]) * (gamma[1:] + 1.0)
    C_odd = (e1[:-1] + e3[:-1]) * (gamma[1:] - 1.0)
    D_odd = e3[:-1] * (c_plus_up[1:] - c_plus_down[:-1]) + e1[:-1] * (c_minus_down[:-1] - c_minus_up[1:])

    A = A.at[idx_odd].set(A_odd)
    B = B.at[idx_odd].set(B_odd)
    C = C.at[idx_odd].set(C_odd)
    D = D.at[idx_odd].set(D_odd)

    A = A.at[L - 1].set(e1[-1])
    B = B.at[L - 1].set(e2[-1])
    C = C.at[L - 1].set(0.0)
    D = D.at[L - 1].set(b_surface - c_plus_down[-1])

    X = _tridiag_solve_batched_jax(A, B, C, D)  # (L,nwl)
    positive = X[0::2] + X[1::2]  # (nlay,nwl)
    negative = X[0::2] - X[1::2]

    G = (1.0 / mu1 - lam) * positive
    H = gamma * (lam + 1.0 / mu1) * negative
    J = gamma * (lam + 1.0 / mu1) * positive
    K = (1.0 / mu1 - lam) * negative
    alpha1 = twopi * (b0 + b1 * (g1_plus_g2 - mu1))
    alpha2 = twopi * b1
    sigma1 = twopi * (b0 - b1 * (g1_plus_g2 - mu1))
    sigma2 = twopi * b1

    mu_nodes = jnp.asarray(_MU_NODES, dtype=jnp.float64)
    wmu = jnp.asarray(_MU_WEIGHTS, dtype=jnp.float64)/2.0

    be_int = jnp.broadcast_to(be_internal, (nwl,)).astype(jnp.float64)

    def one_mu_step(carry, inp2):
        flx_up_acc, flx_down_acc = carry
        mu, w_mu = inp2

        exp_angle = jnp.exp(-dtau / mu)  # (nlay,nwl)

        # downward recursion (2π*I)
        flux_minus0 = (1.0 - jnp.exp(-tau_top / mu)) * be_levels[0] * twopi # (nwl,)

        def down_body(f_prev, layer_inputs):
            exp_k, lam_k, J_k, K_k, sig1_k, sig2_k, exp_pos_k, exp_neg_k, dtau_k = layer_inputs
            denom_j = lam_k * mu + 1.0
            denom_j = jnp.where(jnp.abs(denom_j) > 1.0e-12, denom_j, 1.0e-12)
            denom_k = lam_k * mu - 1.0
            denom_k = jnp.where(jnp.abs(denom_k) > 1.0e-12, denom_k, 1.0e-12)

            top_val = f_prev * exp_k
            term_J = (J_k / denom_j) * (exp_pos_k - exp_k)
            term_K = (K_k / denom_k) * (exp_k - exp_neg_k)
            term_sig = sig1_k * (1.0 - exp_k)
            term_sig2 = sig2_k * (mu * exp_k + dtau_k - mu)
            f_next = top_val + term_J + term_K + term_sig + term_sig2
            return f_next, f_next

        down_inputs = (
            exp_angle,
            lam,
            J,
            K,
            sigma1,
            sigma2,
            exp_pos,
            exp_neg,
            dtau,
        )
        _, flux_minus_layers = lax.scan(down_body, flux_minus0, down_inputs)
        flux_minus = jnp.concatenate([flux_minus0[None, :], flux_minus_layers], axis=0)  # (nlev,nwl)

        # upward recursion (2π*I)
        flux_plusN = (be_levels[-1] + b1[-1] * mu) * twopi  # (nwl,)

        def up_body(f_next, layer_inputs):
            exp_k, lam_k, G_k, H_k, a1_k, a2_k, exp_pos_k, exp_neg_k, dtau_k = layer_inputs
            denom_g = lam_k * mu - 1.0
            denom_g = jnp.where(jnp.abs(denom_g) > 1.0e-12, denom_g, 1.0e-12)
            denom_h = lam_k * mu + 1.0
            denom_h = jnp.where(jnp.abs(denom_h) > 1.0e-12, denom_h, 1.0e-12)

            bot_val = f_next * exp_k
            term_G = (G_k / denom_g) * (exp_pos_k * exp_k - 1.0)
            term_H = (H_k / denom_h) * (1.0 - exp_neg_k * exp_k)
            term_a1 = a1_k * (1.0 - exp_k)
            term_a2 = a2_k * (mu - (dtau_k + mu) * exp_k)
            f_prev = bot_val + term_G + term_H + term_a1 + term_a2
            return f_prev, f_prev

        up_inputs = (
            exp_angle[::-1],
            lam[::-1],
            G[::-1],
            H[::-1],
            alpha1[::-1],
            alpha2[::-1],
            exp_pos[::-1],
            exp_neg[::-1],
            dtau[::-1],
        )
        _, flux_plus_rev = lax.scan(up_body, flux_plusN, up_inputs)
        flux_plus = jnp.concatenate([flux_plus_rev[::-1], flux_plusN[None, :]], axis=0)  # (nlev,nwl)

        flx_up_acc = flx_up_acc + w_mu * flux_plus
        flx_down_acc = flx_down_acc + w_mu * flux_minus
        return (flx_up_acc, flx_down_acc), None

    init = (jnp.zeros((nlev, nwl), dtype=jnp.float64), jnp.zeros((nlev, nwl), dtype=jnp.float64))
    (flx_up, flx_down), _ = lax.scan(one_mu_step, init, (mu_nodes, wmu))
    # Apply the "lw_net_bottom" scheme:
    #   lw_net = lw_up - lw_down
    #   lw_net(bottom) = π * be_internal   (be_internal is an intensity, e.g. σT_int^4/π)
    desired_net_bottom = pi * jnp.broadcast_to(be_internal, (nwl,))
    net_bottom = flx_up[-1] - flx_down[-1]
    flx_up = flx_up.at[-1].add(desired_net_bottom - net_bottom)
    return flx_up, flx_down


def solve_toon89_picaso(
    be_levels: jnp.ndarray,
    dtau_layers: jnp.ndarray,
    ssa: jnp.ndarray,
    g_phase: jnp.ndarray,
    be_internal: jnp.ndarray,
    return_layer_contrib: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Toon et al. (1989) thermal emission solver with multi-stream quadrature.

    This scheme integrates the radiative transfer equation over angle using
    Gaussian quadrature with 8 streams (4 nodes × 2 hemispheres). It solves
    the full two-stream equations with exponential terms and tridiagonal
    matrix inversion.

    Parameters
    ----------
    be_levels : `~jax.numpy.ndarray`, shape (nlev, nwl)
        Planck function at level interfaces in W m⁻² sr⁻¹.
    dtau_layers : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Optical depth per layer (dimensionless).
    ssa : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Single-scattering albedo per layer (0-1).
    g_phase : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Asymmetry parameter (Henyey-Greenstein) per layer.
    be_internal : `~jax.numpy.ndarray`, shape (nwl,)
        Internal heat flux at bottom boundary in W m⁻² sr⁻¹.
    return_layer_contrib : bool, optional
        If True, return contribution function per layer. Currently not
        implemented for Toon89 scheme (returns zeros).

    Returns
    -------
    lw_up_flux : `~jax.numpy.ndarray`, shape (nlev, nwl)
        Upward longwave flux at each level in W m⁻².
    lw_down_flux : `~jax.numpy.ndarray`, shape (nlev, nwl)
        Downward longwave flux at each level in W m⁻².
    layer_contrib_flux : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Layer contribution function (zeros for Toon89).

    References
    ----------
    Toon et al. (1989), "Rapid calculation of radiative heating rates and
    photodissociation rates in inhomogeneous multiple scattering atmospheres"
    Journal of Geophysical Research, Vol. 94, No. D13, pp. 16,287-16,301.

    Notes
    -----
    The Toon89 scheme is more accurate than the alpha-EAA method for
    optically thick atmospheres with strong scattering, but is also
    more computationally expensive due to the tridiagonal solve per
    wavelength and quadrature integration.
    """
    # Call the core implementation
    lw_up_flux, lw_down_flux = _solve_toon89_picaso_jax(
        be_levels, dtau_layers, ssa, g_phase, be_internal
    )

    # Create dummy contribution function (not implemented for Toon89)
    nlev, nwl = be_levels.shape
    nlay = nlev - 1

    # Return zeros with correct shape for compatibility
    layer_contrib_flux = jnp.zeros((nlay, nwl), dtype=be_levels.dtype)

    return lw_up_flux, lw_down_flux, layer_contrib_flux


def solve_alpha_eaa(
    be_levels: jnp.ndarray,
    dtau_layers: jnp.ndarray,
    ssa: jnp.ndarray,
    g_phase: jnp.ndarray,
    be_internal: jnp.ndarray,
    return_layer_contrib: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    nlev, nwl = be_levels.shape
    nlay = nlev - 1

    be_levels = be_levels.astype(jnp.float64)[::-1]
    dtau_layers = dtau_layers.astype(jnp.float64)[::-1]
    ssa = ssa.astype(jnp.float64)[::-1]
    g_phase = g_phase.astype(jnp.float64)[::-1]

    al = be_levels[1:] - be_levels[:-1]

    mask = g_phase >= 1.0e-4
    fc = jnp.where(mask, g_phase**nstreams, 0.0)
    pmom2 = jnp.where(mask, g_phase**(nstreams + 1), 0.0)
    ratio = jnp.maximum((fc**2) / jnp.maximum(pmom2**2, 1.0e-30), 1.0e-30)
    sigma_sq = jnp.where(mask, ((nstreams + 1) ** 2 - nstreams**2) / jnp.log(ratio), 1.0)
    c = jnp.exp((nstreams**2) / (2.0 * sigma_sq))
    fc_scaled = c * fc

    w_in = jnp.clip(ssa, 0.0, 0.99)
    denom = jnp.maximum(1.0 - fc_scaled * w_in, 1.0e-12)
    w0 = jnp.where(mask, w_in * ((1.0 - fc_scaled) / denom), w_in)
    dtau = jnp.where(mask, (1.0 - w_in * fc_scaled) * dtau_layers, dtau_layers)
    hg = g_phase
    eps = jnp.sqrt((1.0 - w0) * (1.0 - hg * w0))
    dtau_a = eps * dtau

    tau_interface = jnp.concatenate([jnp.zeros((1, nwl), dtype=dtau_a.dtype),
                                     jnp.cumsum(dtau_a, axis=0)], axis=0)
    tau_top_layer = tau_interface[:-1]

    if return_layer_contrib:
        # Vectorize over streams with contribution function
        def stream_fn(mu, weight):
            return _compute_stream_with_contrib(
                mu, weight, dtau_a, be_levels, al, be_internal,
                tau_top_layer, nlev, nlay, nwl
            )

        lw_down_all, lw_up_all, layer_contrib_all = jax.vmap(stream_fn)(
            _MU_NODES, _MU_WEIGHTS
        )
        # Shape: (n_streams, nlev, nwl) -> weighted sum over streams
        lw_down_sum = jnp.sum(lw_down_all * _MU_WEIGHTS[:, None, None], axis=0)
        lw_up_sum = jnp.sum(lw_up_all * _MU_WEIGHTS[:, None, None], axis=0)
        # layer_contrib already weighted inside the function
        layer_contrib_sum = jnp.sum(layer_contrib_all, axis=0)
        layer_contrib_flux = jnp.pi * layer_contrib_sum[::-1]
    else:
        # Vectorize over streams without contribution function
        def stream_fn(mu):
            return _compute_stream_no_contrib(
                mu, dtau_a, be_levels, al, be_internal, nlev, nlay, nwl
            )

        lw_down_all, lw_up_all = jax.vmap(stream_fn)(_MU_NODES)
        # Shape: (n_streams, nlev, nwl) -> weighted sum over streams
        lw_down_sum = jnp.sum(lw_down_all * _MU_WEIGHTS[:, None, None], axis=0)
        lw_up_sum = jnp.sum(lw_up_all * _MU_WEIGHTS[:, None, None], axis=0)
        layer_contrib_flux = jnp.zeros((nlay, nwl), dtype=be_levels.dtype)

    lw_up_flux = jnp.pi * lw_up_sum
    lw_down_flux = jnp.pi * lw_down_sum

    return lw_up_flux, lw_down_flux, layer_contrib_flux


def get_emission_solver(name: str):
    """Get emission RT solver function by name.

    Parameters
    ----------
    name : str
        Scheme name. Supported values:
        - "eaa", "alpha_eaa": Alpha-EAA single-angle approximation
        - "toon89", "toon89_picaso": Toon et al. (1989) multi-stream method

    Returns
    -------
    solver : callable
        Emission solver function with signature:
        (be_levels, dtau_layers, ssa, g_phase, be_internal, return_layer_contrib)
        -> (lw_up_flux, lw_down_flux, layer_contrib_flux)

    Raises
    ------
    NotImplementedError
        If the scheme name is not recognized.
    """
    name = str(name).lower().strip()
    if name in ("eaa", "alpha_eaa"):
        return solve_alpha_eaa
    elif name in ("toon89", "toon89_picaso"):
        return solve_toon89_picaso
    raise NotImplementedError(f"Unknown emission scheme '{name}'")
