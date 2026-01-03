"""
RT_em_schemes.py
================
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

__all__ = ["solve_alpha_eaa", "get_emission_solver"]


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
    name = str(name).lower().strip()
    if name in ("eaa", "alpha_eaa"):
        return solve_alpha_eaa
    raise NotImplementedError(f"Unknown emission scheme '{name}'")
