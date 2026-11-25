"""
RT_trans_core_os.py
===================

Transmission radiative transfer shared by every geometry and opacity mode, plus the
opacity-sampled (os) entry points.

Shared RT (used by ``RT_trans_core_ck`` too): slant optical depth from a path matrix
(``RT_trans_geometry``), transmission, the area-weighted transit depth, optional
g-weighting, contribution functions and partial-cloud mixing.

Conventions
-----------
- dtau_v: vertical optical depth per cell, kappa * rho * dz, shape (ncell, nwl[, ng]).
- P: path matrix (nray, ncell): chord length / dr.
- area: ray annulus weights (nray,), 2 b db for full annuli (pi cancels).
- D = (R_base^2 + sum_i area_i (1 - T_i)) / R_s^2.

Geometry is chosen in the YAML with ``physics.rt_scheme``: ``transit_1d`` or
``transit_1_5d`` (``transit_2d``/``transit_3d`` are planned).
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Tuple

import jax.numpy as jnp
from jax import lax

from .aux_functions import sum_present
from .RT_trans_geometry import impact_parameters_1d, path_matrix_1d, sector_geometry, shared_ray_grid

__all__ = [
    "vertical_tau",
    "slant_tau",
    "transit_depth",
    "contribution",
    "transit_depth_and_contrib",
    "mix_partial_cloud",
    "transit_depth_1d_from_opacity",
    "sector_dR2",
    "depth_from_dR2",
    "transit_depth_columns",
    "column_dtau_os",
    "compute_transit_depth_1d_os",
]

_K_MIN = 1.0e-99
_TAU_EPS = 1.0e-30
_OS_COMPONENTS = ("line", "rayleigh", "cia", "special", "cloud")


# ---------------------------------------------------------------------------
# Shared radiative transfer
# ---------------------------------------------------------------------------

def vertical_tau(k: jnp.ndarray, rho: jnp.ndarray, dz: jnp.ndarray) -> jnp.ndarray:
    """dtau_v = max(k, 1e-99) * rho * dz, broadcasting rho, dz (ncell,) over trailing axes."""
    extra = (None,) * (k.ndim - 1)
    return jnp.maximum(k, _K_MIN) * rho[(slice(None),) + extra] * dz[(slice(None),) + extra]


def slant_tau(P: jnp.ndarray, dtau_v: jnp.ndarray) -> jnp.ndarray:
    """Slant optical depth per ray: (nray, ncell) x (ncell, ...) -> (nray, ...)."""
    if dtau_v.ndim == 2:
        return jnp.matmul(P, dtau_v)
    return jnp.einsum("ij,j...->i...", P, dtau_v)


def transit_depth(
    one_minus_trans: jnp.ndarray,  # (nray, nwl[, ng])
    area: jnp.ndarray,             # (nray,)
    R_base: jnp.ndarray,
    R_s: jnp.ndarray,
    g_weights: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Transit depth D (nwl,) and dR2 = sum_i area_i (1 - T_i) (nwl,), g-weighted if ck."""
    extra = (None,) * (one_minus_trans.ndim - 1)
    dR2 = jnp.sum(area[(slice(None),) + extra] * one_minus_trans, axis=0)
    D = (R_base**2 + dR2) / (R_s**2)
    if g_weights is not None:
        w = g_weights.astype(D.dtype)
        D = jnp.sum(D * w, axis=-1)
        dR2 = jnp.sum(dR2 * w, axis=-1)
    return D, dR2


def contribution(
    P: jnp.ndarray,
    dtau_v: jnp.ndarray,
    tau: jnp.ndarray,
    one_minus_trans: jnp.ndarray,
    area: jnp.ndarray,
    g_weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Per-cell share of dR2 (ncell, nwl): dtau_v * P^T [area (1 - T) / tau], g-weighted if ck.

    Summed over cells it equals dR2 exactly, since sum_j P_ij dtau_j = tau_i.
    """
    extra = (None,) * (tau.ndim - 1)
    ratio = jnp.where(tau > _TAU_EPS, one_minus_trans / tau, 1.0)
    W = area[(slice(None),) + extra] * ratio
    geom = jnp.matmul(P.T, W) if W.ndim == 2 else jnp.einsum("ji,i...->j...", P.T, W)
    layer = dtau_v * geom
    if g_weights is not None:
        layer = jnp.sum(layer * g_weights.astype(layer.dtype), axis=-1)
    return layer


def transit_depth_and_contrib(
    P: jnp.ndarray,
    area: jnp.ndarray,
    R_base: jnp.ndarray,
    R_s: jnp.ndarray,
    dtau_v: jnp.ndarray,
    g_weights: Optional[jnp.ndarray] = None,
    want_contrib: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    """D (nwl,), dR2 (nwl,) and, if ``want_contrib``, the unnormalised contribution (ncell, nwl)."""
    tau = slant_tau(P, dtau_v)
    one_minus_trans = 1.0 - jnp.exp(-tau)
    D, dR2 = transit_depth(one_minus_trans, area, R_base, R_s, g_weights)
    layer = contribution(P, dtau_v, tau, one_minus_trans, area, g_weights) if want_contrib else None
    return D, dR2, layer


def mix_partial_cloud(
    f_cloud: jnp.ndarray,
    run: Callable[[bool], Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]],
    want_contrib: bool,
    ncell: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Linear mix of a cloudy and a clear run: D = f D_cloud + (1 - f) D_clear.

    ``run(cloudy)`` returns (D, dR2, layer_dR2 or None). Returns (D, normalised contribution).
    """
    f = jnp.clip(f_cloud, 0.0, 1.0)
    D_c, dR2_c, lay_c = run(True)
    D_k, dR2_k, lay_k = run(False)
    D = f * D_c + (1.0 - f) * D_k
    if not want_contrib:
        return D, jnp.zeros((ncell, D.shape[0]), dtype=D.dtype)
    dR2 = f * dR2_c + (1.0 - f) * dR2_k
    layer = f * lay_c + (1.0 - f) * lay_k
    return D, layer / jnp.maximum(dR2[None, :], 1e-30)


def normalised_output(D, dR2, layer, want_contrib: bool, ncell: int):
    """(D, contribution normalised by dR2) or (D, zeros) when contribution is off."""
    if not want_contrib:
        return D, jnp.zeros((ncell, D.shape[0]), dtype=D.dtype)
    return D, layer / jnp.maximum(dR2[None, :], 1e-30)


# ---------------------------------------------------------------------------
# 1D column
# ---------------------------------------------------------------------------

def transit_depth_1d_from_opacity(
    state: Dict[str, jnp.ndarray],
    k_tot: jnp.ndarray,
    want_contrib: bool = False,
    g_weights: Optional[jnp.ndarray] = None,
):
    """D, dR2 and (optionally) the per-layer contribution for one 1D column.

    ``k_tot`` is (nlay, nwl) for os or (nlay, nwl, ng) with ``g_weights`` for ck.
    """
    b, area, R_base = impact_parameters_1d(state)
    P = path_matrix_1d(state, b)
    dtau_v = vertical_tau(k_tot, state["rho_lay"], state["dz"])
    return transit_depth_and_contrib(P, area, R_base, state["R_s"], dtau_v, g_weights, want_contrib)


# ---------------------------------------------------------------------------
# Multi-column atmospheres (sectors x zenith slices)
# ---------------------------------------------------------------------------

def sector_slant_tau(P: jnp.ndarray, dtau_v: jnp.ndarray) -> jnp.ndarray:
    """Slant optical depth for one sector: P (nray, nslice, nlay) x dtau (nslice, nlay, ...) -> (nray, ...)."""
    return jnp.einsum("ikl,kl...->i...", P, dtau_v)


def sector_dR2(
    r_low: jnp.ndarray,         # (nsector, nslice, nlay) layer radii of every column
    r_up: jnp.ndarray,          # (nsector, nslice, nlay)
    theta_edges: jnp.ndarray,   # (nslice + 1,) zenith-slice edges, shared by all sectors
    dtau_fn: Callable[[jnp.ndarray], jnp.ndarray],
    g_weights: Optional[jnp.ndarray] = None,
    want_contrib: bool = False,
):
    """Absorbing area of each sector of a multi-column atmosphere.

    ``dtau_fn(j)`` returns the vertical optical depth of sector j's columns, shape
    (nslice, nlay, nwl[, ng]). It is called inside a ``lax.map`` over sectors (with a
    traced j), so only one sector's columns are in memory at a time. Rays use a grid
    shared by all columns (``shared_ray_grid``); a ray that dips below a column's bottom
    is opaque.

    Returns
    -------
    dR2 : (nsector, nwl[, ng])
        sum_i area_i (1 - T_ij) for each sector j, with full-annulus areas (2 b db), i.e.
        as if sector j covered the whole annulus. Any g axis is kept.
    contrib : (nsector, nslice, nlay, nwl) or None
        Per-cell share of dR2 (g-weighted), if ``want_contrib``.
    R_base : scalar
        Opaque base radius (lowest column bottom).

    The transit depth is (R_base^2 + sum_j f_j dR2_j) / R_s^2 with f_j the azimuthal
    fraction of sector j (then g-weighted for ck). With one sector and one slice covering
    the whole chord this is the 1D calculation.
    """
    b, area, R_base = shared_ray_grid(r_low, r_up)

    def sector(j):
        P, opaque = sector_geometry(b, r_low[j], r_up[j], theta_edges)
        dtau_v = dtau_fn(j)
        tau = sector_slant_tau(P, dtau_v)
        omt = 1.0 - jnp.exp(-tau)
        shape = (-1,) + (1,) * (omt.ndim - 1)
        omt = jnp.where(opaque.reshape(shape), 1.0, omt)
        extra = (None,) * (omt.ndim - 1)
        a = area[(slice(None),) + extra]
        dR2_j = jnp.sum(a * omt, axis=0)
        if not want_contrib:
            return dR2_j, None
        # Per-cell share: dtau * P^T [area (1 - T) / tau] over the non-opaque rays.
        ratio = jnp.where((tau > _TAU_EPS) & ~opaque.reshape(shape), omt / jnp.where(tau > _TAU_EPS, tau, 1.0),
                          jnp.where(opaque.reshape(shape), 0.0, 1.0))
        cell = dtau_v * jnp.einsum("ikl,i...->kl...", P, a * ratio)
        if g_weights is not None:
            cell = jnp.sum(cell * g_weights.astype(cell.dtype), axis=-1)
        return dR2_j, cell

    dR2, contrib = lax.map(sector, jnp.arange(r_low.shape[0]))
    return dR2, contrib, R_base


def depth_from_dR2(dR2: jnp.ndarray, R_base: jnp.ndarray, R_s: jnp.ndarray,
                   g_weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """(R_base^2 + dR2) / R_s^2, g-weighted over the last axis for ck (same order as the 1D path)."""
    D = (R_base**2 + dR2) / (R_s**2)
    if g_weights is not None:
        D = jnp.sum(D * g_weights.astype(D.dtype), axis=-1)
    return D


def transit_depth_columns(
    r_low: jnp.ndarray,         # (nsector, nslice, nlay)
    r_up: jnp.ndarray,          # (nsector, nslice, nlay)
    theta_edges: jnp.ndarray,   # (nslice + 1,)
    sector_frac: jnp.ndarray,   # (nsector,) azimuthal fraction of each sector (sums to 1)
    dtau_fn: Callable[[jnp.ndarray], jnp.ndarray],
    R_s: jnp.ndarray,
    g_weights: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Transit depth D (nwl,) and dR2 (nwl,) of a multi-column atmosphere (see ``sector_dR2``)."""
    dR2_s, _, R_base = sector_dR2(r_low, r_up, theta_edges, dtau_fn, g_weights)
    frac = sector_frac.reshape((-1,) + (1,) * (dR2_s.ndim - 1))
    dR2 = jnp.sum(frac * dR2_s, axis=0)
    D = depth_from_dR2(dR2, R_base, R_s, g_weights)
    if g_weights is not None:
        dR2 = jnp.sum(dR2 * g_weights.astype(dR2.dtype), axis=-1)
    return D, dR2


def column_dtau_os(state: Dict[str, jnp.ndarray], comps: Mapping[str, jnp.ndarray], clear: bool = False):
    """Vertical optical depth (nlay, nwl) of one column; ``clear`` drops the cloud opacity."""
    k = _sum_opacity_components_os(state, comps)
    if clear and "cloud" in comps:
        k = k - comps["cloud"]
    return vertical_tau(k, state["rho_lay"], state["dz"])


# ---------------------------------------------------------------------------
# os single-column entry point (registry: transit_1d/os)
# ---------------------------------------------------------------------------

def _sum_opacity_components_os(
    state: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    """Summed opacity (nlay, nwl) of all components present (zeros if none)."""
    total = sum_present(opacity_components, _OS_COMPONENTS) if opacity_components else None
    if total is None:
        nwl = state["wl"].shape[0] if "wl" in state else int(state["nwl"])
        return jnp.zeros((state["rho_lay"].shape[0], nwl))
    return total


def compute_transit_depth_1d_os(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
    opac: Mapping[str, jnp.ndarray] | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """1D transit depth (nwl,) and normalised contribution function (nlay, nwl)."""
    del opac
    contri_func = state.get("contri_func", False)
    nlay = state["dz"].shape[0]
    k_tot = _sum_opacity_components_os(state, opacity_components)

    if "f_cloud" in params and "cloud" in opacity_components:
        k_clear = k_tot - opacity_components["cloud"]
        return mix_partial_cloud(
            params["f_cloud"],
            lambda cloudy: transit_depth_1d_from_opacity(state, k_tot if cloudy else k_clear, contri_func),
            contri_func, nlay,
        )
    return normalised_output(*transit_depth_1d_from_opacity(state, k_tot, contri_func), contri_func, nlay)
