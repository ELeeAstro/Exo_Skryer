"""
RT_trans_core_ck.py
===================

Correlated-k transmission entry points, for every geometry:

- ``ck``: species already mixed into one k-table (ck_mix RORR / PRAS / LSE / DEEPSET).
  The slant optical depth is summed along each ray at fixed g, and the transit depth
  is the g-weighted mean.
- ``ck_trans``: per-species tables mixed by random overlap in transmission space
  (ck_mix: TRANS), T = exp(-tau_cont) * prod_s sum_g w_g exp(-tau_s,g).

The shared radiative transfer is in ``RT_trans_core_os`` and the geometry in
``RT_trans_geometry``. Geometry is chosen in the YAML with ``physics.rt_scheme``.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import jax.numpy as jnp
from jax import lax

from .aux_functions import sum_present
from .RT_trans_core_os import (
    mix_partial_cloud, normalised_output, sector_slant_tau, slant_tau, transit_depth,
    transit_depth_1d_from_opacity, vertical_tau,
)
from .RT_trans_geometry import impact_parameters_1d, path_matrix_1d, sector_geometry, shared_ray_grid

__all__ = [
    "transit_depth_ro_product",
    "sector_dR2_ro",
    "transit_depth_columns_ro",
    "column_dtau_ck",
    "column_dtau_ck_trans",
    "compute_transit_depth_1d_ck",
    "compute_transit_depth_1d_ck_trans",
]

_CONTINUUM = ("rayleigh", "cia", "special", "cloud")


def _get_ck_weights(opac: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    g_weights = opac.get("g_weights")
    if g_weights is None:
        raise RuntimeError("Missing opac['g_weights'] for c-k integration.")
    return g_weights[0] if g_weights.ndim > 1 else g_weights


def _nwl(state):
    return state["wl"].shape[0] if "wl" in state else int(state["nwl"])


# ---------------------------------------------------------------------------
# Mixed k-tables (ck)
# ---------------------------------------------------------------------------

def _sum_opacity_components_ck(state, opacity_components, opac) -> jnp.ndarray:
    """Total opacity (nlay, nwl, ng): line (nlay, nwl, ng) plus the (nlay, nwl) components broadcast over g."""
    line_opacity = opacity_components.get("line")
    if line_opacity is None:
        ng = _get_ck_weights(opac).shape[-1]
        line_opacity = jnp.zeros((state["rho_lay"].shape[0], _nwl(state), ng))
    summed_2d = sum_present(opacity_components, _CONTINUUM)
    if summed_2d is None:
        return line_opacity
    return line_opacity + summed_2d[:, :, None]


def compute_transit_depth_1d_ck(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
    opac: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """1D transit depth (nwl,) and normalised contribution function (nlay, nwl)."""
    contri_func = state.get("contri_func", False)
    nlay = state["dz"].shape[0]
    k_tot = _sum_opacity_components_ck(state, opacity_components, opac)
    g_weights = _get_ck_weights(opac)[: k_tot.shape[-1]]

    def run(k):
        return transit_depth_1d_from_opacity(state, k, contri_func, g_weights)

    if "f_cloud" in params and "cloud" in opacity_components:
        cloud = opacity_components["cloud"]
        k_clear = k_tot - (cloud[:, :, None] if cloud.ndim == 2 else cloud)
        return mix_partial_cloud(params["f_cloud"], lambda cloudy: run(k_tot if cloudy else k_clear),
                                 contri_func, nlay)
    return normalised_output(*run(k_tot), contri_func, nlay)


# ---------------------------------------------------------------------------
# Per-species random overlap in transmission (ck_trans)
# ---------------------------------------------------------------------------

def transit_depth_ro_product(
    P: jnp.ndarray,
    area: jnp.ndarray,
    R_base: jnp.ndarray,
    R_s: jnp.ndarray,
    dtau_v_cont: jnp.ndarray,       # (ncell, nwl) continuum-like (rayleigh, cia, special, cloud)
    sigma_perspecies: jnp.ndarray,  # (nspec, ncell, nwl, ng) mass opacity per species
    vmr_perspecies: jnp.ndarray,    # (nspec, ncell)
    scale: jnp.ndarray,             # (ncell,) rho * dz
    g_weights: jnp.ndarray,         # (ng,)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """T = exp(-tau_cont) * prod_s sum_g w_g exp(-tau_s,g). Returns D, dR2."""
    nray, nwl = P.shape[0], dtau_v_cont.shape[1]
    nspec, ng = sigma_perspecies.shape[0], sigma_perspecies.shape[-1]
    w = g_weights[:ng].astype(sigma_perspecies.dtype)
    tau_cont = slant_tau(P, dtau_v_cont)

    def _species(spec_idx: int, T_prod: jnp.ndarray) -> jnp.ndarray:
        dtau_s = sigma_perspecies[spec_idx] * vmr_perspecies[spec_idx][:, None, None] * scale[:, None, None]
        tau_s = jnp.clip(slant_tau(P, dtau_s), 0.0, 300.0)
        T_s = jnp.sum(jnp.exp(-tau_s) * w[None, None, :], axis=-1)
        return T_prod * jnp.clip(T_s, 1e-99, 1.0)

    # Unrolled over species (static count); the barrier keeps species sequential so only
    # one (nray, nwl, ng) tau array is live at a time. A fori_loop would fail for nspec == 0.
    T_prod = jnp.ones((nray, nwl), dtype=sigma_perspecies.dtype)
    for spec_idx in range(nspec):
        T_prod = lax.optimization_barrier(_species(spec_idx, T_prod))

    T_total = jnp.exp(-jnp.clip(tau_cont, 0.0, 300.0)) * T_prod
    return transit_depth(1.0 - jnp.clip(T_total, 0.0, 1.0), area, R_base, R_s)


def compute_transit_depth_1d_ck_trans(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
    opac: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """1D transit depth with ck_mix: TRANS. Expects 'line_perspecies' (nspec, nlay, nwl, ng)
    and 'vmr_perspecies' (nspec, nlay). Contribution functions are not available."""
    if state.get("contri_func", False):
        raise NotImplementedError(
            "Contribution functions are not implemented for ck_mix=TRANS "
            "(RO transmission-product approximation)."
        )
    sigma = opacity_components.get("line_perspecies")
    vmr = opacity_components.get("vmr_perspecies")
    if sigma is None or vmr is None:
        raise ValueError(
            "compute_transit_depth_1d_ck_trans requires 'line_perspecies' and "
            "'vmr_perspecies' in opacity_components. Use ck_mix: trans in config."
        )

    b, area, R_base = impact_parameters_1d(state)
    P = path_matrix_1d(state, b)
    scale = state["rho_lay"] * state["dz"]
    g_weights = _get_ck_weights(opac)
    nlay = state["dz"].shape[0]
    cont = sum_present(opacity_components, _CONTINUUM)
    if cont is None:
        cont = jnp.zeros((nlay, _nwl(state)), dtype=state["rho_lay"].dtype)

    def run(k_cont):
        dtau_cont = jnp.maximum(k_cont, 0.0) * scale[:, None]
        return transit_depth_ro_product(P, area, R_base, state["R_s"], dtau_cont, sigma, vmr, scale, g_weights)

    if "f_cloud" in params and "cloud" in opacity_components:
        cont_clear = jnp.maximum(cont - opacity_components["cloud"], 0.0)
        return mix_partial_cloud(params["f_cloud"], lambda cloudy: (*run(cont if cloudy else cont_clear), None),
                                 False, nlay)
    D, _ = run(cont)
    return D, jnp.zeros((nlay, D.shape[0]), dtype=D.dtype)


def sector_dR2_ro(
    r_low: jnp.ndarray,        # (nsector, nslice, nlay)
    r_up: jnp.ndarray,
    theta_edges: jnp.ndarray,  # (nslice + 1,)
    dtau_fn,                   # j -> (dtau_cont (nslice, nlay, nwl), dtau_species (nspec, nslice, nlay, nwl, ng))
    g_weights: jnp.ndarray,
):
    """ck_trans version of ``sector_dR2``: per-sector absorbing area (nsector, nwl) and R_base."""
    b, area, R_base = shared_ray_grid(r_low, r_up)

    def sector(j):
        P, opaque = sector_geometry(b, r_low[j], r_up[j], theta_edges)
        dtau_cont, dtau_spec = dtau_fn(j)
        ng = dtau_spec.shape[-1]
        w = g_weights[:ng].astype(dtau_spec.dtype)
        T_prod = jnp.ones(dtau_cont.shape[-1:], dtype=dtau_spec.dtype)[None, :] * jnp.ones((b.shape[0], 1))
        for s_idx in range(dtau_spec.shape[0]):
            tau_s = jnp.clip(sector_slant_tau(P, dtau_spec[s_idx]), 0.0, 300.0)
            T_prod = lax.optimization_barrier(T_prod * jnp.clip(jnp.sum(jnp.exp(-tau_s) * w, axis=-1), 1e-99, 1.0))
        T = jnp.exp(-jnp.clip(sector_slant_tau(P, dtau_cont), 0.0, 300.0)) * T_prod
        omt = jnp.where(opaque[:, None], 1.0, 1.0 - jnp.clip(T, 0.0, 1.0))
        return jnp.sum(area[:, None] * omt, axis=0)

    return lax.map(sector, jnp.arange(r_low.shape[0])), R_base


def transit_depth_columns_ro(
    r_low: jnp.ndarray,
    r_up: jnp.ndarray,
    theta_edges: jnp.ndarray,
    sector_frac: jnp.ndarray,
    dtau_fn,
    R_s: jnp.ndarray,
    g_weights: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """ck_trans transit depth of a multi-column atmosphere (see ``sector_dR2_ro``)."""
    dR2_s, R_base = sector_dR2_ro(r_low, r_up, theta_edges, dtau_fn, g_weights)
    dR2 = jnp.sum(sector_frac[:, None] * dR2_s, axis=0)
    return (R_base**2 + dR2) / (R_s**2), dR2


def column_dtau_ck(state, comps, opac, clear: bool = False):
    """Vertical optical depth (nlay, nwl, ng) of one column; ``clear`` drops the cloud opacity."""
    k = _sum_opacity_components_ck(state, comps, opac)
    if clear and "cloud" in comps:
        cloud = comps["cloud"]
        k = k - (cloud[:, :, None] if cloud.ndim == 2 else cloud)
    return vertical_tau(k, state["rho_lay"], state["dz"])


def column_dtau_ck_trans(state, comps, clear: bool = False):
    """(continuum dtau (nlay, nwl), per-species dtau (nspec, nlay, nwl, ng)) of one column."""
    scale = state["rho_lay"] * state["dz"]
    cont = sum_present(comps, _CONTINUUM)
    if cont is None:
        cont = jnp.zeros((state["rho_lay"].shape[0], _nwl(state)), dtype=state["rho_lay"].dtype)
    if clear and "cloud" in comps:
        cont = cont - comps["cloud"]
    dcont = jnp.maximum(cont, 0.0) * scale[:, None]
    dspec = comps["line_perspecies"] * comps["vmr_perspecies"][:, :, None, None] * scale[None, :, None, None]
    return dcont, dspec
