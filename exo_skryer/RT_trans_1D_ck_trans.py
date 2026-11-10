"""
RT_trans_1D_ck_trans.py
=======================

Transit transmission spectrum calculation using the transmission multiplication
random overlap method for correlated-k species mixing.

This module differs from RT_trans_1D_ck.py in that species are combined by
multiplying their mean transmissions under the random-overlap assumption:

    T_total = exp(-tau_cont) * Π_s [ Σ_g w_g exp(-tau_s(g)) ]

This avoids ROM sorting / k-distribution mixing entirely and is intended as a
fast transmission-only approximation.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from .aux_functions import safe_sqrt
from .refraction import maybe_refraction_cutoff_mask
from .RT_trans_1D_os import _get_base_transit_radius
from .opacity_ck import make_ck_species_kappa

__all__ = ["compute_transit_depth_1d_ck_trans"]


def _get_ck_quadrature(opac):
    """Extract g-points and weights from opac cache."""
    g_points_all = opac["g_points"]
    g_weights = opac["g_weights"]

    if g_points_all.ndim == 1:
        g_points = g_points_all
    else:
        g_points = g_points_all[0]

    if g_weights.ndim > 1:
        g_weights = g_weights[0]

    return g_points, g_weights


def _build_transit_geometry(state: Dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute geometry terms for transit depth calculation."""
    R0 = state["R0"]
    z_lev = state["z_lev"]
    z_lay = state["z_lay"]

    r_mid = R0 + z_lay
    r_low = R0 + z_lev[:-1]
    r_up = R0 + z_lev[1:]
    dr = r_up - r_low

    r_mid_2d = r_mid[:, None]
    r_up_2d = r_up[None, :]
    r_low_2d = r_low[None, :]
    dr_2d = dr[None, :]

    sqrt_up = safe_sqrt(r_up_2d**2 - r_mid_2d**2)
    sqrt_low = safe_sqrt(r_low_2d**2 - r_mid_2d**2)

    P_case1 = jnp.zeros_like(sqrt_up)
    P_case2 = 2.0 / dr_2d * sqrt_up
    P_case3 = 2.0 / dr_2d * (sqrt_up - sqrt_low)

    cond1 = r_up_2d <= r_mid_2d
    cond2 = (r_low_2d <= r_mid_2d) & (r_mid_2d < r_up_2d)

    P1D = jnp.where(cond1, P_case1, jnp.where(cond2, P_case2, P_case3))
    area_weight = 2.0 * r_mid * dr

    return P1D, area_weight


def _sum_opacity_components_2d(
    state: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    """Sum 2D opacity components (rayleigh, cia, special, cloud).

    Returns shape (nlay, nwl).
    """
    nlay = state["rho_lay"].shape[0]
    nwl = state["wl"].shape[0] if "wl" in state else int(state["nwl"])
    zeros_2d = jnp.zeros((nlay, nwl), dtype=state["rho_lay"].dtype)

    component_keys = ("rayleigh", "cia", "special", "cloud")
    components = jnp.stack([opacity_components.get(k, zeros_2d) for k in component_keys], axis=0)
    return jnp.sum(components, axis=0)


def _species_transmission(
    kappa_of: Callable[[jnp.ndarray], jnp.ndarray],
    nspec: int,
    vmr_perspecies: jnp.ndarray,    # (n_species, nlay)
    g_weights: jnp.ndarray,         # (ng,)
    state: Dict[str, jnp.ndarray],
    P1D: jnp.ndarray,               # (nlay, nlay) slant path lengths
    nwl: int,
) -> jnp.ndarray:
    """Product over species of the g-averaged slant transmission, (nlay, nwl).

    T_prod = Π_s Σ_g w_g exp(-tau_s(g)), at each impact parameter and wavelength.
    ``kappa_of(s)`` interpolates species s's mass opacity (nlay, nwl, ng) on demand,
    so only one species is held at a time. The loop body is checkpointed: under
    reverse-mode AD it is recomputed instead of storing its (nlay, nwl, ng)
    intermediates for every species.
    """
    scale = state["rho_lay"] * state["dz"]  # (nlay,)
    nlay = scale.shape[0]

    def _body(spec_idx, T_prod):
        kappa_s = kappa_of(spec_idx)                                      # (nlay, nwl, ng)
        vmr_s = vmr_perspecies[spec_idx]                                  # (nlay,)
        w = g_weights[: kappa_s.shape[-1]].astype(kappa_s.dtype)

        dtau_v_s = kappa_s * vmr_s[:, None, None] * scale[:, None, None]  # (nlay, nwl, ng)
        tau_path_s = jnp.einsum("ij,jwg->iwg", P1D, dtau_v_s)             # (nlay, nwl, ng)
        tau_path_s = jnp.clip(tau_path_s, 0.0, 300.0)
        T_s = jnp.sum(jnp.exp(-tau_path_s) * w[None, None, :], axis=-1)   # (nlay, nwl)
        return T_prod * jnp.clip(T_s, 1e-99, 1.0)

    _body = jax.checkpoint(_body)

    T_prod0 = jnp.ones((nlay, nwl), dtype=scale.dtype)
    # With nspec == 0 (continuum-only diagnostics) tracing the loop would index an
    # empty leading dimension, so skip it.
    if nspec == 0:
        return T_prod0
    return lax.fori_loop(0, nspec, _body, T_prod0)


def _transit_depth_from_transmission(
    T_prod: jnp.ndarray,            # (nlay, nwl) species transmission product
    other_opacity_2d: jnp.ndarray,  # (nlay, nwl) - rayleigh, CIA, cloud, etc.
    state: Dict[str, jnp.ndarray],
    geometry: tuple[jnp.ndarray, jnp.ndarray],
    refraction_mask: jnp.ndarray | None,
) -> jnp.ndarray:
    """Transit depth with the continuum added: T_total = exp(-tau_cont) * T_prod."""
    R_base = _get_base_transit_radius(state)
    R_s = state["R_s"]
    P1D, area_weight = geometry

    scale = state["rho_lay"] * state["dz"]
    dtau_v_cont = jnp.maximum(other_opacity_2d, 0.0) * scale[:, None]  # (nlay, nwl)
    tau_path_cont = jnp.einsum("ij,jw->iw", P1D, dtau_v_cont)          # (nlay, nwl)

    T_total = jnp.exp(-jnp.clip(tau_path_cont, 0.0, 300.0)) * T_prod
    if refraction_mask is not None:
        T_total = T_total * (1.0 - refraction_mask)
    one_minus_trans = 1.0 - jnp.clip(T_total, 0.0, 1.0)               # (nlay, nwl)

    dR2 = jnp.sum(area_weight[:, None] * one_minus_trans, axis=0)     # (nwl,)
    return (R_base**2 + dR2) / (R_s**2)


def compute_transit_depth_1d_ck_trans(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
    opac: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute 1D transit depth using transmission multiplication random overlap.

    Expects 'vmr_perspecies' (n_species, nlay) in ``opacity_components``. Each
    species' opacity is interpolated from the k-tables in ``opac`` inside the
    species loop, so only one species' (nlay, nwl, ng) opacities are held at a
    time. A species with zero VMR contributes a transmission of exactly 1 (used
    to isolate single species for component spectra).

    The species transmission product does not depend on the cloud, so it is
    computed once and shared by the cloudy and clear columns when 'f_cloud' is
    present.

    Parameters
    ----------
    state : dict
        Atmospheric state dictionary.
    params : dict
        Parameter dictionary (may contain 'f_cloud').
    opacity_components : dict
        Opacity components, see above, plus optionally 'rayleigh', 'cia',
        'special', 'cloud'.
    opac : dict
        Runtime opacity tables (k-tables and g-quadrature).

    Returns
    -------
    D_net : array, shape (nwl,)
        Transit depth spectrum.
    contrib_func : array, shape (nlay, nwl)
        Zeros: contribution functions are not implemented for ck_mix=TRANS.
    """
    if state.get("contri_func", False):
        raise NotImplementedError(
            "Contribution functions are not implemented for ck_mix=TRANS "
            "(RO transmission-product approximation)."
        )
    refraction_mask = maybe_refraction_cutoff_mask(state, params, opac)

    geometry = _build_transit_geometry(state)
    _, g_weights = _get_ck_quadrature(opac)

    vmr_perspecies = opacity_components.get("vmr_perspecies")
    if vmr_perspecies is None:
        raise ValueError(
            "compute_transit_depth_1d_ck_trans requires 'vmr_perspecies' in "
            "opacity_components. Use ck_mix: trans in config."
        )
    nspec = vmr_perspecies.shape[0]
    kappa_of = make_ck_species_kappa(state, opac)

    other_opacity_2d = _sum_opacity_components_2d(state, opacity_components)
    nwl = other_opacity_2d.shape[1]

    T_prod = _species_transmission(
        kappa_of, nspec, vmr_perspecies, g_weights, state, geometry[0], nwl
    )

    if "f_cloud" in params and "cloud" in opacity_components:
        f_cloud = jnp.clip(params["f_cloud"], 0.0, 1.0)
        other_no_cloud = jnp.maximum(other_opacity_2d - opacity_components["cloud"], 0.0)
        D_cloud = _transit_depth_from_transmission(T_prod, other_opacity_2d, state, geometry, refraction_mask)
        D_clear = _transit_depth_from_transmission(T_prod, other_no_cloud, state, geometry, refraction_mask)
        D_net = f_cloud * D_cloud + (1.0 - f_cloud) * D_clear
    else:
        D_net = _transit_depth_from_transmission(T_prod, other_opacity_2d, state, geometry, refraction_mask)

    contrib_func_norm = jnp.zeros((state["dz"].shape[0], D_net.shape[0]), dtype=D_net.dtype)
    return D_net, contrib_func_norm
