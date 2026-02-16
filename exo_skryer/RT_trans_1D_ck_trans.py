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

from typing import Dict, Mapping, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from . import build_opacities as XS
from .refraction import refraction_cutoff_mask

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

    sqrt_up = jnp.sqrt(jnp.maximum(r_up_2d**2 - r_mid_2d**2, 0.0))
    sqrt_low = jnp.sqrt(jnp.maximum(r_low_2d**2 - r_mid_2d**2, 0.0))

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
    nlay = state["nlay"]
    nwl = state["nwl"]
    zeros_2d = jnp.zeros((nlay, nwl), dtype=state["rho_lay"].dtype)

    component_keys = ("rayleigh", "cia", "special", "cloud")
    components = jnp.stack([opacity_components.get(k, zeros_2d) for k in component_keys], axis=0)
    return jnp.sum(components, axis=0)


def _integrate_g_points_trans(
    sigma_perspecies: jnp.ndarray,  # (n_species, nlay, nwl, ng)
    vmr_perspecies: jnp.ndarray,    # (n_species, nlay)
    other_opacity_2d: jnp.ndarray,  # (nlay, nwl) - rayleigh, CIA, etc.
    g_points: jnp.ndarray,          # (ng,)
    g_weights: jnp.ndarray,         # (ng,)
    state: Dict[str, jnp.ndarray],
    geometry: tuple[jnp.ndarray, jnp.ndarray],
    refraction_mask: jnp.ndarray | None,
    want_contrib: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Integrate transit depth using RO transmission-function multiplication.

    This computes the slant-path transmission per impact parameter and wavelength as:
        T_total = exp(-tau_cont) * Π_s <exp(-tau_s)>_g
    where <...>_g is the g-quadrature average at fixed (impact parameter, wavelength).
    """
    R0 = state["R0"]
    R_s = state["R_s"]
    rho = state["rho_lay"]
    dz = state["dz"]
    P1D, area_weight = geometry

    nlay = state["nlay"]
    nwl = state["nwl"]

    del g_points

    if want_contrib:
        raise NotImplementedError(
            "Contribution functions are not implemented for ck_mix=TRANS "
            "(RO transmission-product approximation)."
        )

    scale = rho * dz  # (nlay,)

    # Continuum-like opacity (2D) -> slant optical depth (nlay, nwl)
    k_cont = jnp.maximum(other_opacity_2d, 0.0)
    dtau_v_cont = k_cont * scale[:, None]  # (nlay, nwl)
    tau_path_cont = jnp.einsum("ij,jw->iw", P1D, dtau_v_cont)  # (nlay, nwl)

    nspec = sigma_perspecies.shape[0]
    ng = sigma_perspecies.shape[-1]
    w = g_weights[:ng].astype(sigma_perspecies.dtype)

    # Multiply mean transmissions over species at each (impact parameter, wavelength)
    T_prod0 = jnp.ones((nlay, nwl), dtype=sigma_perspecies.dtype)

    def _body(spec_idx: int, T_prod: jnp.ndarray) -> jnp.ndarray:
        kappa_s = sigma_perspecies[spec_idx]  # (nlay, nwl, ng)
        vmr_s = vmr_perspecies[spec_idx]      # (nlay,)

        dtau_v_s = kappa_s * vmr_s[:, None, None] * scale[:, None, None]  # (nlay, nwl, ng)
        tau_path_s = jnp.einsum("ij,jwg->iwg", P1D, dtau_v_s)             # (nlay, nwl, ng)
        tau_path_s = jnp.clip(tau_path_s, 0.0, 300.0)
        T_s_g = jnp.exp(-tau_path_s)                                      # (nlay, nwl, ng)
        T_s = jnp.sum(T_s_g * w[None, None, :], axis=-1)                  # (nlay, nwl)
        return T_prod * jnp.clip(T_s, 1e-99, 1.0)

    # Important: when nspec==0 (e.g. continuum-only diagnostics), tracing a fori_loop
    # would still stage `_body` and attempt `sigma_perspecies[0]`, which is invalid
    # for a zero-length leading dimension.
    if nspec == 0:
        T_prod = T_prod0
    else:
        T_prod = lax.fori_loop(0, nspec, _body, T_prod0)  # (nlay, nwl)

    T_total = jnp.exp(-jnp.clip(tau_path_cont, 0.0, 300.0)) * T_prod  # (nlay, nwl)
    if refraction_mask is not None:
        T_total = jnp.where(refraction_mask, 0.0, T_total)
    one_minus_trans = 1.0 - jnp.clip(T_total, 0.0, 1.0)               # (nlay, nwl)

    dR2 = jnp.sum(area_weight[:, None] * one_minus_trans, axis=0)     # (nwl,)
    D_net = (R0**2 + dR2) / (R_s**2)

    layer_dR2 = jnp.zeros((nlay, nwl), dtype=D_net.dtype)
    return D_net, dR2, layer_dR2


def compute_transit_depth_1d_ck_trans(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
    opac: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute 1D transit depth using transmission multiplication random overlap.

    This function expects per-species opacities in opacity_components:
    - 'line_perspecies': (n_species, nlay, nwl, ng) per-species mass opacities
    - 'vmr_perspecies': (n_species, nlay) volume mixing ratios

    Parameters
    ----------
    state : dict
        Atmospheric state dictionary.
    params : dict
        Parameter dictionary (may contain 'f_cloud').
    opacity_components : dict
        Opacity components including 'line_perspecies', 'vmr_perspecies',
        and optionally 'rayleigh', 'cia', 'special', 'cloud'.

    Returns
    -------
    D_net : array, shape (nwl,)
        Transit depth spectrum.
    contrib_func : array, shape (nlay, nwl)
        Normalized contribution function.
    """
    contri_func = state.get("contri_func", False)
    nlay = state["nlay"]
    nwl = state["nwl"]

    refraction_mask = None
    if int(state.get("refraction_mode", 0)) == 1:
        refraction_mask = refraction_cutoff_mask(state, params, opac)

    geometry = _build_transit_geometry(state)
    g_points, g_weights = _get_ck_quadrature(opac)

    # Get per-species opacities
    sigma_perspecies = opacity_components.get("line_perspecies")
    vmr_perspecies = opacity_components.get("vmr_perspecies")

    if sigma_perspecies is None or vmr_perspecies is None:
        raise ValueError(
            "compute_transit_depth_1d_ck_trans requires 'line_perspecies' and "
            "'vmr_perspecies' in opacity_components. Use ck_mix: trans in config."
        )

    # Sum 2D opacity components
    other_opacity_2d = _sum_opacity_components_2d(state, opacity_components)

    # Handle cloud fraction if present
    if "f_cloud" in params and "cloud" in opacity_components:
        f_cloud = jnp.clip(params["f_cloud"], 0.0, 1.0)
        cloud_component = opacity_components["cloud"]

        # With clouds
        other_with_cloud = other_opacity_2d

        # Without clouds
        other_no_cloud = jnp.maximum(other_opacity_2d - cloud_component, 0.0)

        if contri_func:
            D_cloud, dR2_cloud, layer_dR2_cloud = _integrate_g_points_trans(
                sigma_perspecies, vmr_perspecies, other_with_cloud,
                g_points, g_weights, state, geometry, refraction_mask, want_contrib=True
            )
            D_clear, dR2_clear, layer_dR2_clear = _integrate_g_points_trans(
                sigma_perspecies, vmr_perspecies, other_no_cloud,
                g_points, g_weights, state, geometry, refraction_mask, want_contrib=True
            )

            D_net = f_cloud * D_cloud + (1.0 - f_cloud) * D_clear
            dR2 = f_cloud * dR2_cloud + (1.0 - f_cloud) * dR2_clear
            layer_dR2 = f_cloud * layer_dR2_cloud + (1.0 - f_cloud) * layer_dR2_clear
            contrib_func_norm = layer_dR2 / jnp.maximum(dR2[None, :], 1e-30)
        else:
            D_cloud, _, _ = _integrate_g_points_trans(
                sigma_perspecies, vmr_perspecies, other_with_cloud,
                g_points, g_weights, state, geometry, refraction_mask, want_contrib=False
            )
            D_clear, _, _ = _integrate_g_points_trans(
                sigma_perspecies, vmr_perspecies, other_no_cloud,
                g_points, g_weights, state, geometry, refraction_mask, want_contrib=False
            )

            D_net = f_cloud * D_cloud + (1.0 - f_cloud) * D_clear
            contrib_func_norm = jnp.zeros((nlay, nwl), dtype=D_net.dtype)
    else:
        # No cloud fraction handling
        if contri_func:
            D_net, dR2, layer_dR2 = _integrate_g_points_trans(
                sigma_perspecies, vmr_perspecies, other_opacity_2d,
                g_points, g_weights, state, geometry, refraction_mask, want_contrib=True
            )
            contrib_func_norm = layer_dR2 / jnp.maximum(dR2[None, :], 1e-30)
        else:
            D_net, _, _ = _integrate_g_points_trans(
                sigma_perspecies, vmr_perspecies, other_opacity_2d,
                g_points, g_weights, state, geometry, refraction_mask, want_contrib=False
            )
            contrib_func_norm = jnp.zeros((nlay, nwl), dtype=D_net.dtype)

    return D_net, contrib_func_norm
