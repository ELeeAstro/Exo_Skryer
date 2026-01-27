"""
RT_trans_1D_lbl.py
==================
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import jax.numpy as jnp

from .refraction import refraction_cutoff_mask

__all__ = ["compute_transit_depth_1d_lbl"]


def _sum_opacity_components_lbl(
    state: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    """Return the summed opacity grid for all provided components."""
    nlay = state["nlay"]
    nwl = state["nwl"]

    if not opacity_components:
        return jnp.zeros((nlay, nwl))

    component_keys = ("line", "rayleigh", "cia", "special", "cloud")
    first = next((opacity_components.get(k) for k in component_keys if k in opacity_components), None)
    if first is None:
        return jnp.zeros((nlay, nwl))

    zeros = jnp.zeros_like(first)
    stacked = jnp.stack([opacity_components.get(k, zeros) for k in component_keys], axis=0)
    return jnp.sum(stacked, axis=0)


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


def _transit_depth_and_contrib_from_opacity(
    state: Dict[str, jnp.ndarray],
    k_tot: jnp.ndarray,  # (nlay, nwl)
    geometry: tuple[jnp.ndarray, jnp.ndarray],
    refraction_mask: jnp.ndarray | None,
    want_contrib: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    R0 = state["R0"]
    R_s = state["R_s"]
    rho = state["rho_lay"]
    dz = state["dz"]
    P1D, area_weight = geometry

    k_eff = jnp.maximum(k_tot, 1.0e-99)
    dtau_v = k_eff * rho[:, None] * dz[:, None]
    tau_path = jnp.matmul(P1D, dtau_v)
    if refraction_mask is not None:
        tau_path = jnp.where(refraction_mask, 1.0e30, tau_path)

    one_minus_trans = 1.0 - jnp.exp(-tau_path)
    dR2_i = area_weight[:, None] * one_minus_trans
    dR2 = jnp.sum(dR2_i, axis=0)

    D = (R0**2 + dR2) / (R_s**2)

    if not want_contrib:
        layer_dR2 = jnp.zeros_like(dtau_v)
        return D, dR2, layer_dR2

    tau_eps = 1.0e-30
    ratio = jnp.where(tau_path > tau_eps, one_minus_trans / tau_path, 1.0)
    W = area_weight[:, None] * ratio

    geom_weighted = jnp.matmul(P1D.T, W)
    layer_dR2 = dtau_v * geom_weighted
    return D, dR2, layer_dR2


def _transit_depth_from_opacity(
    state: Dict[str, jnp.ndarray],
    k_tot: jnp.ndarray,  # (nlay, nwl)
    geometry: tuple[jnp.ndarray, jnp.ndarray],
    refraction_mask: jnp.ndarray | None,
) -> jnp.ndarray:
    R0 = state["R0"]
    R_s = state["R_s"]
    rho = state["rho_lay"]
    dz = state["dz"]
    P1D, area_weight = geometry

    k_eff = jnp.maximum(k_tot, 1.0e-99)
    dtau_v = k_eff * rho[:, None] * dz[:, None]
    tau_path = jnp.matmul(P1D, dtau_v)
    if refraction_mask is not None:
        tau_path = jnp.where(refraction_mask, 1.0e30, tau_path)

    one_minus_trans = 1.0 - jnp.exp(-tau_path)
    dR2 = jnp.sum(area_weight[:, None] * one_minus_trans, axis=0)
    return (R0**2 + dR2) / (R_s**2)


def compute_transit_depth_1d_lbl(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
    opac: Mapping[str, jnp.ndarray] | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    contri_func = state.get("contri_func", False)
    nlay = state["nlay"]
    nwl = state["nwl"]

    refraction_mask = None
    if int(state.get("refraction_mode", 0)) == 1:
        if opac is None:
            raise RuntimeError("Refraction requested but no opac cache was provided to the LBL transit RT kernel.")
        refraction_mask = refraction_cutoff_mask(state, params, opac)

    geometry = _build_transit_geometry(state)
    k_tot = _sum_opacity_components_lbl(state, opacity_components)  # (nlay, nwl)

    if "f_cloud" in params and "cloud" in opacity_components:
        f_cloud = jnp.clip(params["f_cloud"], 0.0, 1.0)
        k_no_cloud = k_tot - opacity_components["cloud"]

        if contri_func:
            D_cloud, dR2_cloud, layer_dR2_cloud = _transit_depth_and_contrib_from_opacity(
                state, k_tot, geometry=geometry, refraction_mask=refraction_mask, want_contrib=True
            )
            D_clear, dR2_clear, layer_dR2_clear = _transit_depth_and_contrib_from_opacity(
                state, k_no_cloud, geometry=geometry, refraction_mask=refraction_mask, want_contrib=True
            )

            D_net = f_cloud * D_cloud + (1.0 - f_cloud) * D_clear
            dR2 = f_cloud * dR2_cloud + (1.0 - f_cloud) * dR2_clear
            layer_dR2 = f_cloud * layer_dR2_cloud + (1.0 - f_cloud) * layer_dR2_clear
            contrib_func_norm = layer_dR2 / jnp.maximum(dR2[None, :], 1e-30)
        else:
            D_cloud = _transit_depth_from_opacity(state, k_tot, geometry=geometry, refraction_mask=refraction_mask)
            D_clear = _transit_depth_from_opacity(state, k_no_cloud, geometry=geometry, refraction_mask=refraction_mask)
            D_net = f_cloud * D_cloud + (1.0 - f_cloud) * D_clear
            contrib_func_norm = jnp.zeros((nlay, nwl), dtype=D_net.dtype)
    else:
        if contri_func:
            D_net, dR2, layer_dR2 = _transit_depth_and_contrib_from_opacity(
                state, k_tot, geometry=geometry, refraction_mask=refraction_mask, want_contrib=True
            )
            contrib_func_norm = layer_dR2 / jnp.maximum(dR2[None, :], 1e-30)
        else:
            D_net = _transit_depth_from_opacity(state, k_tot, geometry=geometry, refraction_mask=refraction_mask)
            contrib_func_norm = jnp.zeros((nlay, nwl), dtype=D_net.dtype)

    return D_net, contrib_func_norm
