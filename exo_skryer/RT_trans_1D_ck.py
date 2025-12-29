"""
RT_trans_1D_ck.py
=================
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import jax
import jax.numpy as jnp
from . import build_opacities as XS

__all__ = ["compute_transit_depth_1d_ck"]


def _get_ck_weights(state):
    g_weights = state.get("g_weights")
    if g_weights is not None:
        return g_weights
    if not XS.has_ck_data():
        raise RuntimeError("c-k g-weights not built; run build_opacities() with ck tables.")
    g_weights = XS.ck_g_weights()
    if g_weights.ndim > 1:
        g_weights = g_weights[0]
    return g_weights


def _sum_opacity_components_ck(
    state: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    """
    Return the summed opacity grid for correlated-k mode.

    Line opacity has shape (nlay, nwl, ng).
    Other opacities have shape (nlay, nwl) and are broadcast over g-dimension.
    Returns shape (nlay, nwl, ng).
    """
    nlay = state["nlay"]
    nwl = state["nwl"]

    line_opacity = opacity_components.get("line")
    if line_opacity is None:
        g_weights = _get_ck_weights(state)
        ng = g_weights.shape[-1]
        line_opacity = jnp.zeros((nlay, nwl, ng))

    zeros_2d = jnp.zeros((nlay, nwl), dtype=line_opacity.dtype)
    component_keys_2d = ("rayleigh", "cia", "special", "cloud")
    components_2d = jnp.stack([opacity_components.get(k, zeros_2d) for k in component_keys_2d], axis=0)
    summed_2d = jnp.sum(components_2d, axis=0)

    return line_opacity + summed_2d[:, :, None]


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
) -> jnp.ndarray:
    R0 = state["R0"]
    R_s = state["R_s"]
    rho = state["rho_lay"]
    dz = state["dz"]
    P1D, area_weight = geometry

    k_eff = jnp.maximum(k_tot, 1.0e-99)
    dtau_v = k_eff * rho[:, None] * dz[:, None]
    tau_path = jnp.matmul(P1D, dtau_v)

    one_minus_trans = 1.0 - jnp.exp(-tau_path)
    dR2 = jnp.sum(area_weight[:, None] * one_minus_trans, axis=0)
    return (R0**2 + dR2) / (R_s**2)


def _integrate_g_points_simple(
    k_array: jnp.ndarray,
    g_indices: jnp.ndarray,
    g_weights: jnp.ndarray,
    state: Dict[str, jnp.ndarray],
    geometry: tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Integrate transit depth over g-points using vmap.

    Computes: D_net = Σ(weight_i × D_i) with parallel evaluation of each D_i.

    Parameters
    ----------
    k_array : jnp.ndarray, shape (nlay, nwl, ng)
        Opacity array (k_tot or k_no_cloud).
    g_indices : jnp.ndarray, shape (ng,)
        G-point indices for slicing k_array.
    g_weights : jnp.ndarray, shape (ng,)
        Quadrature weights for integration.
    state : Dict[str, jnp.ndarray]
        Atmospheric state dictionary.
    geometry : tuple
        Precomputed geometry arrays (P1D, area_weight).

    Returns
    -------
    D_net : jnp.ndarray, shape (nwl,)
        Integrated transit depth spectrum.
    """
    def _compute_one_g(g_idx, g_weight):
        k_slice = k_array[:, :, g_idx]
        D_i = _transit_depth_from_opacity(state, k_slice, geometry)
        w = g_weight.astype(D_i.dtype)  # Preserve original dtype handling
        return w * D_i

    D_per_g = jax.vmap(_compute_one_g)(g_indices, g_weights)  # (ng, nwl)
    return jnp.sum(D_per_g, axis=0)  # (nwl,)


def _integrate_g_points_with_contrib(
    k_array: jnp.ndarray,
    g_indices: jnp.ndarray,
    g_weights: jnp.ndarray,
    state: Dict[str, jnp.ndarray],
    geometry: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Integrate transit depth and contribution function over g-points using vmap.

    Computes weighted sums in parallel:
    - D_net = Σ(weight_i × D_i)
    - dR2 = Σ(weight_i × dR2_i)
    - layer_dR2 = Σ(weight_i × layer_i)

    Parameters
    ----------
    k_array : jnp.ndarray, shape (nlay, nwl, ng)
        Opacity array.
    g_indices : jnp.ndarray, shape (ng,)
        G-point indices.
    g_weights : jnp.ndarray, shape (ng,)
        Quadrature weights.
    state : Dict[str, jnp.ndarray]
        Atmospheric state.
    geometry : tuple
        Precomputed geometry.

    Returns
    -------
    D_net : jnp.ndarray, shape (nwl,)
        Integrated transit depth.
    dR2 : jnp.ndarray, shape (nwl,)
        Integrated radius-squared change.
    layer_dR2 : jnp.ndarray, shape (nlay, nwl)
        Integrated layer contribution.
    """
    def _compute_one_g(g_idx, g_weight):
        k_slice = k_array[:, :, g_idx]
        D_i, dR2_i, layer_i = _transit_depth_and_contrib_from_opacity(
            state, k_slice, geometry, want_contrib=True
        )
        w = g_weight.astype(D_i.dtype)
        return w * D_i, w * dR2_i, w * layer_i

    D_per_g, dR2_per_g, layer_per_g = jax.vmap(_compute_one_g)(g_indices, g_weights)

    return (
        jnp.sum(D_per_g, axis=0),
        jnp.sum(dR2_per_g, axis=0),
        jnp.sum(layer_per_g, axis=0),
    )


def compute_transit_depth_1d_ck(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    contri_func = state.get("contri_func", False)
    nlay = state["nlay"]
    nwl = state["nwl"]

    geometry = _build_transit_geometry(state)
    k_tot = _sum_opacity_components_ck(state, opacity_components)  # (nlay, nwl, ng)

    ng = k_tot.shape[-1]
    g_weights = _get_ck_weights(state)[:ng]
    g_indices = jnp.arange(ng)

    if "f_cloud" in params and "cloud" in opacity_components:
        f_cloud = jnp.clip(params["f_cloud"], 0.0, 1.0)
        cloud_component = opacity_components["cloud"]
        if cloud_component.ndim == 2:
            cloud_component = cloud_component[:, :, None]
        k_no_cloud = k_tot - cloud_component

        if contri_func:
            D_cloud, dR2_cloud, layer_dR2_cloud = _integrate_g_points_with_contrib(
                k_tot, g_indices, g_weights, state, geometry
            )
            D_clear, dR2_clear, layer_dR2_clear = _integrate_g_points_with_contrib(
                k_no_cloud, g_indices, g_weights, state, geometry
            )

            D_net = f_cloud * D_cloud + (1.0 - f_cloud) * D_clear
            dR2 = f_cloud * dR2_cloud + (1.0 - f_cloud) * dR2_clear
            layer_dR2 = f_cloud * layer_dR2_cloud + (1.0 - f_cloud) * layer_dR2_clear
            contrib_func_norm = layer_dR2 / jnp.maximum(dR2[None, :], 1e-30)
        else:
            D_cloud = _integrate_g_points_simple(k_tot, g_indices, g_weights, state, geometry)
            D_clear = _integrate_g_points_simple(k_no_cloud, g_indices, g_weights, state, geometry)

            D_net = f_cloud * D_cloud + (1.0 - f_cloud) * D_clear
            contrib_func_norm = jnp.zeros((nlay, nwl), dtype=D_net.dtype)
    else:
        if contri_func:
            D_net, dR2, layer_dR2 = _integrate_g_points_with_contrib(
                k_tot, g_indices, g_weights, state, geometry
            )
            contrib_func_norm = layer_dR2 / jnp.maximum(dR2[None, :], 1e-30)
        else:
            D_net = _integrate_g_points_simple(k_tot, g_indices, g_weights, state, geometry)
            contrib_func_norm = jnp.zeros((nlay, nwl), dtype=D_net.dtype)

    return D_net, contrib_func_norm
