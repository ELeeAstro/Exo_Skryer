"""
RT_trans_1D.py
==============

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from __future__ import annotations

from typing import Dict, Mapping

import jax
import jax.numpy as jnp
import build_opacities as XS

def _get_ck_weights(state):
    g_weights = state.get("g_weights")
    if g_weights is not None:
        return jnp.asarray(g_weights)
    if not XS.has_ck_data():
        raise RuntimeError("c-k g-weights not built; run build_opacities() with ck tables.")
    g_weights = XS.ck_g_weights()
    if g_weights.ndim > 1:
        g_weights = g_weights[0]
    return jnp.asarray(g_weights)


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

    nlay = int(state["nlay"])
    nwl = int(state["nwl"])

    if not opacity_components:
        # Need to get ng from ck data
        g_weights = _get_ck_weights(state)
        ng = g_weights.shape[-1]
        return jnp.zeros((nlay, nwl, ng))

    # Get line opacity (3D: nlay, nwl, ng)
    line_opacity = opacity_components.get("line", None)

    if line_opacity is not None:
        # Start with line opacity (already has g-dimension)
        k_tot = line_opacity
    else:
        # No line opacity, need to create 3D array
        g_weights = _get_ck_weights(state)
        ng = g_weights.shape[-1]
        k_tot = jnp.zeros((nlay, nwl, ng))

    # Add other components (ray, cia, cloud) - these are 2D
    # Need to broadcast them over g-dimension
    for name, component in opacity_components.items():
        if name != "line":
            # component shape: (nlay, nwl)
            # Expand to (nlay, nwl, ng) by broadcasting
            k_tot = k_tot + component[:, :, None]

    return k_tot

def _sum_opacity_components_lbl(
    state: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    """Return the summed opacity grid for all provided components."""
    if opacity_components:
        first = next(iter(opacity_components.values()))
        k_tot = jnp.zeros_like(first)
        for component in opacity_components.values():
            k_tot = k_tot + component
    else:
        nlay = int(state["nlay"])
        nwl = int(state["nwl"])
        k_tot = jnp.zeros((nlay, nwl))

    return k_tot


def _transit_depth_from_opacity(
    state: Dict[str, jnp.ndarray],
    k_tot: jnp.ndarray,
) -> jnp.ndarray:
    """Compute a transit spectrum for a provided total opacity grid."""

    R0 = jnp.asarray(state["R0"])
    R_s = jnp.asarray(state["R_s"])
    z_lev = jnp.asarray(state["z_lev"])
    z_lay = jnp.asarray(state["z_lay"])
    rho = jnp.asarray(state["rho_lay"])
    dz = jnp.asarray(state["dz"])

    k_floor = 1.0e-99
    k_eff = jnp.maximum(k_tot, k_floor)
    dtau_v = k_eff * rho[:, None] * dz[:, None]

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

    tau_path = jnp.matmul(P1D, dtau_v)

    area_weight = 2.0 * r_mid * dr
    one_minus_trans = 1.0 - jnp.exp(-tau_path)
    dR2 = jnp.sum(area_weight[:, None] * one_minus_trans, axis=0)

    R_eff2 = R0**2 + dR2
    return R_eff2 / (R_s**2)


def _transit_depth_ck(
    state: Dict[str, jnp.ndarray],
    k_tot: jnp.ndarray,
) -> jnp.ndarray:
    """Integrate correlated-k opacities over g-points to obtain transit depth."""
    g_weights = _get_ck_weights(state)
    ng = k_tot.shape[-1]
    g_weights = g_weights[:ng]

    def _depth_for_g(k_slice: jnp.ndarray) -> jnp.ndarray:
        return _transit_depth_from_opacity(state, k_slice)

    k_tot_g = jnp.moveaxis(k_tot, -1, 0)  # (ng, nlay, nwl)
    g_depths = jax.vmap(_depth_for_g)(k_tot_g)  # (ng, nwl)
    return jnp.sum(g_weights[:, None] * g_depths, axis=0)


def compute_transit_depth_1d(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    """
    Compute the wavelength-dependent transit depth using a 1D path-length formalism.

    Parameters
    ----------
    state : Dict[str, jnp.ndarray]
        Atmospheric state dictionary containing at least R0, R_s, z_lev, z_lay, rho, and dz.
    params : Dict[str, jnp.ndarray]
        Retrieval parameters used for flexible RT options (e.g., cloud coverage weighting).
    opacity_components : Mapping[str, jnp.ndarray]
        Mapping of opacity component names to arrays shaped (nlay, nlambda).

    Returns
    -------
    jnp.ndarray
        Transit depth spectrum (dimensionless) at the native wavelength grid.
    """

    D_net: jnp.ndarray

    if state["ck"] == True:
        # Corr-k mode: build total opacity then integrate over g-points
        k_tot = _sum_opacity_components_ck(state, opacity_components)  # (nlay, nwl, ng)
        if "f_cloud" in params and "cloud" in opacity_components:
            f_cloud = jnp.clip(jnp.asarray(params["f_cloud"]), 0.0, 1.0)
            cloud_component = opacity_components["cloud"]
            if cloud_component.ndim == 2:
                cloud_component = cloud_component[:, :, None]
            k_no_cloud = k_tot - cloud_component
            D_cloud = _transit_depth_ck(state, k_tot)
            D_clear = _transit_depth_ck(state, k_no_cloud)
            D_net = f_cloud * D_cloud + (1.0 - f_cloud) * D_clear
        else:
            D_net = _transit_depth_ck(state, k_tot)
    else:
        # Lbl mode
        k_tot = _sum_opacity_components_lbl(state, opacity_components) # Return (nlay, nwl)

        if "f_cloud" in params and "cloud" in opacity_components:
            f_cloud = jnp.clip(jnp.asarray(params["f_cloud"]), 0.0, 1.0)
            k_no_cloud = k_tot - opacity_components["cloud"]
            D_cloud = _transit_depth_from_opacity(state, k_tot)
            D_clear = _transit_depth_from_opacity(state, k_no_cloud)
            D_net = f_cloud * D_cloud + (1.0 - f_cloud) * D_clear
        else:
            D_net = _transit_depth_from_opacity(state, k_tot)

    return D_net
