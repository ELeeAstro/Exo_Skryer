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
from . import build_opacities as XS

__all__ = ['compute_transit_depth_1d']


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
    """Precompute geometry terms for transit depth calculation.

    Returns
    -------
    P1D : jax.numpy.ndarray
        Path-length operator with shape `(nlay, nlay)`.
    area_weight : jax.numpy.ndarray
        Annulus area weights with shape `(nlay,)`.
    """
    # State values are already JAX arrays, no need to wrap
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


def _transit_depth_from_opacity(
    state: Dict[str, jnp.ndarray],
    k_tot: jnp.ndarray,
    geometry: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Compute a transit spectrum for a provided total opacity grid."""

    # State values are already JAX arrays, no need to wrap
    R0 = state["R0"]
    R_s = state["R_s"]
    rho = state["rho_lay"]
    dz = state["dz"]

    k_floor = 1.0e-99
    k_eff = jnp.maximum(k_tot, k_floor)
    dtau_v = k_eff * rho[:, None] * dz[:, None]

    if geometry is None:
        P1D, area_weight = _build_transit_geometry(state)
    else:
        P1D, area_weight = geometry
    tau_path = jnp.matmul(P1D, dtau_v)

    one_minus_trans = 1.0 - jnp.exp(-tau_path)
    dR2 = jnp.sum(area_weight[:, None] * one_minus_trans, axis=0)

    R_eff2 = R0**2 + dR2
    return R_eff2 / (R_s**2)


def _transit_depth_ck(
    state: Dict[str, jnp.ndarray],
    k_tot: jnp.ndarray,
    geometry: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Integrate correlated-k opacities over g-points to obtain transit depth."""
    g_weights = _get_ck_weights(state)
    ng = k_tot.shape[-1]
    g_weights = g_weights[:ng]

    def _depth_for_g(k_slice: jnp.ndarray) -> jnp.ndarray:
        return _transit_depth_from_opacity(state, k_slice, geometry=geometry)

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
    `~jax.numpy.ndarray`
        Transit depth spectrum (dimensionless) at the native wavelength grid.
    """

    D_net: jnp.ndarray

    geometry = _build_transit_geometry(state)

    # Use direct comparison instead of == True for JIT compatibility
    if state["ck"]:
        # Corr-k mode: build total opacity then integrate over g-points
        k_tot = _sum_opacity_components_ck(state, opacity_components)  # (nlay, nwl, ng)
        if "f_cloud" in params and "cloud" in opacity_components:
            # Parameter is already a JAX array, no need to wrap
            f_cloud = jnp.clip(params["f_cloud"], 0.0, 1.0)
            cloud_component = opacity_components["cloud"]
            if cloud_component.ndim == 2:
                cloud_component = cloud_component[:, :, None]
            k_no_cloud = k_tot - cloud_component
            D_cloud = _transit_depth_ck(state, k_tot, geometry=geometry)
            D_clear = _transit_depth_ck(state, k_no_cloud, geometry=geometry)
            D_net = f_cloud * D_cloud + (1.0 - f_cloud) * D_clear
        else:
            D_net = _transit_depth_ck(state, k_tot, geometry=geometry)
    else:
        # Lbl mode
        k_tot = _sum_opacity_components_lbl(state, opacity_components) # Return (nlay, nwl)

        if "f_cloud" in params and "cloud" in opacity_components:
            # Parameter is already a JAX array, no need to wrap
            f_cloud = jnp.clip(params["f_cloud"], 0.0, 1.0)
            k_no_cloud = k_tot - opacity_components["cloud"]
            D_cloud = _transit_depth_from_opacity(state, k_tot, geometry=geometry)
            D_clear = _transit_depth_from_opacity(state, k_no_cloud, geometry=geometry)
            D_net = f_cloud * D_cloud + (1.0 - f_cloud) * D_clear
        else:
            D_net = _transit_depth_from_opacity(state, k_tot, geometry=geometry)

    return D_net
