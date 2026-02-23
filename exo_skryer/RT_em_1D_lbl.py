"""
RT_em_1D_lbl.py
===============
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import jax.numpy as jnp

from .data_constants import kb, h, c_light, pc
from .RT_em_schemes import solve_alpha_eaa

__all__ = ["compute_emission_spectrum_1d_lbl"]


def _sum_opacity_components_lbl(
    state: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
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


def _planck_lambda(wavelength_cm: jnp.ndarray, temperature: jnp.ndarray) -> jnp.ndarray:
    exponent = (h * c_light) / (wavelength_cm * kb * jnp.maximum(temperature, 1.0))
    expm1 = jnp.expm1(jnp.clip(exponent, a_min=None, a_max=80.0))
    prefactor = 2.0 * h * c_light**2 / (wavelength_cm**5)
    return prefactor / jnp.maximum(expm1, 1e-300)


def _layer_optical_depth_lbl(k_tot: jnp.ndarray, rho: jnp.ndarray, dz: jnp.ndarray) -> jnp.ndarray:
    return k_tot * rho[:, None] * dz[:, None]


def _compute_scattering_properties(
    opacity_components: Mapping[str, jnp.ndarray],
    state: Dict[str, jnp.ndarray],
    k_tot: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    nlay = state["nlay"]
    nwl = state["nwl"]

    def _get_component(name, shape):
        arr = opacity_components.get(name)
        if arr is None:
            return jnp.zeros(shape)
        return arr

    base_shape = (nlay, nwl)
    k_ray = _get_component("rayleigh", base_shape)
    k_cloud_ext = _get_component("cloud", base_shape)
    cloud_ssa = _get_component("cloud_ssa", base_shape)
    cloud_g = _get_component("cloud_g", base_shape)

    k_cloud_scat = cloud_ssa * k_cloud_ext
    k_tot_scat = k_ray + k_cloud_scat
    k_tot_safe = jnp.maximum(k_tot, 1.0e-30)

    ssa = jnp.clip(k_tot_scat / k_tot_safe, a_min=0.0, a_max=0.95)
    g = cloud_g
    return ssa, g


def _scale_flux_ratio(
    flux: jnp.ndarray,
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    dtype = flux.dtype
    stellar_flux = jnp.asarray(state.get("stellar_flux", jnp.ones_like(flux)), dtype=dtype)
    has_stellar_flux = jnp.asarray(state.get("has_stellar_flux", 0), dtype=jnp.int32)
    f_star_param = jnp.asarray(params.get("F_star", jnp.ones_like(stellar_flux)), dtype=dtype)
    use_stellar = has_stellar_flux != 0
    F_star = jnp.where(use_stellar, stellar_flux, f_star_param)
    R0 = jnp.asarray(state["R0"], dtype=dtype)
    R_s = jnp.asarray(state["R_s"], dtype=dtype)
    scale = (R0**2) / (jnp.maximum(F_star, 1.0e-30) * (R_s**2))
    return flux * scale


def compute_emission_spectrum_1d_lbl(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
    emission_solver=solve_alpha_eaa,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    contri_func = state.get("contri_func", False)
    dtype = state["wl"].dtype
    wl_cm = jnp.asarray(state["wl"], dtype=dtype) * 1.0e-4
    T_lev = jnp.asarray(state["T_lev"], dtype=dtype)
    rho_lay = jnp.asarray(state["rho_lay"], dtype=dtype)
    dz = jnp.asarray(state["dz"], dtype=dtype)
    be_levels = _planck_lambda(wl_cm[None, :], T_lev[:, None])
    if "T_int" in params:
        T_int = jnp.asarray(params["T_int"], dtype=dtype)
        be_internal = _planck_lambda(wl_cm[None, :], T_int[None, None])[0]
    else:
        be_internal = jnp.zeros_like(be_levels[-1])

    def _lw_up_for_components(components: Mapping[str, jnp.ndarray], k_tot_local: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        ssa_lbl, g_lbl = _compute_scattering_properties(
            components,
            state,
            k_tot_local,
        )
        dtau_lbl = _layer_optical_depth_lbl(k_tot_local, rho_lay, dz)
        lw_up_out, _, layer_contrib_out = emission_solver(
            be_levels, dtau_lbl, ssa_lbl, g_lbl, be_internal,
            return_layer_contrib=contri_func
        )
        return lw_up_out, layer_contrib_out

    k_tot_cloud = _sum_opacity_components_lbl(state, opacity_components)
    lw_up_cloud, layer_contrib_cloud = _lw_up_for_components(opacity_components, k_tot_cloud)

    if "f_cloud" in params and "cloud" in opacity_components:
        f_cloud = jnp.clip(params["f_cloud"], 0.0, 1.0)
        k_tot_clear = k_tot_cloud - opacity_components["cloud"]

        zeros = jnp.zeros_like(opacity_components["cloud"])
        opacity_clear = dict(opacity_components)
        opacity_clear["cloud"] = zeros
        opacity_clear["cloud_ssa"] = zeros
        opacity_clear["cloud_g"] = zeros

        lw_up_clear, layer_contrib_clear = _lw_up_for_components(opacity_clear, k_tot_clear)
        lw_up = f_cloud * lw_up_cloud + (1.0 - f_cloud) * lw_up_clear
        layer_contrib_flux = f_cloud * layer_contrib_cloud + (1.0 - f_cloud) * layer_contrib_clear
    else:
        lw_up = lw_up_cloud
        layer_contrib_flux = layer_contrib_cloud

    top_flux = lw_up[0]
    if state.get("is_brown_dwarf", False):
        R0 = jnp.asarray(state["R0"], dtype=top_flux.dtype)
        D = params["D"]
        distance = D * pc
        final_spectrum = top_flux * (R0 / distance) ** 2
    else:
        final_spectrum = _scale_flux_ratio(top_flux, state, params)

    if contri_func:
        layer_contrib = jnp.clip(layer_contrib_flux, 0.0)
        contrib_func_norm = layer_contrib / jnp.maximum(layer_contrib.sum(axis=0, keepdims=True), 1e-30)
    else:
        nlay = state["nlay"]
        nwl = state["nwl"]
        contrib_func_norm = jnp.zeros((nlay, nwl), dtype=final_spectrum.dtype)

    return final_spectrum, contrib_func_norm
