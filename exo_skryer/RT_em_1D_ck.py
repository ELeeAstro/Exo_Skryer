"""
RT_em_1D_ck.py
==============
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from . import build_opacities as XS
from .data_constants import kb, h, c_light, pc
from .RT_em_schemes import solve_alpha_eaa

__all__ = ["compute_emission_spectrum_1d_ck"]


def _get_ck_weights(opac: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    g_weights = opac.get("g_weights")
    if g_weights is None:
        raise RuntimeError("Missing opac['g_weights'] for c-k integration.")
    return g_weights


def _sum_opacity_components_ck(
    state: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
    opac: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    nlay = state["nlay"]
    nwl = state["nwl"]

    if not opacity_components:
        g_weights = _get_ck_weights(opac)
        ng = g_weights.shape[-1]
        return jnp.zeros((nlay, nwl, ng))

    line_opacity = opacity_components.get("line")
    if line_opacity is None:
        g_weights = _get_ck_weights(opac)
        ng = g_weights.shape[-1]
        line_opacity = jnp.zeros((nlay, nwl, ng))

    zeros_2d = jnp.zeros((nlay, nwl), dtype=line_opacity.dtype)
    component_keys_2d = ("rayleigh", "cia", "special", "cloud")
    components_2d = jnp.stack([opacity_components.get(k, zeros_2d) for k in component_keys_2d], axis=0)
    summed_2d = jnp.sum(components_2d, axis=0)

    return line_opacity + summed_2d[:, :, None]


def _planck_lambda(wavelength_cm: jnp.ndarray, temperature: jnp.ndarray) -> jnp.ndarray:
    exponent = (h * c_light) / (wavelength_cm * kb * jnp.maximum(temperature, 1.0))
    expm1 = jnp.expm1(jnp.clip(exponent, a_min=None, a_max=80.0))
    prefactor = 2.0 * h * c_light**2 / (wavelength_cm**5)
    return prefactor / jnp.maximum(expm1, 1e-300)


def _layer_optical_depth_ck(k_tot: jnp.ndarray, rho: jnp.ndarray, dz: jnp.ndarray) -> jnp.ndarray:
    return k_tot * rho[:, None, None] * dz[:, None, None]


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

    k_tot_scat = k_tot_scat[:, :, None]
    cloud_g = jnp.broadcast_to(cloud_g[:, :, None], k_tot.shape)

    ssa = jnp.clip(k_tot_scat / k_tot_safe, a_min=0.0, a_max=0.95)
    g = cloud_g
    return ssa, g


def _scale_flux_ratio(
    flux: jnp.ndarray,
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    stellar_flux = state.get("stellar_flux")
    has_stellar_flux = state.get("has_stellar_flux")
    use_stellar = (has_stellar_flux is not None) & (has_stellar_flux != 0)
    if stellar_flux is not None and use_stellar:
        F_star = stellar_flux.astype(jnp.float64)
    else:
        if "F_star" not in params:
            raise ValueError("compute_emission_spectrum_1d_ck requires stellar_flux or parameter 'F_star'.")
        F_star = params["F_star"].astype(jnp.float64)
    R0 = state["R0"].astype(jnp.float64)
    R_s = state["R_s"].astype(jnp.float64)
    scale = (R0**2) / (jnp.maximum(F_star, 1.0e-30) * (R_s**2))
    return flux * scale


def compute_emission_spectrum_1d_ck(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
    opac: Dict[str, jnp.ndarray],
    emission_solver=solve_alpha_eaa,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    contri_func = state.get("contri_func", False)
    wl_cm = state["wl"].astype(jnp.float64) * 1.0e-4
    T_lev = state["T_lev"].astype(jnp.float64)
    rho_lay = state["rho_lay"].astype(jnp.float64)
    dz = state["dz"].astype(jnp.float64)
    be_levels = _planck_lambda(wl_cm[None, :], T_lev[:, None])
    if "T_int" in params:
        T_int = params["T_int"].astype(jnp.float64)
        be_internal = _planck_lambda(wl_cm[None, :], T_int[None, None])[0]
    else:
        be_internal = jnp.zeros_like(be_levels[-1])

    if contri_func:
        def _lw_up_for_components(components: Mapping[str, jnp.ndarray], k_tot_local: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            ssa_ck, g_ck = _compute_scattering_properties(
                components,
                state,
                k_tot_local,
            )
            dtau_ck = _layer_optical_depth_ck(k_tot_local, rho_lay, dz)
            g_weights = _get_ck_weights(opac)
            dtau_by_g = jnp.moveaxis(dtau_ck, -1, 0)
            ssa_by_g = jnp.moveaxis(ssa_ck, -1, 0)
            g_by_g = jnp.moveaxis(g_ck, -1, 0)
            g_weights = g_weights[: dtau_by_g.shape[0]]

            def _scan_body(carry, inputs):
                lw_up_accum, contrib_accum = carry
                dtau_slice, ssa_slice, g_slice, weight = inputs
                lw_up_g, _, layer_contrib_g = emission_solver(
                    be_levels, dtau_slice, ssa_slice, g_slice, be_internal,
                    return_layer_contrib=True
                )
                weight_lw = weight.astype(lw_up_accum.dtype)
                weight_cf = weight.astype(contrib_accum.dtype)
                lw_up_accum = lw_up_accum + weight_lw * lw_up_g
                contrib_accum = contrib_accum + weight_cf * layer_contrib_g
                return (lw_up_accum, contrib_accum), None

            lw_up_init = jnp.zeros_like(be_levels)
            nlay = state["nlay"]
            nwl = state["nwl"]
            contrib_init = jnp.zeros((nlay, nwl), dtype=be_levels.dtype)
            (lw_up_out, contrib_out), _ = lax.scan(
                _scan_body, (lw_up_init, contrib_init), (dtau_by_g, ssa_by_g, g_by_g, g_weights)
            )
            return lw_up_out, contrib_out
    else:
        def _lw_up_for_components(components: Mapping[str, jnp.ndarray], k_tot_local: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            ssa_ck, g_ck = _compute_scattering_properties(
                components,
                state,
                k_tot_local,
            )
            dtau_ck = _layer_optical_depth_ck(k_tot_local, rho_lay, dz)
            g_weights = _get_ck_weights(opac)
            dtau_by_g = jnp.moveaxis(dtau_ck, -1, 0)
            ssa_by_g = jnp.moveaxis(ssa_ck, -1, 0)
            g_by_g = jnp.moveaxis(g_ck, -1, 0)
            g_weights = g_weights[: dtau_by_g.shape[0]]

            def _scan_body(lw_up_accum, inputs):
                dtau_slice, ssa_slice, g_slice, weight = inputs
                lw_up_g, _, _ = emission_solver(
                    be_levels, dtau_slice, ssa_slice, g_slice, be_internal,
                    return_layer_contrib=False
                )
                return lw_up_accum + weight.astype(lw_up_accum.dtype) * lw_up_g, None

            lw_up_init = jnp.zeros_like(be_levels)
            lw_up_out, _ = lax.scan(_scan_body, lw_up_init, (dtau_by_g, ssa_by_g, g_by_g, g_weights))
            nlay = state["nlay"]
            nwl = state["nwl"]
            contrib_out = jnp.zeros((nlay, nwl), dtype=be_levels.dtype)
            return lw_up_out, contrib_out

    k_tot_cloud = _sum_opacity_components_ck(state, opacity_components, opac)
    lw_up_cloud, layer_contrib_cloud = _lw_up_for_components(opacity_components, k_tot_cloud)

    if "f_cloud" in params and "cloud" in opacity_components:
        f_cloud = jnp.clip(params["f_cloud"], 0.0, 1.0)
        cloud_ext = opacity_components["cloud"]
        k_tot_clear = k_tot_cloud - cloud_ext[:, :, None]

        zeros = jnp.zeros_like(cloud_ext)
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
        R0 = state["R0"].astype(jnp.float64)
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
