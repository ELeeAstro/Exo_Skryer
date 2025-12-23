"""
RT_em_1D.py
===========
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax

from . import build_opacities as XS
from .data_constants import kb, h, c_light, pc


_MU_NODES = (0.0454586727, 0.2322334416, 0.5740198775, 0.9030775973)
_MU_WEIGHTS = (0.0092068785, 0.1285704278, 0.4323381850, 0.4298845087)
nstreams = len(_MU_NODES) * 2
_DT_THRESHOLD = 1.0e-4
_DT_SAFE = 1.0e-12

__all__ = ["compute_emission_spectrum_1d"]


def _get_ck_weights(state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
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
    nlay = state["nlay"]
    nwl = state["nwl"]

    if not opacity_components:
        g_weights = _get_ck_weights(state)
        ng = g_weights.shape[-1]
        return jnp.zeros((nlay, nwl, ng))

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
    # Inputs are already JAX arrays, no need to wrap
    exponent = (h * c_light) / (wavelength_cm * kb * jnp.maximum(temperature, 1.0))
    expm1 = jnp.expm1(jnp.clip(exponent, a_min=None, a_max=80.0))
    prefactor = 2.0 * h * c_light**2 / (wavelength_cm**5)
    return prefactor / jnp.maximum(expm1, 1e-300)


def _layer_optical_depth_lbl(k_tot: jnp.ndarray, rho: jnp.ndarray, dz: jnp.ndarray) -> jnp.ndarray:
    return k_tot * rho[:, None] * dz[:, None]


def _layer_optical_depth_ck(k_tot: jnp.ndarray, rho: jnp.ndarray, dz: jnp.ndarray) -> jnp.ndarray:
    return k_tot * rho[:, None, None] * dz[:, None, None]


def _solve_alpha_eaa(
    be_levels: jnp.ndarray,
    dtau_layers: jnp.ndarray,
    ssa: jnp.ndarray,
    g_phase: jnp.ndarray,
    be_internal: jnp.ndarray,
    return_layer_contrib: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    nlev, nwl = be_levels.shape
    nlay = nlev - 1

    be_levels = be_levels.astype(jnp.float64)[::-1]
    dtau_layers = dtau_layers.astype(jnp.float64)[::-1]
    ssa = ssa.astype(jnp.float64)[::-1]
    g_phase = g_phase.astype(jnp.float64)[::-1]

    al = be_levels[1:] - be_levels[:-1]
    lw_up_sum = jnp.zeros((nlev, nwl))
    lw_down_sum = jnp.zeros((nlev, nwl))

    # --- your EAA machinery (unchanged) ---
    mask = g_phase >= 1.0e-4
    fc = jnp.where(mask, g_phase**nstreams, 0.0)
    pmom2 = jnp.where(mask, g_phase**(nstreams + 1), 0.0)
    ratio = jnp.maximum((fc**2) / jnp.maximum(pmom2**2, 1.0e-30), 1.0e-30)
    sigma_sq = jnp.where(mask, ((nstreams + 1) ** 2 - nstreams**2) / jnp.log(ratio), 1.0)
    c = jnp.exp((nstreams**2) / (2.0 * sigma_sq))
    fc_scaled = c * fc

    w_in = jnp.clip(ssa, 0.0, 0.99)
    denom = jnp.maximum(1.0 - fc_scaled * w_in, 1.0e-12)
    w0 = jnp.where(mask, w_in * ((1.0 - fc_scaled) / denom), w_in)
    dtau = jnp.where(mask, (1.0 - w_in * fc_scaled) * dtau_layers, dtau_layers)
    hg = g_phase
    eps = jnp.sqrt((1.0 - w0) * (1.0 - hg * w0))
    dtau_a = eps * dtau
    # --------------------------------------

    # Optical depth above each layer top (per wavelength), using the transported dtau_a
    tau_interface = jnp.concatenate([jnp.zeros((1, nwl), dtype=dtau_a.dtype),
                                     jnp.cumsum(dtau_a, axis=0)], axis=0)
    tau_top_layer = tau_interface[:-1]  # shape (nlay, nwl)

    if return_layer_contrib:
        layer_contrib_sum = jnp.zeros((nlay, nwl), dtype=be_levels.dtype)

    for mu, weight in zip(_MU_NODES, _MU_WEIGHTS):
        T_trans = jnp.exp(-dtau_a / mu)
        mu_over_dtau = mu / jnp.maximum(dtau_a, _DT_SAFE)

        def down_body(k, lw):
            linear = (
                lw[k] * T_trans[k]
                + be_levels[k + 1]
                - al[k] * mu_over_dtau[k]
                - (be_levels[k] - al[k] * mu_over_dtau[k]) * T_trans[k]
            )
            iso = (
                lw[k] * T_trans[k]
                + 0.5 * (be_levels[k] + be_levels[k + 1]) * (1.0 - T_trans[k])
            )
            next_val = jnp.where(dtau_a[k] > _DT_THRESHOLD, linear, iso)
            return lw.at[k + 1].set(next_val)

        lw_init = jnp.zeros((nlev, nwl), dtype=be_levels.dtype)
        lw_down = lax.fori_loop(0, nlay, down_body, lw_init)

        lw_up_init = jnp.zeros((nlev, nwl), dtype=be_levels.dtype).at[-1].set(
            lw_down[-1] + be_internal
        )

        if return_layer_contrib:
            # Transmission from each layer-top interface to space for this stream
            T_toa = jnp.exp(-tau_top_layer / mu)  # (nlay, nwl)

            def up_body(idx, carry):
                lw, layer_acc = carry
                k = nlay - 1 - idx

                linear = (
                    lw[k + 1] * T_trans[k]
                    + be_levels[k]
                    + al[k] * mu_over_dtau[k]
                    - (be_levels[k + 1] + al[k] * mu_over_dtau[k]) * T_trans[k]
                )
                iso = (
                    lw[k + 1] * T_trans[k]
                    + 0.5 * (be_levels[k] + be_levels[k + 1]) * (1.0 - T_trans[k])
                )
                I_top = jnp.where(dtau_a[k] > _DT_THRESHOLD, linear, iso)

                # source-created intensity at the layer top for this stream
                source = I_top - lw[k + 1] * T_trans[k]  # (nwl,)

                # accumulate TOA-attributed contribution
                layer_acc = layer_acc.at[k].add(weight * source * T_toa[k])
                lw = lw.at[k].set(I_top)
                return (lw, layer_acc)

            lw_up, layer_contrib_sum = lax.fori_loop(
                0, nlay, up_body, (lw_up_init, layer_contrib_sum)
            )
        else:
            def up_body(idx, lw):
                k = nlay - 1 - idx
                linear = (
                    lw[k + 1] * T_trans[k]
                    + be_levels[k]
                    + al[k] * mu_over_dtau[k]
                    - (be_levels[k + 1] + al[k] * mu_over_dtau[k]) * T_trans[k]
                )
                iso = (
                    lw[k + 1] * T_trans[k]
                    + 0.5 * (be_levels[k] + be_levels[k + 1]) * (1.0 - T_trans[k])
                )
                I_top = jnp.where(dtau_a[k] > _DT_THRESHOLD, linear, iso)
                return lw.at[k].set(I_top)

            lw_up = lax.fori_loop(0, nlay, up_body, lw_up_init)

        lw_down_sum = lw_down_sum + lw_down * weight
        lw_up_sum = lw_up_sum + lw_up * weight

    # Flux outputs (your convention)
    lw_up_flux = jnp.pi * lw_up_sum
    lw_down_flux = jnp.pi * lw_down_sum

    if return_layer_contrib:
        # Reverse back to match original pressure grid (bottom to top)
        layer_contrib_flux = jnp.pi * layer_contrib_sum[::-1]
    else:
        layer_contrib_flux = jnp.zeros((nlay, nwl), dtype=lw_up_flux.dtype)

    return lw_up_flux, lw_down_flux, layer_contrib_flux



def compute_emission_spectrum_1d(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Use direct comparison instead of bool() for JIT compatibility
    ck_mode = state.get("ck", False)
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

    if ck_mode:
        if contri_func:
            def _lw_up_for_components(components: Mapping[str, jnp.ndarray], k_tot_local: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
                ssa_ck, g_ck = _compute_scattering_properties(
                    components,
                    state,
                    k_tot_local,
                    ck_mode=True,
                )
                dtau_ck = _layer_optical_depth_ck(k_tot_local, rho_lay, dz)
                g_weights = _get_ck_weights(state)
                dtau_by_g = jnp.moveaxis(dtau_ck, -1, 0)
                ssa_by_g = jnp.moveaxis(ssa_ck, -1, 0)
                g_by_g = jnp.moveaxis(g_ck, -1, 0)
                g_weights = g_weights[: dtau_by_g.shape[0]]

                def _scan_body(carry, inputs):
                    lw_up_accum, contrib_accum = carry
                    dtau_slice, ssa_slice, g_slice, weight = inputs
                    lw_up_g, _, layer_contrib_g = _solve_alpha_eaa(
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
                    ck_mode=True,
                )
                dtau_ck = _layer_optical_depth_ck(k_tot_local, rho_lay, dz)
                g_weights = _get_ck_weights(state)
                dtau_by_g = jnp.moveaxis(dtau_ck, -1, 0)
                ssa_by_g = jnp.moveaxis(ssa_ck, -1, 0)
                g_by_g = jnp.moveaxis(g_ck, -1, 0)
                g_weights = g_weights[: dtau_by_g.shape[0]]

                def _scan_body(lw_up_accum, inputs):
                    dtau_slice, ssa_slice, g_slice, weight = inputs
                    lw_up_g, _, _ = _solve_alpha_eaa(
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

        k_tot_cloud = _sum_opacity_components_ck(state, opacity_components)
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
    else:
        def _lw_up_for_components(components: Mapping[str, jnp.ndarray], k_tot_local: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            ssa_lbl, g_lbl = _compute_scattering_properties(
                components,
                state,
                k_tot_local,
                ck_mode=False,
            )
            dtau_lbl = _layer_optical_depth_lbl(k_tot_local, rho_lay, dz)
            lw_up_out, _, layer_contrib_out = _solve_alpha_eaa(
                be_levels, dtau_lbl, ssa_lbl, g_lbl, be_internal,
                return_layer_contrib=contri_func
            )
            return lw_up_out, layer_contrib_out

        k_tot_cloud = _sum_opacity_components_lbl(state, opacity_components)
        lw_up_cloud, layer_contrib_cloud = _lw_up_for_components(opacity_components, k_tot_cloud)

        if "f_cloud" in params and "cloud" in opacity_components:
            f_cloud = jnp.clip(params["f_cloud"], 0.0, 1.0)
            cloud_ext = opacity_components["cloud"]
            k_tot_clear = k_tot_cloud - cloud_ext

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
    # Use direct comparison instead of bool() for JIT compatibility
    if state.get("is_brown_dwarf", False):
        R0 = state["R0"].astype(jnp.float64) 
        D = params["D"]
        distance = D * pc
        final_spectrum = top_flux * (R0/distance)**2
    else:
        final_spectrum = _scale_flux_ratio(top_flux, state, params)

    # Compute contribution function if requested, otherwise return zeros
    if contri_func:
        layer_contrib = jnp.clip(layer_contrib_flux, 0.0)  # (nlay, nwl)
        contrib_func_norm = layer_contrib / jnp.maximum(layer_contrib.sum(axis=0, keepdims=True), 1e-30)
    else:
        # Return zeros with correct shape (nlay, nwl)
        nlay = state["nlay"]
        nwl = state["nwl"]
        contrib_func_norm = jnp.zeros((nlay, nwl), dtype=final_spectrum.dtype)

    return final_spectrum, contrib_func_norm


def _compute_scattering_properties(
    opacity_components: Mapping[str, jnp.ndarray],
    state: Dict[str, jnp.ndarray],
    k_tot: jnp.ndarray,
    ck_mode: bool,
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

    if ck_mode:
        k_tot_scat = k_tot_scat[:, :, None]
        # Asymmetry parameter does not depend on g-point; broadcast to match c-k k_tot shape.
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
    if stellar_flux is not None:
        F_star = stellar_flux.astype(jnp.float64)
    else:
        if "F_star" not in params:
            raise ValueError("compute_emission_spectrum_1d requires stellar_flux or parameter 'F_star'.")
        F_star = params["F_star"].astype(jnp.float64)
    R0 = state["R0"].astype(jnp.float64)
    R_s = state["R_s"].astype(jnp.float64)
    scale = (R0**2) / (jnp.maximum(F_star, 1.0e-30) * (R_s**2))
    return flux * scale
