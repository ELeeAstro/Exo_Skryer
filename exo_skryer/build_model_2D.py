"""
build_model_2D.py
=================
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Dict, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from .data_constants import R_jup, R_sun, amu, bar, kb
from .opacity_line import zero_line_opacity, compute_line_opacity
from .opacity_ck import zero_ck_opacity, compute_ck_opacity, compute_ck_opacity_perspecies
from .opacity_ray import zero_ray_opacity, compute_ray_opacity
from .opacity_cia import zero_cia_opacity, compute_cia_opacity
from .opacity_special import zero_special_opacity, compute_special_opacity
from .opacity_cloud import zero_cloud_opacity
from .RT_trans_2D_os import compute_transit_depth_2d_os
from .RT_trans_2D_ck import compute_transit_depth_2d_ck
from .RT_trans_2D_ck_trans import compute_transit_depth_2d_ck_trans
from .instru_convolve import apply_response_functions_cached, get_bandpass_cache
from .vert_chem import constant_vmr, constant_vmr_clr
from .vert_mu import build_compute_mu, constant_mu
from .build_chem import (
    prepare_chemistry_kernel,
    init_fastchem_grid_if_needed,
    init_element_potentials_if_needed,
    init_atmodeller_if_needed,
)
from .limb_asymmetry import (
    split_limb_tag,
    split_limb_parameter_dict,
    merge_limb_parameter_dict,
    validate_limb_parameter_names,
)
from . import build_opacities as XS
from . import kernel_registry as KR
from .build_model import _extract_fixed_params, _build_opac_cache, _require_cache_keys, _resolve_os_ck_opac

__all__ = ["build_forward_model_2d"]


def _stack_two_param_dicts(
    params_east: Dict[str, jnp.ndarray],
    params_west: Dict[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """Stack east/west limb parameter dictionaries onto a leading limb axis."""
    east_keys = set(params_east.keys())
    west_keys = set(params_west.keys())
    if east_keys != west_keys:
        missing_east = sorted(west_keys - east_keys)
        missing_west = sorted(east_keys - west_keys)
        raise ValueError(
            "East/west limb parameter dictionaries must share identical base keys. "
            f"Missing east={missing_east}, missing west={missing_west}"
        )
    return {key: jnp.stack([params_east[key], params_west[key]], axis=0) for key in sorted(east_keys)}


def _split_limb_pytree(tree):
    """Split a two-limb pytree into east/west pytrees by slicing the leading axis."""
    limb_east = tree_util.tree_map(lambda x: x[0], tree)
    limb_west = tree_util.tree_map(lambda x: x[1], tree)
    return limb_east, limb_west


def _resolve_os_none(phys, key: str, fn: Callable):
    raw = getattr(phys, key, None)
    if raw is None:
        raise ValueError(
            f"physics.{key} must be specified explicitly (use 'None' to disable)."
        )
    s = str(raw).lower()
    if s == "none":
        return s, None
    if s == "os":
        return s, fn
    raise NotImplementedError(f"Unknown physics.{key}='{raw}'. Options: none | os")


def _resolve_refraction(phys) -> int:
    refraction_raw = getattr(phys, "refraction", None)
    if refraction_raw is None:
        return 0
    s = str(refraction_raw).strip().lower()
    if s in ("none", "off", "false", "0"):
        return 0
    if s in ("cutoff", "refractive_cutoff", "refraction_cutoff"):
        return 1
    raise NotImplementedError(f"Unknown physics.refraction='{refraction_raw}'")


def _select_kernels_2d(cfg) -> SimpleNamespace:
    phys = cfg.physics

    rt_raw = getattr(phys, "rt_scheme", None)
    rt_scheme = str(rt_raw).lower()
    if rt_scheme != "transit_2d":
        raise ValueError("build_forward_model_2d requires physics.rt_scheme='transit_2d'.")

    if str(getattr(phys, "emission_mode", "planet")).lower() not in ("planet", "none"):
        raise NotImplementedError("transit_2d is only implemented for transit radiative transfer.")

    Tp_kernel = KR.resolve(getattr(phys, "vert_Tp", None) or getattr(phys, "vert_struct", None), KR.VERT_TP, "physics.vert_Tp")
    altitude_kernel = KR.resolve(getattr(phys, "vert_alt", None), KR.VERT_ALT, "physics.vert_alt")
    chemistry_kernel = KR.resolve(getattr(phys, "vert_chem", None), KR.VERT_CHEM, "physics.vert_chem")
    mu_kernel = KR.resolve(getattr(phys, "vert_mu", None), KR.VERT_MU, "physics.vert_mu")
    vert_cloud_raw = getattr(phys, "vert_cloud", "none") or "none"
    vert_cloud_kernel = KR.resolve(vert_cloud_raw, KR.VERT_CLOUD, "physics.vert_cloud")

    line_raw = getattr(phys, "opac_line", None)
    if line_raw is None:
        raise ValueError("physics.opac_line must be specified explicitly (use 'None' to disable).")
    line_str = str(line_raw).lower()
    if line_str == "none":
        line_opac_kernel = None
    elif line_str == "os":
        line_opac_kernel = compute_line_opacity
    elif line_str == "ck":
        line_opac_kernel = compute_ck_opacity
    else:
        raise NotImplementedError("transit_2d currently supports only physics.opac_line: os, ck, or None.")

    if line_str == "ck":
        ray_str, ray_opac_kernel = _resolve_os_ck_opac(phys, "opac_ray", compute_ray_opacity)
        cia_str, cia_opac_kernel = _resolve_os_ck_opac(phys, "opac_cia", compute_cia_opacity)
    else:
        ray_str, ray_opac_kernel = _resolve_os_none(phys, "opac_ray", compute_ray_opacity)
        cia_str, cia_opac_kernel = _resolve_os_none(phys, "opac_cia", compute_cia_opacity)

    cld_raw = getattr(phys, "opac_cloud", None)
    if cld_raw is None:
        raise ValueError("physics.opac_cloud must be specified explicitly (use 'None' to disable).")
    cld_str = str(cld_raw).lower()
    cld_opac_kernel = KR.resolve(cld_raw, KR.OPAC_CLOUD, "physics.opac_cloud")

    special_str = str(getattr(phys, "opac_special", "on")).lower()
    if special_str == "ck":
        raise NotImplementedError("transit_2d does not support ck special opacity.")
    special_opac_kernel = (
        None if special_str in ("none", "off", "false", "0")
        else compute_special_opacity
    )

    ck = (line_str == "ck")
    ck_mix_str = str(getattr(cfg.opac, "ck_mix", "RORR")).upper() if ck else "RORR"
    if ck_mix_str == "PRAS":
        raise NotImplementedError("transit_2d does not support ck_mix: PRAS yet.")
    if ck and ck_mix_str not in ("RORR", "TRANS"):
        raise NotImplementedError(f"transit_2d does not support ck_mix: {ck_mix_str}")

    if ck:
        rt_kernel = compute_transit_depth_2d_ck_trans if ck_mix_str == "TRANS" else compute_transit_depth_2d_ck
        ck_mix_code_static = 3 if ck_mix_str == "TRANS" else 1
    else:
        rt_kernel = compute_transit_depth_2d_os
        ck_mix_code_static = None

    return SimpleNamespace(
        Tp_kernel=Tp_kernel,
        altitude_kernel=altitude_kernel,
        chemistry_kernel=chemistry_kernel,
        mu_kernel=mu_kernel,
        vert_cloud_kernel=vert_cloud_kernel,
        line_opac_kernel=line_opac_kernel,
        ray_opac_kernel=ray_opac_kernel,
        cia_opac_kernel=cia_opac_kernel,
        cld_opac_kernel=cld_opac_kernel,
        special_opac_kernel=special_opac_kernel,
        rt_kernel=rt_kernel,
        ck=ck,
        ck_mix_str=ck_mix_str,
        ck_mix_code_static=ck_mix_code_static,
        contri_func_enabled=bool(getattr(phys, "contri_func", False)),
        refraction_mode=_resolve_refraction(phys),
        line_opac_str=line_str,
        ray_opac_str=ray_str,
        cia_opac_str=cia_str,
        cld_opac_str=cld_str,
        special_opac_str=special_str,
    )


def _validate_config_2d(cfg, k: SimpleNamespace, opac_cache: Dict[str, jnp.ndarray]) -> None:
    validate_limb_parameter_names(p.name for p in cfg.params)
    param_base_names = {
        split_limb_tag(str(getattr(p, "name", "")))[0]
        for p in getattr(cfg, "params", [])
        if getattr(p, "name", None) is not None
    }
    if "M_p" in param_base_names:
        raise NotImplementedError(
            "transit_2d does not support M_p-based gravity inference. "
            "Provide explicit log_10_g_east/log_10_g_west instead."
        )
    if k.contri_func_enabled:
        raise NotImplementedError("physics.contri_func=True is not supported for transit_2d.")

    if k.ck:
        ck_flag = str(getattr(cfg.opac, "ck", False)).strip().lower() in ("true", "1", "yes", "on")
        if not ck_flag:
            raise ValueError("transit_2d with physics.opac_line: ck requires opac.ck: True.")
        _require_cache_keys(
            opac_cache,
            ("ck_sigma_cube", "ck_log10_pressure_grid", "ck_log10_temperature_grids", "g_points", "g_weights"),
            "correlated-k opacity",
        )
    if k.line_opac_str == "os" and not XS.has_line_data():
        raise RuntimeError("Line opacity requested but registry is empty for transit_2d.")
    if k.line_opac_str == "os":
        _require_cache_keys(
            opac_cache,
            ("line_sigma_cube", "line_log10_pressure_grid", "line_log10_temperature_grids"),
            "opacity sampling",
        )

    if k.cia_opac_kernel is not None:
        _require_cache_keys(
            opac_cache,
            ("cia_master_wavelength", "cia_pair_species_i", "cia_pair_species_j",
             "cia_retained_sigma_cube", "cia_retained_log10_temperature_grids",
             "cia_retained_temperature_grids"),
            "CIA",
        )
    if k.ray_opac_kernel is not None:
        _require_cache_keys(opac_cache, ("ray_master_wavelength", "ray_sigma_linear_table"), "Rayleigh")
    if k.special_opac_kernel is not None and XS.has_special_data():
        _require_cache_keys(
            opac_cache,
            ("hminus_master_wavelength", "hminus_temperature_grid", "hminus_log10_temperature_grid"),
            "special-opacity",
        )

    if k.refraction_mode == 1:
        param_names = {p.name for p in cfg.params}
        if "a_sm_joint" not in param_names:
            raise ValueError("transit_2d refraction cutoff requires explicit joint parameter 'a_sm_joint'.")
        if not XS.has_ray_data():
            raise RuntimeError("transit_2d refraction cutoff requires Rayleigh registry data.")

    if k.cld_opac_str in ("madt_rayleigh", "madt-rayleigh", "mie_madt", "lxmie", "mie_full", "full_mie"):
        _require_cache_keys(opac_cache, ("cloud_nk_n", "cloud_nk_k"), "cloud n,k")


def build_forward_model_2d(
    cfg,
    obs: Dict,
    stellar_flux: Optional[np.ndarray] = None,
    return_highres: bool = False,
) -> Callable[[Dict[str, jnp.ndarray]], Union[jnp.ndarray, Dict[str, jnp.ndarray]]]:
    fixed_params = _extract_fixed_params(cfg)
    nlay = int(getattr(cfg.physics, "nlay", 99))
    nlev = nlay + 1

    _ = np.asarray(obs["wl"], dtype=float)
    _ = np.asarray(obs["dwl"], dtype=float)
    has_limb_observations = bool(obs.get("has_limb_observations", False))

    k = _select_kernels_2d(cfg)
    opac_cache = _build_opac_cache()
    _validate_config_2d(cfg, k, opac_cache)

    wl_hi_array = np.asarray(XS.master_wavelength_cut(), dtype=float)
    wl_hi = jnp.asarray(wl_hi_array)
    bandpass_cache = get_bandpass_cache()

    init_fastchem_grid_if_needed(cfg, None)
    init_element_potentials_if_needed(cfg, None)
    init_atmodeller_if_needed(cfg, None)

    chemistry_kernel, trace_species = prepare_chemistry_kernel(
        cfg,
        k.chemistry_kernel,
        {
            "line_opac": k.line_opac_str,
            "ray_opac": k.ray_opac_str,
            "cia_opac": k.cia_opac_str,
            "special_opac": k.special_opac_str,
        },
    )

    mu_kernel = k.mu_kernel
    if k.chemistry_kernel in (constant_vmr, constant_vmr_clr):
        cfg_param_base_names = {
            split_limb_tag(str(getattr(p, "name", "")))[0]
            for p in getattr(cfg, "params", [])
            if getattr(p, "name", None) is not None
        }
        include_atomic_h = "log_10_H_over_H2" in cfg_param_base_names
        packed_mu_species = tuple(
            dict.fromkeys((*trace_species, "H2", "He", *(("H",) if include_atomic_h else ())))
        )
        compute_mu_fast = build_compute_mu(packed_mu_species)
        mu_mode = str(getattr(cfg.physics, "vert_mu", "auto")).lower()

        if mu_mode == "auto":
            def mu_kernel(params, vmr_lay, _nlay, _compute_mu_fast=compute_mu_fast):
                if "mu" in params:
                    return constant_mu(params, _nlay)
                if "__mu_lay__" in vmr_lay:
                    return vmr_lay["__mu_lay__"]
                return _compute_mu_fast(vmr_lay)
        elif mu_mode in ("dynamic", "variable", "vmr"):
            def mu_kernel(params, vmr_lay, _nlay, _compute_mu_fast=compute_mu_fast):
                if "__mu_lay__" in vmr_lay:
                    return vmr_lay["__mu_lay__"]
                return _compute_mu_fast(vmr_lay)

    Tp_kernel = k.Tp_kernel
    altitude_kernel = k.altitude_kernel
    vert_cloud_kernel = k.vert_cloud_kernel
    line_opac_kernel = k.line_opac_kernel
    ray_opac_kernel = k.ray_opac_kernel
    cia_opac_kernel = k.cia_opac_kernel
    cld_opac_kernel = k.cld_opac_kernel
    special_opac_kernel = k.special_opac_kernel
    rt_kernel = k.rt_kernel
    ck = k.ck
    ck_mix_code_static = k.ck_mix_code_static

    @jax.jit
    def _forward_model_impl(
        params: Dict[str, jnp.ndarray],
        wl_runtime: jnp.ndarray,
        opac_cache_runtime: Dict[str, jnp.ndarray],
        bandpass_cache_runtime: Dict[str, jnp.ndarray],
    ):
        full_params = {**fixed_params, **params}
        shared_params, east_params, west_params = split_limb_parameter_dict(full_params)
        params_east = merge_limb_parameter_dict(shared_params, east_params)
        params_west = merge_limb_parameter_dict(shared_params, west_params)

        wl = wl_runtime
        nwl = wl.shape[0]
        opac = opac_cache_runtime

        def build_limb(params_limb: Dict[str, jnp.ndarray]):
            R0 = params_limb["R_p"] * R_jup
            R_s = params_limb["R_s"] * R_sun

            p_bot = params_limb["p_bot"] * bar
            p_top = params_limb["p_top"] * bar
            p_lev = jnp.logspace(jnp.log10(p_bot), jnp.log10(p_top), nlev)
            p_lay = (p_lev[1:] - p_lev[:-1]) / jnp.log(p_lev[1:] / p_lev[:-1])

            T_lev, T_lay = Tp_kernel(p_lev, params_limb)
            vmr_lay = chemistry_kernel(p_lay, T_lay, params_limb, nlay)
            mu_lay = mu_kernel(params_limb, vmr_lay, nlay)
            z_lev, z_lay, dz = altitude_kernel(p_lev, T_lay, mu_lay, params_limb)

            rho_lay = (mu_lay * amu * p_lay) / (kb * T_lay)
            nd_lay = p_lay / (kb * T_lay)
            q_c_lay = vert_cloud_kernel(p_lay, T_lay, mu_lay, rho_lay, nd_lay, params_limb)

            state = {
                "nwl": nwl,
                "nlay": nlay,
                "wl": wl,
                "R0": R0,
                "R_s": R_s,
                "p_lev": p_lev,
                "p_lay": p_lay,
                "T_lev": T_lev,
                "T_lay": T_lay,
                "z_lev": z_lev,
                "z_lay": z_lay,
                "dz": dz,
                "mu_lay": mu_lay,
                "rho_lay": rho_lay,
                "nd_lay": nd_lay,
                "q_c_lay": q_c_lay,
                "vmr_lay": vmr_lay,
                "contri_func": k.contri_func_enabled,
                "refraction_mode": k.refraction_mode,
            }
            if "cloud_nk_n" in opac:
                state["cloud_nk_n"] = opac["cloud_nk_n"]
                state["cloud_nk_k"] = opac["cloud_nk_k"]
            if ck_mix_code_static is not None:
                state["ck_mix"] = ck_mix_code_static

            opacity_components = {
                "line": zero_ck_opacity(state, opac, params_limb) if ck else zero_line_opacity(state, params_limb),
                "rayleigh": zero_ray_opacity(state, params_limb),
                "cia": zero_cia_opacity(state, params_limb),
                "special": zero_special_opacity(state, params_limb),
                "cloud": zero_cloud_opacity(state, params_limb)[0],
            }
            if line_opac_kernel is not None:
                if ck and ck_mix_code_static == 3:
                    sigma_ps, vmr_ps = compute_ck_opacity_perspecies(state, opac, params_limb)
                    opacity_components["line_perspecies"] = sigma_ps
                    opacity_components["vmr_perspecies"] = vmr_ps
                else:
                    opacity_components["line"] = line_opac_kernel(state, opac, params_limb)
            if ray_opac_kernel is not None:
                opacity_components["rayleigh"] = ray_opac_kernel(state, opac, params_limb)
            if cia_opac_kernel is not None:
                opacity_components["cia"] = cia_opac_kernel(state, opac, params_limb)
            if special_opac_kernel is not None:
                opacity_components["special"] = special_opac_kernel(state, opac, params_limb)
            if cld_opac_kernel is not None:
                opacity_components["cloud"] = cld_opac_kernel(state, params_limb)[0]

            return state, params_limb, opacity_components
        params_limb_stacked = _stack_two_param_dicts(params_east, params_west)
        state_stacked, params_runtime_stacked, opacity_stacked = jax.vmap(build_limb)(params_limb_stacked)
        state_east, state_west = _split_limb_pytree(state_stacked)
        params_east_runtime, params_west_runtime = _split_limb_pytree(params_runtime_stacked)
        opacity_east, opacity_west = _split_limb_pytree(opacity_stacked)

        rt_out = rt_kernel(
            state_east,
            params_east_runtime,
            opacity_east,
            state_west,
            params_west_runtime,
            opacity_west,
            opac,
        )

        hires_stacked = jnp.stack([rt_out["hires_east"], rt_out["hires_west"]], axis=0)
        binned_stacked = jax.vmap(
            lambda spectrum: apply_response_functions_cached(spectrum, bandpass_cache_runtime)
        )(hires_stacked)
        binned_east, binned_west = binned_stacked[0], binned_stacked[1]
        hires_east_scaled = 0.5 * rt_out["hires_east"]
        hires_west_scaled = 0.5 * rt_out["hires_west"]
        binned_east_scaled = 0.5 * binned_east
        binned_west_scaled = 0.5 * binned_west
        if has_limb_observations:
            east_slice = obs["east_slice"]
            west_slice = obs["west_slice"]
            binned_likelihood = jnp.concatenate(
                [binned_east_scaled[east_slice], binned_west_scaled[west_slice]],
                axis=0,
            )
        else:
            binned_likelihood = jnp.concatenate([binned_east_scaled, binned_west_scaled], axis=0)

        if return_highres:
            result = {
                "hires_east": rt_out["hires_east"],
                "hires_west": rt_out["hires_west"],
                "hires_east_scaled": hires_east_scaled,
                "hires_west_scaled": hires_west_scaled,
                "binned_east": binned_east,
                "binned_west": binned_west,
                "binned_east_scaled": binned_east_scaled,
                "binned_west_scaled": binned_west_scaled,
                "binned_likelihood": binned_likelihood,
                "p_lay_east": state_east["p_lay"],
                "p_lay_west": state_west["p_lay"],
                "T_lay_east": state_east["T_lay"],
                "T_lay_west": state_west["T_lay"],
            }
            return result

        return binned_likelihood

    def forward_model(params: Dict[str, jnp.ndarray]):
        return _forward_model_impl(params, wl_hi, opac_cache, bandpass_cache)

    return forward_model
