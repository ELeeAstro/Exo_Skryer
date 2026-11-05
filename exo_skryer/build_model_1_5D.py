"""
build_model_1_5D.py
=================
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from .instru_convolve import apply_response_functions_cached, get_bandpass_cache
from .build_chem import prepare_chemistry_kernel, init_chemistry
from .build_common import (
    select_model,
    validate_model,
    extract_fixed_params,
    make_mu_kernel,
    make_atmosphere_fn,
    make_opacity_fn,
    check_model_traces,
)
from .limb_asymmetry import split_limb_parameter_dict, merge_limb_parameter_dict
from . import build_opacities as XS

__all__ = ["build_forward_model_1_5d"]


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


def build_forward_model_1_5d(
    cfg,
    obs: Dict,
    stellar_flux: Optional[np.ndarray] = None,
    return_highres: bool = False,
    exp_dir: Optional[Path] = None,
) -> Callable[[Dict[str, jnp.ndarray]], Union[jnp.ndarray, Dict[str, jnp.ndarray]]]:
    """Two-limb (east/west) transit model; each limb runs the 1D atmosphere and opacities."""
    spec = select_model(cfg, "1_5d")
    opac_cache = XS.runtime_opacity_cache()
    validate_model(cfg, spec, opac_cache)

    nlay = int(getattr(cfg.physics, "nlay", 99))
    has_limb_observations = bool(obs.get("has_limb_observations", False))
    wl_hi = jnp.asarray(np.asarray(XS.master_wavelength_cut(), dtype=float))
    bandpass_cache = get_bandpass_cache()

    init_chemistry(cfg, exp_dir)
    chem, trace_species = prepare_chemistry_kernel(cfg, spec.chem, spec.opacity_schemes())
    mu = make_mu_kernel(cfg, spec, trace_species)
    atmosphere = make_atmosphere_fn(spec, chem, mu, nlay)
    opacities = make_opacity_fn(spec)
    fixed_params = extract_fixed_params(cfg, "1_5d")

    @jax.jit
    def _forward_model_impl(params, wl, opac, bandpass):
        joint, east, west = split_limb_parameter_dict({**fixed_params, **params})
        params_east = merge_limb_parameter_dict(joint, east)
        params_west = merge_limb_parameter_dict(joint, west)

        def build_limb(params_limb):
            state, params_limb = atmosphere(params_limb, wl, opac)
            return state, params_limb, opacities(state, params_limb, opac)

        stacked = jax.vmap(build_limb)(_stack_two_param_dicts(params_east, params_west))
        (state_east, state_west), (params_east, params_west), (opacity_east, opacity_west) = (
            _split_limb_pytree(tree) for tree in stacked
        )

        rt_out = spec.rt(state_east, params_east, opacity_east, state_west, params_west, opacity_west, opac)

        hires_stacked = jnp.stack([rt_out["hires_east"], rt_out["hires_west"]], axis=0)
        binned_stacked = jax.vmap(lambda spectrum: apply_response_functions_cached(spectrum, bandpass))(hires_stacked)
        binned_east, binned_west = binned_stacked[0], binned_stacked[1]
        hires_east_scaled = 0.5 * rt_out["hires_east"]
        hires_west_scaled = 0.5 * rt_out["hires_west"]
        binned_east_scaled = 0.5 * binned_east
        binned_west_scaled = 0.5 * binned_west
        if has_limb_observations:
            binned_likelihood = jnp.concatenate(
                [binned_east_scaled[obs["east_slice"]], binned_west_scaled[obs["west_slice"]]],
                axis=0,
            )
        else:
            binned_likelihood = binned_east_scaled + binned_west_scaled

        if return_highres:
            return {
                "hires_east": rt_out["hires_east"],
                "hires_west": rt_out["hires_west"],
                "hires_combined": hires_east_scaled + hires_west_scaled,
                "hires_east_scaled": hires_east_scaled,
                "hires_west_scaled": hires_west_scaled,
                "binned_east": binned_east,
                "binned_west": binned_west,
                "binned_combined": binned_east_scaled + binned_west_scaled,
                "binned_east_scaled": binned_east_scaled,
                "binned_west_scaled": binned_west_scaled,
                "binned_likelihood": binned_likelihood,
                "p_lay_east": state_east["p_lay"],
                "p_lay_west": state_west["p_lay"],
                "T_lay_east": state_east["T_lay"],
                "T_lay_west": state_west["T_lay"],
            }
        return binned_likelihood

    runtime_args = (wl_hi, opac_cache, bandpass_cache)
    check_model_traces(cfg, _forward_model_impl, *runtime_args)

    def forward_model(params: Dict[str, jnp.ndarray]):
        return _forward_model_impl(params, *runtime_args)

    return forward_model
