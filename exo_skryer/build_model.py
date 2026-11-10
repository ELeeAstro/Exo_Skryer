"""
build_model.py
==============
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Callable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from . import build_opacities as XS
from .build_chem import prepare_chemistry_kernel, init_chemistry
from .build_common import (
    select_model,
    validate_model,
    extract_fixed_params,
    make_mu_kernel,
    make_atmosphere_fn,
    make_opacity_fn,
    check_model_traces,
    attach_runtime,
)
from .instru_convolve import apply_response_functions_cached, get_bandpass_cache

__all__ = [
    'build_forward_model'
]


def build_forward_model(
    cfg,
    obs: Dict,
    stellar_flux: Optional[np.ndarray] = None,
    return_highres: bool = False,
    exp_dir: Optional[Path] = None,
) -> Callable[[Dict[str, jnp.ndarray]], Union[jnp.ndarray, Dict[str, jnp.ndarray]]]:
    """Build a JIT-compiled forward model for atmospheric retrieval.

    The model runs ``atmosphere -> opacities -> radiative transfer -> instrument
    binning`` using the kernels named in ``cfg.physics`` (see
    :mod:`exo_skryer.kernel_registry`).

    Parameters
    ----------
    cfg : config object
        Configuration object containing physics settings (`cfg.physics`),
        opacity configuration (`cfg.opac`), and retrieval parameters (`cfg.params`).
    obs : dict
        Observational data dictionary (used by the 1.5D model for limb slices).
    stellar_flux : `~numpy.ndarray`, optional
        Stellar flux on the high-resolution grid. Planet emission needs either
        this or an ``F_star`` parameter.
    return_highres : bool, optional
        If True, return ``{'hires': D_hires, 'binned': D_bin}`` (plus
        ``contrib_func`` and ``p_lay`` when contribution functions are on).
        Otherwise return only the binned spectrum.
    exp_dir : `~pathlib.Path`, optional
        Experiment directory used to resolve relative chemistry data paths
        (NASA-9, FastChem grid, atmodeller). Defaults to the current working
        directory when the data has not already been loaded.

    Returns
    -------
    forward_model : callable
        ``forward_model(params) -> spectrum`` (or dict, see ``return_highres``).
        Build-time checks raise if the config or loaded data are inconsistent.
    """
    if str(getattr(cfg.physics, "rt_scheme", None)).lower() == "transit_1_5d":
        from .build_model_1_5D import build_forward_model_1_5d
        return build_forward_model_1_5d(
            cfg,
            obs,
            stellar_flux=stellar_flux,
            return_highres=return_highres,
            exp_dir=exp_dir,
        )

    spec = select_model(cfg, "1d")
    # Setup loaders first: user profiles set nlay and add cloud tables to the cache.
    init_chemistry(cfg, exp_dir)
    opac_cache = XS.runtime_opacity_cache()
    validate_model(cfg, spec, opac_cache)

    # Planet emission needs stellar normalisation up front (no fallback inside JIT).
    if spec.rt_scheme == "emission_1d" and not spec.is_brown_dwarf and stellar_flux is None:
        if "F_star" not in {p.name for p in cfg.params}:
            raise ValueError("Planet emission mode requires either stellar_flux input or parameter 'F_star'.")

    nlay = int(getattr(cfg.physics, "nlay", 99))
    wl_hi = jnp.asarray(np.asarray(XS.master_wavelength_cut(), dtype=float))
    has_stellar_flux = jnp.asarray(1 if stellar_flux is not None else 0, dtype=jnp.int32)
    if stellar_flux is not None:
        stellar_flux_arr = jnp.asarray(stellar_flux, dtype=jnp.float64)
    else:
        stellar_flux_arr = jnp.zeros_like(wl_hi, dtype=jnp.float64)
    bandpass_cache = get_bandpass_cache()

    chem, trace_species = prepare_chemistry_kernel(cfg, spec.chem, spec.opacity_schemes())
    mu = make_mu_kernel(cfg, spec, trace_species)
    atmosphere = make_atmosphere_fn(spec, chem, mu, nlay)
    opacities = make_opacity_fn(spec)
    fixed_params = extract_fixed_params(cfg)

    @jax.jit
    def _forward_model_impl(params, wl, stellar_flux_runtime, has_stellar_flux_runtime, opac, bandpass):
        state, full_params = atmosphere({**fixed_params, **params}, wl, opac)
        state["is_brown_dwarf"] = spec.is_brown_dwarf
        state["stellar_flux"] = stellar_flux_runtime
        state["has_stellar_flux"] = has_stellar_flux_runtime

        components = opacities(state, full_params, opac)
        # RT kernels return (spectrum, contrib_func); contrib_func is zeros when off.
        D_hires, contrib_func = spec.rt(state, full_params, components, opac)
        D_bin = apply_response_functions_cached(D_hires, bandpass)

        if return_highres:
            result = {"hires": D_hires, "binned": D_bin}
            # The atmosphere and cloud optics the RT used, for checking against inputs
            atm = {k: state[k] for k in ("p_lev", "T_lev", "p_lay", "T_lay", "mu_lay", "q_c_lay", "Kzz_lay")
                   if k in state}
            atm["vmr"] = {k: v for k, v in state["vmr_lay"].items() if not k.startswith("__")}
            result["atm"] = atm
            if spec.cloud is not None:
                result["cloud_optics"] = {k: components[k] for k in ("cloud", "cloud_ssa", "cloud_g")}
            if spec.contri_func:
                result["contrib_func"] = contrib_func
                result["p_lay"] = state["p_lay"]
            return result
        return D_bin

    runtime_args = (wl_hi, stellar_flux_arr, has_stellar_flux, opac_cache, bandpass_cache)
    check_model_traces(cfg, _forward_model_impl, *runtime_args)

    def forward_model(params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        return _forward_model_impl(params, *runtime_args)

    return attach_runtime(forward_model, _forward_model_impl, runtime_args)
