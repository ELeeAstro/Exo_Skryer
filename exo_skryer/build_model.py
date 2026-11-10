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
from .wl_blocks import check_wavelength_axes, map_wavelength_blocks, slice_block, slice_opac

__all__ = [
    'build_forward_model'
]


def _parse_wl_block(cfg, spec, nwl: int) -> int:
    """``opac.wl_block``: wavelengths per block, 0 (default) = the whole grid at once."""
    raw = getattr(getattr(cfg, "opac", None), "wl_block", None)
    block = 0 if raw in (None, "None", "none") else int(raw)
    if block < 0:
        raise ValueError(f"opac.wl_block must be >= 0, got {block}.")
    if block == 0:
        return 0
    if spec.rt_scheme != "emission_1d":
        raise NotImplementedError("opac.wl_block is only implemented for rt_scheme: emission_1d so far.")
    if block >= nwl:
        print(f"[info] opac.wl_block={block} >= nwl={nwl}: running the whole grid as one block")
        return 0
    print(f"[info] wavelength blocks: {block} wavelengths per block, {-(-nwl // block)} blocks")
    return block


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

    init_chemistry(cfg, exp_dir)
    chem, trace_species = prepare_chemistry_kernel(cfg, spec.chem, spec.opacity_schemes())
    mu = make_mu_kernel(cfg, spec, trace_species)
    atmosphere = make_atmosphere_fn(spec, chem, mu, nlay)
    opacities = make_opacity_fn(spec)
    fixed_params = extract_fixed_params(cfg)
    wl_block = _parse_wl_block(cfg, spec, int(wl_hi.shape[0]))
    if wl_block:
        check_wavelength_axes(opac_cache, int(wl_hi.shape[0]))

    @jax.jit
    def _forward_model_impl(params, wl, stellar_flux_runtime, has_stellar_flux_runtime, opac, bandpass):
        state, full_params = atmosphere({**fixed_params, **params}, wl, opac)
        state["is_brown_dwarf"] = spec.is_brown_dwarf
        state["stellar_flux"] = stellar_flux_runtime
        state["has_stellar_flux"] = has_stellar_flux_runtime

        if wl_block:
            # Opacities + RT one block of wavelengths at a time (see wl_blocks.py).
            def column(start):
                state_b = {
                    **state,
                    "wl": slice_block(wl, start, wl_block, 0),
                    "nwl": wl_block,
                    "stellar_flux": slice_block(stellar_flux_runtime, start, wl_block, 0),
                }
                opac_b = slice_opac(opac, start, wl_block)
                if "cloud_nk_n" in opac_b:
                    state_b["cloud_nk_n"] = opac_b["cloud_nk_n"]
                    state_b["cloud_nk_k"] = opac_b["cloud_nk_k"]
                D_b, cf_b = spec.rt(state_b, full_params, opacities(state_b, full_params, opac_b), opac_b)
                return (D_b, cf_b) if spec.contri_func else (D_b,)

            outs = map_wavelength_blocks(column, wl.shape[0], wl_block)
            D_hires = outs[0]
            contrib_func = outs[1] if spec.contri_func else None
        else:
            components = opacities(state, full_params, opac)
            # RT kernels return (spectrum, contrib_func); contrib_func is zeros when off.
            D_hires, contrib_func = spec.rt(state, full_params, components, opac)
        D_bin = apply_response_functions_cached(D_hires, bandpass)

        if return_highres:
            result = {"hires": D_hires, "binned": D_bin}
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
