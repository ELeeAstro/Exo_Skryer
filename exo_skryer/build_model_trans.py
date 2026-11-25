"""
build_model_trans.py
====================

Forward model for transmission spectra, for every geometry, chosen with
``physics.rt_scheme``:

- ``transit_1d``: one column covering the whole terminator.
- ``transit_1_5d``: two azimuthal sectors (east = evening, west = morning), each a
  half annulus with its own column. Parameters are tagged ``_joint/_east/_west``
  (``limb_asymmetry``).
- ``transit_2d`` / ``transit_3d``: planned (more sectors and zenith slices).

Every column is a full 1D atmosphere (T-p, chemistry, mu, hydrostatic altitude,
clouds, opacities) built with the shared kernels in ``build_common``; columns are
vmapped. The radiative transfer goes through the multi-column core
(``RT_trans_core_os.sector_dR2`` / ``RT_trans_core_ck.sector_dR2_ro``) on a ray grid
shared by all columns (``RT_trans_geometry``), so 1D and multi-D use one code path.
Partial cloud cover (``f_cloud``) mixes cloudy and clear columns per sector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from . import build_opacities as XS
from .build_chem import prepare_chemistry_kernel, init_chemistry
from .build_common import (
    CkMix,
    attach_runtime,
    check_model_traces,
    extract_fixed_params,
    make_atmosphere_fn,
    make_mu_kernel,
    make_opacity_fn,
    select_model,
    validate_model,
)
from .instru_convolve import apply_response_functions_cached, get_bandpass_cache
from .limb_asymmetry import merge_limb_parameter_dict, split_limb_parameter_dict
from .RT_trans_core_ck import column_dtau_ck, column_dtau_ck_trans, sector_dR2_ro, _get_ck_weights
from .RT_trans_core_os import column_dtau_os, sector_dR2
from .RT_trans_geometry import FULL_CHORD

__all__ = ["build_forward_model_trans"]

# Sector layouts: names and azimuthal fractions of the annulus.
_LAYOUTS = {
    "1d": (("all",), (1.0,)),
    "1_5d": (("east", "west"), (0.5, 0.5)),
}


def _column_params(geometry: str, params: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Parameters of every column, stacked on a leading column axis."""
    if geometry == "1d":
        return tree_util.tree_map(lambda x: jnp.asarray(x)[None], params)
    joint, east, west = split_limb_parameter_dict(params)
    p_e, p_w = merge_limb_parameter_dict(joint, east), merge_limb_parameter_dict(joint, west)
    if set(p_e) != set(p_w):
        raise ValueError(
            "East/west limb parameters must share identical base names. "
            f"Missing east={sorted(set(p_w) - set(p_e))}, missing west={sorted(set(p_e) - set(p_w))}"
        )
    return {k: jnp.stack([jnp.asarray(p_e[k]), jnp.asarray(p_w[k])]) for k in sorted(p_e)}


def build_forward_model_trans(
    cfg,
    obs: Dict,
    stellar_flux: Optional[np.ndarray] = None,
    return_highres: bool = False,
    exp_dir: Optional[Path] = None,
) -> Callable[[Dict[str, jnp.ndarray]], Union[jnp.ndarray, Dict[str, jnp.ndarray]]]:
    """Build the JIT-compiled transmission forward model (see module docstring).

    Returns ``forward_model(params)`` giving the binned spectrum the likelihood uses
    (for 1.5D with limb observations: the evening then morning half-annulus spectra), or
    with ``return_highres`` a dict of high-resolution and binned spectra and diagnostics.
    """
    del stellar_flux  # transmission does not use the stellar spectrum
    rt_scheme = str(cfg.physics.rt_scheme).strip().lower()
    geometry = "1_5d" if rt_scheme == "transit_1_5d" else "1d"
    spec = select_model(cfg, geometry)
    init_chemistry(cfg, exp_dir)
    opac_cache = XS.runtime_opacity_cache()
    validate_model(cfg, spec, opac_cache)

    names, fracs = _LAYOUTS[geometry]
    n_sector = len(names)
    sector_frac = jnp.asarray(fracs)
    theta_edges = FULL_CHORD                  # one zenith slice per sector (for now)
    has_limb_obs = geometry == "1_5d" and bool(obs.get("has_limb_observations", False))
    mode = "os" if not spec.ck else ("ck_trans" if spec.ck_mix is CkMix.TRANS else "ck")
    want_contrib = spec.contri_func

    nlay = int(getattr(cfg.physics, "nlay", 99))
    wl_hi = jnp.asarray(np.asarray(XS.master_wavelength_cut(), dtype=float))
    bandpass_cache = get_bandpass_cache()
    chem, trace_species = prepare_chemistry_kernel(cfg, spec.chem, spec.opacity_schemes())
    atmosphere = make_atmosphere_fn(spec, chem, make_mu_kernel(cfg, spec, trace_species), nlay)
    opacities = make_opacity_fn(spec)
    fixed_params = extract_fixed_params(cfg, geometry)

    @jax.jit
    def _forward_model_impl(params, wl, opac, bandpass):
        def build_column(p):
            state, p = atmosphere(p, wl, opac)
            return state, p, opacities(state, p, opac)

        states, col_params, comps = jax.vmap(build_column)(_column_params(geometry, {**fixed_params, **params}))
        R_s = states["R_s"][0]
        r_low = (states["R0"][:, None] + states["z_lev"][:, :-1])[:, None, :]   # (nsector, nslice=1, nlay)
        r_up = (states["R0"][:, None] + states["z_lev"][:, 1:])[:, None, :]
        g_weights = _get_ck_weights(opac)[: comps["line"].shape[-1]] if mode == "ck" else None
        partial_cloud = "f_cloud" in col_params and "cloud" in comps

        def sector_areas(clear: bool):
            """Per-sector absorbing area (nsector, nwl[, ng]), per-cell contribution, R_base."""
            if mode == "ck_trans":
                dcont, dspec = jax.vmap(lambda s, c: column_dtau_ck_trans(s, c, clear))(states, comps)
                dR2, R_base = sector_dR2_ro(r_low, r_up, theta_edges,
                                            lambda j: (dcont[j][None], dspec[j][:, None]), _get_ck_weights(opac))
                return dR2, None, R_base
            if mode == "ck":
                dtau = jax.vmap(lambda s, c: column_dtau_ck(s, c, opac, clear))(states, comps)
            else:
                dtau = jax.vmap(lambda s, c: column_dtau_os(s, c, clear))(states, comps)
            return sector_dR2(r_low, r_up, theta_edges, lambda j: dtau[j][None], g_weights, want_contrib)

        dR2, contrib, R_base = sector_areas(clear=False)
        if partial_cloud:
            f = jnp.clip(col_params["f_cloud"], 0.0, 1.0)                        # (nsector,)
            dR2_clear, contrib_clear, _ = sector_areas(clear=True)
            fb = f.reshape((-1,) + (1,) * (dR2.ndim - 1))
            dR2 = fb * dR2 + (1.0 - fb) * dR2_clear
            if want_contrib:
                fc = f.reshape((-1,) + (1,) * (contrib.ndim - 1))
                contrib = fc * contrib + (1.0 - fc) * contrib_clear

        # Each sector's share of the transit depth; they sum to the total.
        frac = sector_frac.reshape((-1,) + (1,) * (dR2.ndim - 1))
        D_sector = frac * (R_base**2 + dR2) / (R_s**2)                       # (nsector, nwl[, ng])
        if g_weights is not None:
            D_sector = jnp.sum(D_sector * g_weights.astype(D_sector.dtype), axis=-1)
        D_hires = jnp.sum(D_sector, axis=0)

        if geometry == "1d":
            D_bin = apply_response_functions_cached(D_hires, bandpass)
            if not return_highres:
                return D_bin
            state0 = tree_util.tree_map(lambda x: x[0], states)
            comps0 = tree_util.tree_map(lambda x: x[0], comps)
            result = {"hires": D_hires, "binned": D_bin}
            atm = {k: state0[k] for k in ("p_lev", "T_lev", "p_lay", "T_lay", "mu_lay", "q_c_lay", "Kzz_lay")
                   if k in state0}
            atm["vmr"] = {k: v for k, v in state0["vmr_lay"].items() if not k.startswith("__")}
            result["atm"] = atm
            if spec.cloud is not None:
                result["cloud_optics"] = {k: comps0[k] for k in ("cloud", "cloud_ssa", "cloud_g")}
            if want_contrib:
                dR2_w = jnp.sum(dR2 * g_weights, axis=-1) if g_weights is not None else dR2
                result["contrib_func"] = contrib[0, 0] / jnp.maximum(dR2_w[0][None, :], 1e-30)
                result["p_lay"] = state0["p_lay"]
            return result

        # 1.5D: sector j's depth is its half-annulus share (= 0.5 x its full-circle depth).
        binned_sector = jax.vmap(lambda d: apply_response_functions_cached(d, bandpass))(D_sector)
        binned_combined = jnp.sum(binned_sector, axis=0)
        if has_limb_obs:
            binned_likelihood = jnp.concatenate(
                [binned_sector[0][obs["east_slice"]], binned_sector[1][obs["west_slice"]]], axis=0)
        else:
            binned_likelihood = binned_combined
        if not return_highres:
            return binned_likelihood
        return {
            "hires_east": D_sector[0] / fracs[0], "hires_west": D_sector[1] / fracs[1],
            "hires_combined": D_hires,
            "hires_east_scaled": D_sector[0], "hires_west_scaled": D_sector[1],
            "binned_east": binned_sector[0] / fracs[0], "binned_west": binned_sector[1] / fracs[1],
            "binned_combined": binned_combined,
            "binned_east_scaled": binned_sector[0], "binned_west_scaled": binned_sector[1],
            "binned_likelihood": binned_likelihood,
            "p_lay_east": states["p_lay"][0], "p_lay_west": states["p_lay"][1],
            "T_lay_east": states["T_lay"][0], "T_lay_west": states["T_lay"][1],
        }

    runtime_args = (wl_hi, opac_cache, bandpass_cache)
    check_model_traces(cfg, _forward_model_impl, *runtime_args)

    def forward_model(params: Dict[str, jnp.ndarray]):
        return _forward_model_impl(params, *runtime_args)

    return attach_runtime(forward_model, _forward_model_impl, runtime_args)
