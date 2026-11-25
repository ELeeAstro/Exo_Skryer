"""
build_model_trans.py
====================

Forward model for transmission spectra, for every geometry, chosen with
``physics.rt_scheme``:

- ``transit_1d``: one column covering the whole terminator.
- ``transit_1_5d``: two azimuthal sectors (east = evening, west = morning), each a
  half annulus with its own column. Parameters are tagged ``_joint/_east/_west``
  (``limb_asymmetry``).
- ``transit_2d``: day-night terminator (TRIDENT): one sector (the full annulus) cut
  into ``physics.n_zenith`` slices along the line of sight: a dayside slice, a
  nightside slice and ``n_zenith - 2`` transition slices across the opening angle
  ``beta_dn_joint`` [deg]. Day and night anchor columns get the full T-p and
  chemistry from ``_day`` / ``_night`` / ``_joint`` parameters; transition columns
  interpolate T(p), log VMR(p) and the other per-column parameters linearly in angle.
- ``transit_3d``: east/west limbs (two half-annulus sectors), each with its own
  day-night structure: ``n_zenith`` slices per sector from the shared
  ``beta_dn_joint``. Four full-physics anchors (``_east_day``, ``_west_day``,
  ``_east_night``, ``_west_night``); each limb's transition columns interpolate
  between its day and night anchors. Fits limb-resolved spectra like 1.5D.
  (A smooth azimuthal transition between the limbs is planned.)

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
    make_column_fn,
    make_mu_kernel,
    make_opacity_fn,
    make_profile_fn,
    select_model,
    validate_model,
)
from .instru_convolve import apply_response_functions_cached, get_bandpass_cache
from .limb_asymmetry import ANCHOR_TAGS, merge_limb_parameter_dict, split_limb_parameter_dict
from .RT_trans_core_ck import column_dtau_ck, column_dtau_ck_trans, sector_dR2_ro, _get_ck_weights
from .RT_trans_core_os import column_dtau_os, sector_dR2
from .RT_trans_geometry import FULL_CHORD, day_night_slices

__all__ = ["build_forward_model_trans"]

# Sector layouts: names and azimuthal fractions of the annulus.
_LAYOUTS = {
    "1d": (("all",), (1.0,)),
    "1_5d": (("east", "west"), (0.5, 0.5)),
    "2d": (("all",), (1.0,)),
    "3d": (("east", "west"), (0.5, 0.5)),
}
# For day-night geometries: (day anchor, night anchor) index pairs, one per sector,
# in the order of ANCHOR_TAGS[transit_<geometry>].
_DAY_NIGHT_PAIRS = {"2d": ((0, 1),), "3d": ((0, 2), (1, 3))}
_VMR_FLOOR = 1e-300


def _anchor_params(geometry: str, params: Dict[str, jnp.ndarray]):
    """(joint params, parameters of every anchor column stacked on a leading axis)."""
    if geometry == "1d":
        return params, tree_util.tree_map(lambda x: jnp.asarray(x)[None], params)
    joint, *anchors = split_limb_parameter_dict(params, ANCHOR_TAGS[f"transit_{geometry}"])
    merged = [merge_limb_parameter_dict(joint, a) for a in anchors]
    keys = set(merged[0])
    if any(set(m) != keys for m in merged):
        raise ValueError("All anchor columns must share identical parameter base names: "
                         + "; ".join(str(sorted(set(m) ^ keys)) for m in merged))
    return joint, {k: jnp.stack([jnp.asarray(m[k]) for m in merged]) for k in sorted(keys)}


def _interp_columns(anchor: Dict, w: jnp.ndarray, prof: Dict):
    """Day (anchor 0) / night (anchor 1) interpolation to columns with night weights ``w``.

    T and scalar parameters are linear in w, VMRs log-linear; integer parameters and the
    pressure grid are taken from the dayside.
    """
    n = w.shape[0]

    def lin(x):
        x = jnp.asarray(x)
        if not jnp.issubdtype(x.dtype, jnp.floating):
            return jnp.broadcast_to(x[0], (n,) + x.shape[1:])
        ww = w.reshape((n,) + (1,) * (x.ndim - 1))
        return (1.0 - ww) * x[0] + ww * x[1]

    def loglin(x):
        x = jnp.asarray(x)
        ww = w.reshape((n,) + (1,) * (x.ndim - 1))
        lx = jnp.log(jnp.maximum(x, _VMR_FLOOR))
        return jnp.exp((1.0 - ww) * lx[0] + ww * lx[1])

    cols_prof = {
        "p_lev": jnp.broadcast_to(prof["p_lev"][0], (n,) + prof["p_lev"].shape[1:]),
        "p_lay": jnp.broadcast_to(prof["p_lay"][0], (n,) + prof["p_lay"].shape[1:]),
        "T_lev": lin(prof["T_lev"]),
        "T_lay": lin(prof["T_lay"]),
        "vmr_lay": {k: (lin(v) if k.startswith("__") else loglin(v)) for k, v in prof["vmr_lay"].items()},
    }
    return cols_prof, {k: lin(v) for k, v in anchor.items()}


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
    geometry = {"transit_1_5d": "1_5d", "transit_2d": "2d", "transit_3d": "3d"}.get(rt_scheme, "1d")
    spec = select_model(cfg, geometry)
    init_chemistry(cfg, exp_dir)
    opac_cache = XS.runtime_opacity_cache()
    validate_model(cfg, spec, opac_cache)

    names, fracs = _LAYOUTS[geometry]
    n_sector = len(names)
    sector_frac = jnp.asarray(fracs)
    n_slice = int(getattr(cfg.physics, "n_zenith", 6)) if geometry in ("2d", "3d") else 1
    has_limb_obs = geometry in ("1_5d", "3d") and bool(obs.get("has_limb_observations", False))
    mode = "os" if not spec.ck else ("ck_trans" if spec.ck_mix is CkMix.TRANS else "ck")
    want_contrib = spec.contri_func

    nlay = int(getattr(cfg.physics, "nlay", 99))
    wl_hi = jnp.asarray(np.asarray(XS.master_wavelength_cut(), dtype=float))
    bandpass_cache = get_bandpass_cache()
    chem, trace_species = prepare_chemistry_kernel(cfg, spec.chem, spec.opacity_schemes())
    mu_kernel = make_mu_kernel(cfg, spec, trace_species)
    atmosphere = make_atmosphere_fn(spec, chem, mu_kernel, nlay)
    profiles = make_profile_fn(spec, chem, nlay)
    column = make_column_fn(spec, mu_kernel, nlay)
    opacities = make_opacity_fn(spec)
    fixed_params = extract_fixed_params(cfg, geometry)

    @jax.jit
    def _forward_model_impl(params, wl, opac, bandpass):
        joint, anchors = _anchor_params(geometry, {**fixed_params, **params})

        # State and opacities in one vmapped function: static fields of the state (nlay, nwl,
        # flags) are traced once they leave a vmap, and the opacity kernels need them static.
        def with_opacities(state, p):
            return state, p, opacities(state, p, opac)

        if geometry in _DAY_NIGHT_PAIRS:
            # Anchor profiles, then each sector's columns interpolated from its day to its night anchor.
            theta_edges, w_night = day_night_slices(n_slice, jnp.deg2rad(joint["beta_dn"]))
            anchor_prof = jax.vmap(profiles)(anchors)
            pick = lambda tree, ij: tree_util.tree_map(lambda x: x[jnp.asarray(ij)], tree)
            per_sector = [_interp_columns(pick(anchors, ij), w_night, pick(anchor_prof, ij))
                          for ij in _DAY_NIGHT_PAIRS[geometry]]
            col_prof, col_in = tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *per_sector)
            states, col_params, comps = jax.vmap(
                lambda pr, p: with_opacities(*column(pr, p, wl, opac)))(col_prof, col_in)
        else:
            theta_edges = FULL_CHORD
            states, col_params, comps = jax.vmap(lambda p: with_opacities(*atmosphere(p, wl, opac)))(anchors)

        # Columns are sector-major: (nsector * nslice, ...) -> (nsector, nslice, ...).
        grid = lambda x: x.reshape((n_sector, n_slice) + x.shape[1:])
        R_s = states["R_s"][0]
        r_low = grid(states["R0"][:, None] + states["z_lev"][:, :-1])
        r_up = grid(states["R0"][:, None] + states["z_lev"][:, 1:])
        g_weights = _get_ck_weights(opac)[: comps["line"].shape[-1]] if mode == "ck" else None
        partial_cloud = "f_cloud" in col_params and "cloud" in comps

        def sector_areas(clear: bool):
            """Per-sector absorbing area (nsector, nwl[, ng]), per-cell contribution, R_base."""
            if mode == "ck_trans":
                dcont, dspec = jax.vmap(lambda s, c: column_dtau_ck_trans(s, c, clear))(states, comps)
                dcont, dspec = grid(dcont), jnp.moveaxis(grid(dspec), 2, 1)   # (nsec, nspec, nslice, ...)
                dR2, R_base = sector_dR2_ro(r_low, r_up, theta_edges,
                                            lambda j: (dcont[j], dspec[j]), _get_ck_weights(opac))
                return dR2, None, R_base
            if mode == "ck":
                dtau = jax.vmap(lambda s, c: column_dtau_ck(s, c, opac, clear))(states, comps)
            else:
                dtau = jax.vmap(lambda s, c: column_dtau_os(s, c, clear))(states, comps)
            dtau = grid(dtau)
            return sector_dR2(r_low, r_up, theta_edges, lambda j: dtau[j], g_weights, want_contrib)

        dR2, contrib, R_base = sector_areas(clear=False)
        if partial_cloud:
            f = jnp.clip(grid(col_params["f_cloud"])[:, 0], 0.0, 1.0)            # (nsector,)
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

        if geometry == "2d":
            D_bin = apply_response_functions_cached(D_hires, bandpass)
            if not return_highres:
                return D_bin
            return {
                "hires": D_hires, "binned": D_bin,
                "theta_edges": theta_edges, "night_weight": w_night,
                "p_lay": states["p_lay"][0], "T_lay_columns": states["T_lay"],
                "z_lev_columns": states["z_lev"],
            }

        # 1.5D / 3D: sector j's depth is its half-annulus share (= 0.5 x its full-circle depth).
        binned_sector = jax.vmap(lambda d: apply_response_functions_cached(d, bandpass))(D_sector)
        binned_combined = jnp.sum(binned_sector, axis=0)
        if has_limb_obs:
            binned_likelihood = jnp.concatenate(
                [binned_sector[0][obs["east_slice"]], binned_sector[1][obs["west_slice"]]], axis=0)
        else:
            binned_likelihood = binned_combined
        if not return_highres:
            return binned_likelihood
        extra = {}
        if geometry == "3d":
            extra = {"theta_edges": theta_edges, "night_weight": w_night,
                     "T_lay_columns": grid(states["T_lay"]), "z_lev_columns": grid(states["z_lev"])}
        return {**extra,
            "hires_east": D_sector[0] / fracs[0], "hires_west": D_sector[1] / fracs[1],
            "hires_combined": D_hires,
            "hires_east_scaled": D_sector[0], "hires_west_scaled": D_sector[1],
            "binned_east": binned_sector[0] / fracs[0], "binned_west": binned_sector[1] / fracs[1],
            "binned_combined": binned_combined,
            "binned_east_scaled": binned_sector[0], "binned_west_scaled": binned_sector[1],
            "binned_likelihood": binned_likelihood,
            "p_lay_east": grid(states["p_lay"])[0, 0], "p_lay_west": grid(states["p_lay"])[1, 0],
            "T_lay_east": grid(states["T_lay"])[0, 0], "T_lay_west": grid(states["T_lay"])[1, 0],
        }

    runtime_args = (wl_hi, opac_cache, bandpass_cache)
    check_model_traces(cfg, _forward_model_impl, *runtime_args)

    def forward_model(params: Dict[str, jnp.ndarray]):
        return _forward_model_impl(params, *runtime_args)

    return attach_runtime(forward_model, _forward_model_impl, runtime_args)
