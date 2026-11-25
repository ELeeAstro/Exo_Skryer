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

With the TRIDENT parameterisation (``vert_Tp: trident``) every column comes from the
TRIDENT fields; evening/morning spectra and Iceberg clouds (``physics.iceberg_cloud``)
come from full-circle azimuthal weights (``RT_trans_geometry.azimuth_weights``) applied
to clear and cloudy versions of each sector.
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
from .data_constants import bar
from .instru_convolve import apply_response_functions_cached, get_bandpass_cache
from .vert_mu import compute_mu
from .limb_asymmetry import ANCHOR_TAGS, merge_limb_parameter_dict, split_limb_parameter_dict
from .RT_trans_core_ck import column_dtau_ck, column_dtau_ck_trans, sector_dR2_ro, _get_ck_weights
from .RT_trans_core_os import column_dtau_os, sector_dR2
from .RT_trans_geometry import FULL_CHORD, azimuth_weights, day_night_slices, insert_zone_edge

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


# ---------------------------------------------------------------------------
# TRIDENT parameterisation (MacDonald & Lewis 2022, Eqs. 34-36; POSEIDON
# compute_T_field_gradient / compute_X_field_gradient / angular_grids)
# ---------------------------------------------------------------------------

TRIDENT_P_HIGH = 1.0e-5    # bar: columns isothermal / isochemical above this
TRIDENT_P_DEEP = 10.0      # bar: and below this
_TRIDENT_X_PREFIX = "log_X_bar_term_"


def trident_angles(n_slice: Optional[int], opening_deg):
    """Edges, centres and widths of TRIDENT sectors (phi) or zones (theta), POSEIDON layout.

    ``n_slice`` None: one region spanning (-pi/2, pi/2). Otherwise 2 + n_slice regions: the
    two outer ones and ``n_slice`` equal slices across the opening angle (degrees) centred
    on 0. Northern hemisphere only (north-south symmetry).
    """
    half = 0.5 * jnp.pi
    if n_slice is None:
        edges = jnp.array([-half, half])
    else:
        a = jnp.deg2rad(opening_deg)
        inner = -0.5 * a + jnp.arange(n_slice + 1) * (a / n_slice)
        edges = jnp.concatenate([jnp.array([-half]), inner, jnp.array([half])])
    widths = jnp.diff(edges)
    centres = -half + jnp.cumsum(widths) - 0.5 * widths
    return edges, centres, widths


def _trident_angle_value(angle, half_open, low_value, mid, delta_half, high_value):
    """POSEIDON's piecewise rule: low for angle <= -a/2, linear for |angle| < a/2, high otherwise."""
    lin = mid - (angle / jnp.where(half_open > 0, half_open, 1.0)) * delta_half
    return jnp.where(angle <= -half_open, low_value,
                     jnp.where((angle > -half_open) & (angle < half_open), lin, high_value))


def trident_high_value(phi, theta, bar, d_term, d_dn, alpha_deg, beta_deg):
    """Value at P_high for a column at (phi, theta): Eqs. 34-35 (T, or log X for abundances)."""
    a2, b2 = 0.5 * jnp.deg2rad(alpha_deg), 0.5 * jnp.deg2rad(beta_deg)
    term = _trident_angle_value(phi, a2, bar + 0.5 * d_term, bar, 0.5 * d_term, bar - 0.5 * d_term)
    return _trident_angle_value(theta, b2, term + 0.5 * d_dn, term, 0.5 * d_dn, term - 0.5 * d_dn)


def trident_vertical(p_bar, high, deep):
    """Eq. 36: constant above P_high, linear in log P to ``deep`` at P_deep, constant below."""
    slope = (deep - high) / jnp.log10(TRIDENT_P_DEEP / TRIDENT_P_HIGH)
    mid = high + slope * jnp.log10(jnp.maximum(p_bar, 1e-300) / TRIDENT_P_HIGH)
    return jnp.where(p_bar <= TRIDENT_P_HIGH, high, jnp.where(p_bar < TRIDENT_P_DEEP, mid, deep))


def trident_species(params) -> tuple:
    """Species with TRIDENT abundance fields (from log_X_bar_term_<species> parameters)."""
    return tuple(sorted(k[len(_TRIDENT_X_PREFIX):] for k in params if k.startswith(_TRIDENT_X_PREFIX)))


def trident_column_profiles(p_lev, p_lay, phi, theta, params, species):
    """T(p) and VMR(p) of one column at (phi, theta) from the TRIDENT parameters.

    Missing Delta_* parameters are zero. Species without ``log_X_deep_<sp>`` are constant
    with altitude. H2 and He fill the rest with He/H2 = ``He_fraction`` (default 0.17).
    Pressures in dyne cm^-2.
    """
    get = lambda k, d=0.0: params.get(k, d)
    alpha, beta = get("alpha_em"), get("beta_dn")
    T_high = trident_high_value(phi, theta, params["T_bar_term"], get("Delta_T_term"), get("Delta_T_DN"),
                                alpha, beta)
    T_deep = params["T_deep"]
    p_lev_bar, p_lay_bar = p_lev / bar, p_lay / bar
    vmr, x_sum = {}, 0.0
    for sp in species:
        logX_high = trident_high_value(phi, theta, params[_TRIDENT_X_PREFIX + sp], get(f"Delta_log_X_term_{sp}"),
                                       get(f"Delta_log_X_DN_{sp}"), alpha, beta)
        deep_key = f"log_X_deep_{sp}"
        logX = trident_vertical(p_lay_bar, logX_high, params[deep_key]) if deep_key in params \
            else jnp.full_like(p_lay_bar, logX_high)
        vmr[sp] = 10.0 ** logX
        x_sum = x_sum + vmr[sp]
    he_frac = get("He_fraction", 0.17)
    vmr["H2"] = (1.0 - x_sum) / (1.0 + he_frac)
    vmr["He"] = he_frac * vmr["H2"]
    return {"p_lev": p_lev, "p_lay": p_lay,
            "T_lev": trident_vertical(p_lev_bar, T_high, T_deep),
            "T_lay": trident_vertical(p_lay_bar, T_high, T_deep),
            "vmr_lay": vmr}


def _flag(raw) -> bool:
    return str(raw).strip().lower() in ("true", "1", "yes", "on")


def iceberg_dtau(states, params):
    """Grey Iceberg-cloud optical depth (ncol, nlay) of every column (Eq. 47).

    kappa_cld = 10**log_10_k_cld [m^-1] fills P >= P_cld (P_cld = 10**log_10_p_cld bar).
    A layer cut by P_cld gets its fraction (in log P) below P_cld, so a P_cld on a level
    reproduces POSEIDON's layer-centre switch. Pressures in the state are dyne cm^-2,
    dz in cm (kappa converted to cm^-1).
    """
    kappa = 10.0 ** params["log_10_k_cld"] * 1.0e-2
    lp = jnp.log(states["p_lev"])                                       # (ncol, nlev), bottom first
    lp_cld = jnp.log(10.0 ** params["log_10_p_cld"] * bar)
    below = jnp.clip((lp[:, :-1] - lp_cld) / (lp[:, :-1] - lp[:, 1:]), 0.0, 1.0)
    return kappa * states["dz"] * below


def _trident_layout(geometry: str, cfg):
    """(n_slice_em or None, n_slice_dn or None) for a TRIDENT model of the given geometry."""
    def n(key):
        raw = getattr(cfg.physics, key, None)
        if raw is None or str(raw).strip().lower() in ("none", "off"):
            return None
        v = int(raw)
        if v < 0 or v % 2:
            raise ValueError(f"physics.{key} must be an even integer >= 0 (POSEIDON convention), or None.")
        return v
    if geometry == "1d":
        return None, None
    if geometry == "2d":
        return None, n("n_slice_dn")
    return n("n_slice_em"), n("n_slice_dn")


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

    trident = spec.schemes.get("vert_Tp") == "trident"
    names, fracs = _LAYOUTS[geometry]
    n_sector = len(names)
    sector_frac = jnp.asarray(fracs)
    n_slice = int(getattr(cfg.physics, "n_zenith", 6)) if geometry in ("2d", "3d") else 1
    if trident:
        n_em, n_dn = _trident_layout(geometry, cfg)
        n_sector = 1 if n_em is None else 2 + n_em
        n_slice = 1 if n_dn is None else 2 + n_dn
    iceberg = trident and _flag(getattr(cfg.physics, "iceberg_cloud", False))
    has_limb_obs = (geometry in ("1_5d", "3d") or trident) and bool(obs.get("has_limb_observations", False))
    mode = "os" if not spec.ck else ("ck_trans" if spec.ck_mix is CkMix.TRANS else "ck")
    want_contrib = spec.contri_func

    nlay = int(getattr(cfg.physics, "nlay", 99))
    wl_hi = jnp.asarray(np.asarray(XS.master_wavelength_cut(), dtype=float))
    bandpass_cache = get_bandpass_cache()
    if trident:
        mu_kernel = lambda params, vmr_lay, n: compute_mu(vmr_lay)
        atmosphere = profiles = None
    else:
        chem, trace_species = prepare_chemistry_kernel(cfg, spec.chem, spec.opacity_schemes())
        mu_kernel = make_mu_kernel(cfg, spec, trace_species)
        atmosphere = make_atmosphere_fn(spec, chem, mu_kernel, nlay)
        profiles = make_profile_fn(spec, chem, nlay)
    column = make_column_fn(spec, mu_kernel, nlay)
    opacities = make_opacity_fn(spec)
    fixed_params = extract_fixed_params(cfg, "1d" if trident else geometry)
    nlev = nlay + 1

    @jax.jit
    def _forward_model_impl(params, wl, opac, bandpass):
        all_params = {**fixed_params, **params}
        joint, anchors = (all_params, None) if trident else _anchor_params(geometry, all_params)

        # State and opacities in one vmapped function: static fields of the state (nlay, nwl,
        # flags) are traced once they leave a vmap, and the opacity kernels need them static.
        def with_opacities(state, p):
            return state, p, opacities(state, p, opac)

        frac_all = sector_frac
        if trident:
            # Every column from the TRIDENT field at its (phi, theta) centre; sector weights from alpha.
            _, phi_c, phi_w = trident_angles(n_em, all_params.get("alpha_em", 0.0))
            theta_edges, theta_c, _ = trident_angles(n_dn, all_params.get("beta_dn", 0.0))
            frac_all = phi_w / jnp.pi
            p_lev = jnp.logspace(jnp.log10(all_params["p_bot"] * bar), jnp.log10(all_params["p_top"] * bar), nlev)
            p_lay = (p_lev[1:] - p_lev[:-1]) / jnp.log(p_lev[1:] / p_lev[:-1])
            species = trident_species(all_params)
            phis = jnp.repeat(phi_c, n_slice)                     # sector-major
            thetas = jnp.tile(theta_c, n_sector)
            w_night = theta_c

            def trident_column(ph, th):
                prof = trident_column_profiles(p_lev, p_lay, ph, th, all_params, species)
                return with_opacities(*column(prof, all_params, wl, opac))

            states, col_params, comps = jax.vmap(trident_column)(phis, thetas)
        elif geometry in _DAY_NIGHT_PAIRS:
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

        def column_dtau(clear: bool):
            """Vertical optical depths of every column on the (nsector, nslice, ...) grid."""
            if mode == "ck_trans":
                dcont, dspec = jax.vmap(lambda s, c: column_dtau_ck_trans(s, c, clear))(states, comps)
                return grid(dcont), jnp.moveaxis(grid(dspec), 2, 1)          # (nsec, nspec, nslice, ...)
            if mode == "ck":
                return grid(jax.vmap(lambda s, c: column_dtau_ck(s, c, opac, clear))(states, comps))
            return grid(jax.vmap(lambda s, c: column_dtau_os(s, c, clear))(states, comps))

        def areas(dt, rl, ru, edges, zone=None, extra=None):
            """Per-sector absorbing area (nsector, nwl[, ng]), per-cell contribution, R_base.

            ``zone`` (nslice',) picks each slice's background column (an inserted zone edge);
            ``extra`` (nsector, nslice', nlay) is a grey optical depth added to the gas.
            """
            pick = (lambda x: x) if zone is None else (lambda x: x[zone])
            add = (lambda x, j: x) if extra is None else \
                (lambda x, j: x + extra[j].reshape(extra[j].shape + (1,) * (x.ndim - 2)))
            if mode == "ck_trans":
                dcont, dspec = dt
                dR2, R_base = sector_dR2_ro(rl, ru, edges,
                                            lambda j: (add(pick(dcont[j]), j), dspec[j][:, zone] if zone is not None
                                                       else dspec[j]), _get_ck_weights(opac))
                return dR2, None, R_base
            return sector_dR2(rl, ru, edges, lambda j: add(pick(dt[j]), j), g_weights, want_contrib)

        dtau = column_dtau(clear=False)
        dR2, contrib, R_base = areas(dtau, r_low, r_up, theta_edges)
        if partial_cloud:
            f = jnp.clip(grid(col_params["f_cloud"])[:, 0], 0.0, 1.0)            # (nsector,)
            dR2_clear, contrib_clear, _ = areas(column_dtau(clear=True), r_low, r_up, theta_edges)
            fb = f.reshape((-1,) + (1,) * (dR2.ndim - 1))
            dR2 = fb * dR2 + (1.0 - fb) * dR2_clear
            if want_contrib:
                fc = f.reshape((-1,) + (1,) * (contrib.ndim - 1))
                contrib = fc * contrib + (1.0 - fc) * contrib_clear

        def depth(dR2_s):
            """Full-circle-equivalent depth of each sector (nsector, nwl), g-weighted."""
            D = (R_base**2 + dR2_s) / (R_s**2)
            return jnp.sum(D * g_weights.astype(D.dtype), axis=-1) if g_weights is not None else D

        if trident:
            # Shares of the full annulus per (sector, clear/cloudy, evening/morning).
            phi_edges = trident_angles(n_em, all_params.get("alpha_em", 0.0))[0]
            if iceberg:
                W = azimuth_weights(phi_edges, jnp.deg2rad(all_params.get("phi_0_cld", -90.0)),
                                    all_params.get("f_cld", 1.0), cloud=True)
                edges_c, zone_c, cloudy_zone = insert_zone_edge(
                    theta_edges, jnp.deg2rad(all_params.get("theta_0_cld", -90.0)))
                tau_cld = grid(iceberg_dtau(states, all_params))[:, zone_c] * cloudy_zone[None, :, None]
                dR2_cld, _, _ = areas(dtau, r_low[:, zone_c], r_up[:, zone_c], edges_c, zone_c, tau_cld)
                D_cs = jnp.stack([depth(dR2), depth(dR2_cld)], axis=1)        # (nsec, 2, nwl)
            else:
                W = azimuth_weights(phi_edges)
                D_cs = depth(dR2)[:, None]
                W = W[:, :1]
            D_limb = jnp.einsum("jce,jc...->e...", W, D_cs)                  # (2, nwl): evening, morning
            frac_all = jnp.sum(W, axis=(1, 2))
            D_hires = jnp.sum(D_limb, axis=0)
        else:
            # Each sector's share of the transit depth; they sum to the total.
            frac = frac_all.reshape((-1,) + (1,) * (dR2.ndim - 1))
            D_sector = frac * (R_base**2 + dR2) / (R_s**2)                   # (nsector, nwl[, ng])
            if g_weights is not None:
                D_sector = jnp.sum(D_sector * g_weights.astype(D_sector.dtype), axis=-1)
            D_hires = jnp.sum(D_sector, axis=0)

        if geometry == "1d" and not trident:
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

        if trident:
            # Evening / morning half-annulus shares (each a 1/2 of the annulus), like 1.5D east / west.
            binned_limb = jax.vmap(lambda d: apply_response_functions_cached(d, bandpass))(D_limb)
            D_bin = jnp.sum(binned_limb, axis=0)
            if has_limb_obs:
                likelihood = jnp.concatenate([binned_limb[0][obs["east_slice"]], binned_limb[1][obs["west_slice"]]])
            else:
                likelihood = D_bin
            if not return_highres:
                return likelihood
            diag = {"r_low_columns": r_low, "r_up_columns": r_up, "sector_frac": frac_all, "azimuth_weights": W}
            if mode == "os":
                diag["dtau_columns"] = dtau
            if iceberg:
                diag["dtau_cloud_columns"] = grid(iceberg_dtau(states, all_params))
            return {
                "hires": D_hires, "binned": D_bin, "binned_likelihood": likelihood,
                "hires_east": 2.0 * D_limb[0], "hires_west": 2.0 * D_limb[1],
                "hires_east_scaled": D_limb[0], "hires_west_scaled": D_limb[1],
                "binned_east": 2.0 * binned_limb[0], "binned_west": 2.0 * binned_limb[1],
                "binned_east_scaled": binned_limb[0], "binned_west_scaled": binned_limb[1],
                "theta_edges": theta_edges, "night_weight": w_night,
                "p_lay": states["p_lay"][0], "T_lay_columns": states["T_lay"],
                "z_lev_columns": states["z_lev"], **diag,
            }

        if geometry == "2d":
            D_bin = apply_response_functions_cached(D_hires, bandpass)
            if not return_highres:
                return D_bin
            diag = {"r_low_columns": r_low, "r_up_columns": r_up, "sector_frac": frac_all}
            if mode == "os":
                diag["dtau_columns"] = dtau
            return {
                "hires": D_hires, "binned": D_bin,
                "theta_edges": theta_edges, "night_weight": w_night,
                "p_lay": states["p_lay"][0], "T_lay_columns": states["T_lay"],
                "z_lev_columns": states["z_lev"], **diag,
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
