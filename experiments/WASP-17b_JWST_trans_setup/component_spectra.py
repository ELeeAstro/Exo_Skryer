#!/usr/bin/env python3
"""
component_spectra.py - Calculate per-species/component transmission spectra.

This script calculates high-resolution transmission spectra for each opacity
component individually (each gas species, each Rayleigh species, each CIA pair,
cloud, H-) and outputs them to a single NPZ file.

Usage:
    python component_spectra.py --config retrieval_config.yaml
    python component_spectra.py --config retrieval_config.yaml --csv nested_samples.csv

Outputs:
    component_spectra.npz containing:
    - wavelength_um: high-res wavelength grid
    - line_H2O, line_CO2, ...: transit depth per line species
    - ray_H2, ray_He, ...: transit depth per Rayleigh species
    - cia_H2-H2, cia_H2-He, ...: transit depth per CIA pair
    - cloud: transit depth from cloud opacity
    - special_H-: transit depth from H- opacity
    - total: total transit depth (all components)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from types import SimpleNamespace

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp


# ============================================================================
# Helper functions (adapted from bestfit_plot.py)
# ============================================================================


def _to_ns(x):
    """Convert nested dicts to SimpleNamespace recursively."""
    if isinstance(x, list):
        return [_to_ns(v) for v in x]
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    return x


def _read_cfg(path: Path):
    """Load YAML config as SimpleNamespace."""
    with path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return _to_ns(y)


def _resolve_path_relative(path_str: str, exp_dir: Path) -> Path:
    """Resolve relative paths against experiment directory and ancestors."""
    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return path_obj
    base_dirs = [exp_dir]
    base_dirs.extend(exp_dir.parents)
    candidates: List[Path] = []
    for base in base_dirs:
        candidate = (base / path_obj).resolve()
        if candidate in candidates:
            continue
        candidates.append(candidate)
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path_obj


def _is_fixed_param(p) -> bool:
    """Check if parameter is fixed/delta."""
    dist = str(getattr(p, "dist", "")).lower()
    return dist == "delta" or bool(getattr(p, "fixed", False))


def _fixed_value_param(p):
    """Get fixed parameter value."""
    val = getattr(p, "value", None)
    if val is not None:
        return float(val)
    init = getattr(p, "init", None)
    if init is not None:
        return float(init)
    return None


def _load_observed(exp_dir: Path, cfg):
    """Load observed data, return lam, dlam, y, dy, response_mode."""
    csv_path = exp_dir / "observed_data.csv"
    if csv_path.exists():
        arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=str)
        if arr.ndim == 1:
            arr = arr[None, :]
        lam = arr[:, 0].astype(float)
        dlam = arr[:, 1].astype(float)
        y = arr[:, 2].astype(float) if arr.shape[1] >= 3 else None
        dy = arr[:, 3].astype(float) if arr.shape[1] >= 4 else None
        resp = arr[:, 4] if arr.shape[1] >= 5 else np.full_like(lam, "boxcar", dtype=object)
        return lam, dlam, y, dy, resp

    # Fallback to raw obs file
    data_cfg = getattr(cfg, "data", None)
    obs_path = getattr(data_cfg, "obs", None) if data_cfg is not None else None
    if obs_path is None:
        obs_cfg = getattr(cfg, "obs", None)
        obs_path = getattr(obs_cfg, "path", None) if obs_cfg is not None else None
    if obs_path is None:
        raise FileNotFoundError("No observed_data.csv and cfg.data.obs/cfg.obs.path missing.")

    data_path = _resolve_path_relative(obs_path, exp_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    arr = np.loadtxt(data_path, dtype=str, comments='#', ndmin=2)
    if arr.ndim == 1:
        arr = arr[None, :]

    lam = arr[:, 0].astype(float)
    if arr.shape[1] >= 5:
        dlam = arr[:, 1].astype(float)
        y = arr[:, 2].astype(float)
        dy = arr[:, 3].astype(float)
        resp = arr[:, 4]
    elif arr.shape[1] == 4:
        dlam = arr[:, 1].astype(float)
        y = arr[:, 2].astype(float)
        dy = arr[:, 3].astype(float)
        resp = np.full_like(lam, "boxcar", dtype=object)
    elif arr.shape[1] == 3:
        dlam = np.zeros_like(lam)
        y = arr[:, 1].astype(float)
        dy = arr[:, 2].astype(float)
        resp = np.full_like(lam, "boxcar", dtype=object)
    else:
        dlam = np.zeros_like(lam)
        y = arr[:, 1].astype(float) if arr.shape[1] >= 2 else None
        dy = None
        resp = np.full_like(lam, "boxcar", dtype=object)

    return lam, dlam, y, dy, resp


def _flatten_param(a: np.ndarray) -> np.ndarray:
    """Flatten chains/draws to 1D (N,) vector."""
    arr = np.asarray(a)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.reshape(-1)
    if arr.ndim >= 3:
        tail = arr.shape[2:]
        if np.prod(tail) == 1:
            return arr.reshape(-1)
        return arr.reshape(arr.shape[0], arr.shape[1], -1)[..., 0].reshape(-1)
    raise ValueError(f"Unsupported param shape {arr.shape}")


def _build_param_draws_from_idata(posterior_ds, params_cfg: List[SimpleNamespace]):
    """Build param draws from ArviZ posterior dataset."""
    out: Dict[str, np.ndarray] = {}
    if "chain" not in posterior_ds.dims or "draw" not in posterior_ds.dims:
        raise ValueError("posterior.nc must have dims ('chain', 'draw').")
    n_chain = int(posterior_ds.sizes["chain"])
    n_draw = int(posterior_ds.sizes["draw"])
    N_total = n_chain * n_draw

    for p in params_cfg:
        name = getattr(p, "name", None)
        if not name:
            continue
        if name in posterior_ds.data_vars:
            arr = posterior_ds[name].values
            out[name] = _flatten_param(arr)
        elif _is_fixed_param(p):
            val = _fixed_value_param(p)
            if val is None:
                raise ValueError(f"Fixed param '{name}' needs value/init in YAML.")
            out[name] = np.full((N_total,), float(val), dtype=float)
        else:
            raise KeyError(f"Free parameter '{name}' not found in posterior.nc.")

    return out, N_total


def _build_param_draws_from_csv(csv_path: Path, params_cfg: List[SimpleNamespace]):
    """Build param draws from nested_samples.csv."""
    df = pd.read_csv(csv_path, skiprows=2)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    out: Dict[str, np.ndarray] = {}
    N_total = len(df)

    for p in params_cfg:
        name = getattr(p, "name", None)
        if not name:
            continue
        if name in df.columns:
            out[name] = df[name].values
        elif _is_fixed_param(p):
            val = _fixed_value_param(p)
            if val is None:
                raise ValueError(f"Fixed param '{name}' needs value/init in YAML.")
            out[name] = np.full((N_total,), float(val), dtype=float)
        else:
            raise KeyError(f"Parameter '{name}' not found in CSV.")

    return out, N_total


def _bump_opacity_paths_one_level(cfg, exp_dir: Path):
    """Resolve relative opacity paths against exp_dir."""
    opac = getattr(cfg, "opac", None)
    if opac is None:
        return

    for attr in dir(opac):
        if attr.startswith("_"):
            continue
        val = getattr(opac, attr)
        if isinstance(val, list):
            for spec in val:
                if hasattr(spec, "path"):
                    p = getattr(spec, "path")
                    if not p:
                        continue
                    path_obj = Path(str(p))
                    if path_obj.is_absolute():
                        continue
                    resolved = _resolve_path_relative(str(p), exp_dir)
                    setattr(spec, "path", str(resolved))
        elif hasattr(val, "path"):
            p = getattr(val, "path")
            if not p:
                continue
            path_obj = Path(str(p))
            if path_obj.is_absolute():
                continue
            resolved = _resolve_path_relative(str(p), exp_dir)
            setattr(val, "path", str(resolved))


# ============================================================================
# Per-species opacity functions
# ============================================================================


def compute_line_opacity_per_species(state, params):
    """
    Compute line opacity for each species separately.

    Returns
    -------
    dict : {f"line_{species}": opacity_array (nlay, nwl)}
    """
    from exo_skryer import build_opacities as XS
    from exo_skryer.data_constants import amu, bar

    if not XS.has_line_data():
        return {}

    layer_pressures = state["p_lay"]
    layer_temperatures = state["T_lay"]
    layer_mu = state["mu_lay"]
    layer_vmr = state["vmr_lay"]
    layer_count = layer_pressures.shape[0]

    species_names = XS.line_species_names()
    sigma_cube = XS.line_sigma_cube()
    log_p_grid = XS.line_log10_pressure_grid()
    log_temperature_grids = XS.line_log10_temperature_grids()

    layer_pressures_bar = layer_pressures / bar
    log_p_layers = jnp.log10(layer_pressures_bar)
    log_t_layers = jnp.log10(layer_temperatures)

    p_idx = jnp.searchsorted(log_p_grid, log_p_layers) - 1
    p_idx = jnp.clip(p_idx, 0, log_p_grid.shape[0] - 2)
    p_weight = (log_p_layers - log_p_grid[p_idx]) / (log_p_grid[p_idx + 1] - log_p_grid[p_idx])
    p_weight = jnp.clip(p_weight, 0.0, 1.0)

    def _interp_one_species(sigma_3d, log_temp_grid):
        t_idx = jnp.searchsorted(log_temp_grid, log_t_layers) - 1
        t_idx = jnp.clip(t_idx, 0, log_temp_grid.shape[0] - 2)
        t_weight = (log_t_layers - log_temp_grid[t_idx]) / (log_temp_grid[t_idx + 1] - log_temp_grid[t_idx])
        t_weight = jnp.clip(t_weight, 0.0, 1.0)
        s_t0_p0 = sigma_3d[t_idx, p_idx, :]
        s_t0_p1 = sigma_3d[t_idx, p_idx + 1, :]
        s_t1_p0 = sigma_3d[t_idx + 1, p_idx, :]
        s_t1_p1 = sigma_3d[t_idx + 1, p_idx + 1, :]
        s_t0 = (1.0 - p_weight)[:, None] * s_t0_p0 + p_weight[:, None] * s_t0_p1
        s_t1 = (1.0 - p_weight)[:, None] * s_t1_p0 + p_weight[:, None] * s_t1_p1
        s_interp = (1.0 - t_weight)[:, None] * s_t0 + t_weight[:, None] * s_t1
        return 10.0 ** s_interp.astype(jnp.float64)

    # Interpolate all species
    sigma_interp_all = jax.vmap(_interp_one_species)(sigma_cube, log_temperature_grids)
    # Shape: (n_species, nlay, nwl)

    result = {}
    for i, name in enumerate(species_names):
        vmr = jnp.broadcast_to(layer_vmr[name], (layer_count,))
        kappa = (sigma_interp_all[i] * vmr[:, None]) / (layer_mu[:, None] * amu)
        result[f"line_{name}"] = kappa

    return result


def compute_ray_opacity_per_species(state, params):
    """
    Compute Rayleigh opacity for each species separately.

    Returns
    -------
    dict : {f"ray_{species}": opacity_array (nlay, nwl)}
    """
    from exo_skryer import registry_ray as XR

    if not XR.has_ray_data():
        return {}

    number_density = state["nd_lay"]
    density = state["rho_lay"]
    layer_vmr = state["vmr_lay"]
    layer_count = number_density.shape[0]

    sigma_log = jnp.asarray(XR.ray_sigma_table(), dtype=jnp.float64)
    sigma_values = 10.0 ** sigma_log
    species_names = XR.ray_species_names()

    result = {}
    for i, name in enumerate(species_names):
        vmr = jnp.broadcast_to(layer_vmr[name], (layer_count,))
        # sigma_values[i] is (nwl,), vmr is (nlay,)
        sigma_weighted = sigma_values[i][None, :] * vmr[:, None]
        kappa = (number_density[:, None] * sigma_weighted) / density[:, None]
        result[f"ray_{name}"] = kappa

    return result


def compute_cia_opacity_per_pair(state, params):
    """
    Compute CIA opacity for each molecular pair separately.

    Returns
    -------
    dict : {f"cia_{pair}": opacity_array (nlay, nwl)}
    """
    from exo_skryer import build_opacities as XS

    if not XS.has_cia_data():
        return {}

    layer_count = state["nlay"]
    layer_temperatures = state["T_lay"]
    number_density = state["nd_lay"]
    density = state["rho_lay"]
    layer_vmr = state["vmr_lay"]

    species_names = XS.cia_species_names()
    sigma_cube = XS.cia_sigma_cube()
    log_temperature_grids = XS.cia_log10_temperature_grids()
    temperature_grids = XS.cia_temperature_grids()

    log_t_layers = jnp.log10(layer_temperatures)

    def _interp_one_species(sigma_2d, log_temp_grid, temp_grid):
        t_idx = jnp.searchsorted(log_temp_grid, log_t_layers) - 1
        t_idx = jnp.clip(t_idx, 0, log_temp_grid.shape[0] - 2)
        t_weight = (log_t_layers - log_temp_grid[t_idx]) / (log_temp_grid[t_idx + 1] - log_temp_grid[t_idx])
        t_weight = jnp.clip(t_weight, 0.0, 1.0)
        s_t0 = sigma_2d[t_idx, :]
        s_t1 = sigma_2d[t_idx + 1, :]
        s_interp = (1.0 - t_weight)[:, None] * s_t0 + t_weight[:, None] * s_t1
        min_temp = temp_grid[0]
        below_min = layer_temperatures < min_temp
        tiny = jnp.array(-199.0, dtype=s_interp.dtype)
        s_interp = jnp.where(below_min[:, None], tiny, s_interp)
        return s_interp

    # Interpolate all CIA species
    sigma_log = jax.vmap(_interp_one_species)(sigma_cube, log_temperature_grids, temperature_grids)
    sigma_values = 10.0 ** sigma_log.astype(jnp.float64)
    # Shape: (n_species, nlay, nwl)

    result = {}
    for i, name in enumerate(species_names):
        name_clean = name.strip()
        if name_clean == "H-":
            continue  # H- handled separately
        parts = name_clean.split("-")
        if len(parts) != 2:
            continue
        species_a, species_b = parts[0], parts[1]
        vmr_a = jnp.broadcast_to(layer_vmr[species_a], (layer_count,))
        vmr_b = jnp.broadcast_to(layer_vmr[species_b], (layer_count,))
        pair_weight = vmr_a * vmr_b
        normalization = pair_weight * (number_density**2 / density)
        kappa = normalization[:, None] * sigma_values[i]
        result[f"cia_{name_clean}"] = kappa

    return result


# ============================================================================
# State-building (adapted from build_model.py)
# ============================================================================


def build_atmospheric_state(cfg, params, wl):
    """
    Build atmospheric state dictionary for opacity calculations.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object.
    params : dict
        Parameter dictionary (median values).
    wl : jnp.ndarray
        High-resolution wavelength grid.

    Returns
    -------
    state : dict
        Atmospheric state dictionary.
    """
    from exo_skryer.data_constants import kb, amu, R_jup, R_sun, bar, G, M_jup
    from exo_skryer.vert_alt import hypsometric, hypsometric_variable_g, hypsometric_variable_g_pref
    from exo_skryer.vert_Tp import isothermal, Milne, Guillot, Barstow, MandS, picket_fence, Milne_modified
    from exo_skryer.vert_chem import constant_vmr, CE_fastchem_jax, CE_rate_jax, quench_approx
    from exo_skryer.vert_mu import constant_mu, compute_mu
    from exo_skryer.vert_cloud import no_cloud, exponential_decay_profile, slab_profile, const_profile
    from exo_skryer.build_chem import prepare_chemistry_kernel

    phys = cfg.physics
    nlay = int(getattr(phys, "nlay", 99))
    nlev = nlay + 1

    # Select T-P kernel
    vert_tp_raw = getattr(phys, "vert_Tp", None)
    if vert_tp_raw in (None, "None"):
        vert_tp_raw = getattr(phys, "vert_struct", None)
    vert_tp_name = str(vert_tp_raw).lower()
    Tp_kernels = {
        "isothermal": isothermal, "constant": isothermal,
        "barstow": Barstow, "milne": Milne, "guillot": Guillot,
        "picket_fence": picket_fence, "mands": MandS,
        "milne_2": Milne_modified, "milne_modified": Milne_modified,
    }
    Tp_kernel = Tp_kernels.get(vert_tp_name, isothermal)

    # Select altitude kernel
    vert_alt_name = str(getattr(phys, "vert_alt", "hypsometric")).lower()
    alt_kernels = {
        "constant": hypsometric, "constant_g": hypsometric, "fixed": hypsometric,
        "hypsometric": hypsometric,
        "variable": hypsometric_variable_g, "variable_g": hypsometric_variable_g,
        "hypsometric_variable_g": hypsometric_variable_g,
        "p_ref": hypsometric_variable_g_pref, "hypsometric_variable_g_pref": hypsometric_variable_g_pref,
    }
    altitude_kernel = alt_kernels.get(vert_alt_name, hypsometric)

    # Select mu kernel
    vert_mu_name = str(getattr(phys, "vert_mu", "auto")).lower()
    if vert_mu_name == "auto":
        def mu_kernel(params, vmr_lay, nlay):
            if "mu" in params:
                return constant_mu(params, nlay)
            return compute_mu(vmr_lay)
    elif vert_mu_name in ("constant", "fixed"):
        def mu_kernel(params, vmr_lay, nlay):
            return constant_mu(params, nlay)
    else:
        def mu_kernel(params, vmr_lay, nlay):
            return compute_mu(vmr_lay)

    # Select cloud profile kernel
    vert_cloud_name = str(getattr(phys, "vert_cloud", "none")).lower()
    cloud_kernels = {
        "none": no_cloud, "off": no_cloud, "no_cloud": no_cloud,
        "exponential": exponential_decay_profile, "exp_decay": exponential_decay_profile,
        "exponential_decay": exponential_decay_profile,
        "slab": slab_profile, "slab_profile": slab_profile,
        "const": const_profile, "constant": const_profile, "const_profile": const_profile,
    }
    vert_cloud_kernel = cloud_kernels.get(vert_cloud_name, no_cloud)

    # Prepare chemistry kernel
    line_opac_scheme = str(getattr(phys, "opac_line", "lbl")).lower()
    ray_opac_scheme = str(getattr(phys, "opac_ray", "lbl")).lower()
    cia_opac_scheme = str(getattr(phys, "opac_cia", "lbl")).lower()
    special_opac_scheme = str(getattr(phys, "opac_special", "lbl")).lower()

    vert_chem_name = str(getattr(phys, "vert_chem", "constant_vmr")).lower()
    if vert_chem_name in ("constant", "constant_vmr"):
        chemistry_kernel_base = constant_vmr
    elif vert_chem_name in ("ce", "chemical_equilibrium", "ce_fastchem_jax", "fastchem_jax"):
        chemistry_kernel_base = CE_fastchem_jax
    elif vert_chem_name in ("rate_ce", "rate_jax", "ce_rate_jax"):
        chemistry_kernel_base = CE_rate_jax
    elif vert_chem_name in ("quench", "quench_approx"):
        chemistry_kernel_base = quench_approx
    else:
        chemistry_kernel_base = constant_vmr

    chemistry_kernel, trace_species = prepare_chemistry_kernel(
        cfg,
        chemistry_kernel_base,
        {
            'line_opac': line_opac_scheme,
            'ray_opac': ray_opac_scheme,
            'cia_opac': cia_opac_scheme,
            'special_opac': special_opac_scheme,
        }
    )

    # Convert params to JAX-compatible
    full_params = {k: jnp.asarray(v) for k, v in params.items()}

    nwl = wl.shape[0]

    # Planet/star radii
    R0 = jnp.asarray(full_params["R_p"]) * R_jup
    R_s = jnp.asarray(full_params["R_s"]) * R_sun

    # Calculate log_10_g from mass and radius if M_p is provided
    if "M_p" in full_params:
        M_p = jnp.asarray(full_params["M_p"]) * M_jup
        R_p = jnp.asarray(full_params["R_p"]) * R_jup
        g = G * M_p / (R_p ** 2)
        full_params["log_10_g"] = jnp.log10(g)

    # Atmospheric pressure grid
    p_bot = jnp.asarray(full_params["p_bot"]) * bar
    p_top = jnp.asarray(full_params["p_top"]) * bar
    p_lev = jnp.logspace(jnp.log10(p_bot), jnp.log10(p_top), nlev)
    p_lay = (p_lev[1:] - p_lev[:-1]) / jnp.log(p_lev[1:] / p_lev[:-1])

    # Temperature structure
    T_lev, T_lay = Tp_kernel(p_lev, full_params)

    # Chemistry (VMRs)
    vmr_lay = chemistry_kernel(p_lay, T_lay, full_params, nlay)

    # Mean molecular weight
    mu_lay = mu_kernel(full_params, vmr_lay, nlay)

    # Altitude grid
    z_lev, z_lay, dz = altitude_kernel(p_lev, T_lay, mu_lay, full_params)

    # Density
    rho_lay = (mu_lay * amu * p_lay) / (kb * T_lay)
    nd_lay = p_lay / (kb * T_lay)

    # Cloud profile
    q_c_lay = vert_cloud_kernel(p_lay, T_lay, mu_lay, rho_lay, nd_lay, full_params)

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
    }

    return state


# ============================================================================
# Main workflow
# ============================================================================


def compute_component_spectra(
    config_path: str,
    csv_path: str | None = None,
) -> Dict[str, np.ndarray]:
    """
    Compute per-species/component transmission spectra.

    Parameters
    ----------
    config_path : str
        Path to retrieval config YAML.
    csv_path : str, optional
        Path to nested_samples.csv (alternative to posterior.nc).

    Returns
    -------
    spectra : dict
        Dictionary mapping component names to transit depth arrays.
    """
    cfg_path = Path(config_path).resolve()
    exp_dir = cfg_path.parent

    # Add repo root to sys.path
    repo_root = (exp_dir / "../..").resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from exo_skryer.build_opacities import build_opacities, master_wavelength_cut
    from exo_skryer.registry_bandpass import load_bandpass_registry
    from exo_skryer.RT_trans_1D_lbl import _build_transit_geometry, _transit_depth_from_opacity
    from exo_skryer.opacity_cloud import compute_cloud_opacity
    from exo_skryer.opacity_special import compute_hminus_opacity
    from exo_skryer import build_opacities as XS

    # Load config
    cfg = _read_cfg(cfg_path)
    _bump_opacity_paths_one_level(cfg, exp_dir)

    # Load observed data
    lam, dlam, y_obs, dy_obs, response_mode = _load_observed(exp_dir, cfg)
    obs = {
        "wl": np.asarray(lam, dtype=float),
        "dwl": np.asarray(dlam, dtype=float),
        "y": np.asarray(y_obs, dtype=float) if y_obs is not None else None,
        "dy": np.asarray(dy_obs, dtype=float) if dy_obs is not None else None,
        "response_mode": np.asarray(response_mode, dtype=object) if response_mode is not None else None,
    }

    # Build opacities (populates registries)
    print("[component_spectra] Building opacities...")
    build_opacities(cfg, obs, exp_dir)

    # Get high-res wavelength grid
    hi_wl = np.asarray(master_wavelength_cut(), dtype=float)
    hi_wl_jax = jnp.asarray(hi_wl)

    # Load bandpass (needed for consistency even if not used)
    load_bandpass_registry(obs, hi_wl, hi_wl)

    # Load posterior and get median parameters
    params_cfg = getattr(cfg, "params", [])
    if not params_cfg:
        raise ValueError("cfg.params is empty.")

    if csv_path is not None:
        csv_file = Path(csv_path).resolve()
        print(f"[component_spectra] Loading samples from CSV: {csv_file}")
        param_draws, N_total = _build_param_draws_from_csv(csv_file, params_cfg)
    else:
        import arviz as az
        posterior_path = exp_dir / "posterior.nc"
        if not posterior_path.exists():
            raise FileNotFoundError(f"Missing {posterior_path}")
        print(f"[component_spectra] Loading samples from NetCDF: {posterior_path}")
        idata = az.from_netcdf(posterior_path)
        posterior_ds = idata.posterior
        param_draws, N_total = _build_param_draws_from_idata(posterior_ds, params_cfg)

    # Get median parameters
    theta_median = {name: float(np.median(arr)) for name, arr in param_draws.items()}
    print(f"[component_spectra] Using median of {N_total} posterior samples")

    # Build atmospheric state
    print("[component_spectra] Building atmospheric state...")
    state = build_atmospheric_state(cfg, theta_median, hi_wl_jax)

    # Compute per-species opacities
    print("[component_spectra] Computing per-species opacities...")
    all_opacities = {}

    # Line opacities
    line_opacs = compute_line_opacity_per_species(state, theta_median)
    all_opacities.update(line_opacs)
    if line_opacs:
        print(f"  - Line species: {list(line_opacs.keys())}")

    # Rayleigh opacities
    ray_opacs = compute_ray_opacity_per_species(state, theta_median)
    all_opacities.update(ray_opacs)
    if ray_opacs:
        print(f"  - Rayleigh species: {list(ray_opacs.keys())}")

    # CIA opacities
    cia_opacs = compute_cia_opacity_per_pair(state, theta_median)
    all_opacities.update(cia_opacs)
    if cia_opacs:
        print(f"  - CIA pairs: {list(cia_opacs.keys())}")

    # Cloud opacity
    phys = cfg.physics
    cloud_scheme = str(getattr(phys, "opac_cloud", "none")).lower()
    if cloud_scheme not in ("none", "null", "off"):
        try:
            from exo_skryer.opacity_cloud import (
                grey_cloud, deck_and_powerlaw, F18_cloud, direct_nk, given_nk
            )
            cloud_schemes = {
                "grey": grey_cloud,
                "grey_cloud": grey_cloud,
                "deck_and_powerlaw": deck_and_powerlaw,
                "f18": F18_cloud,
                "f18_cloud": F18_cloud,
                "direct_nk": direct_nk,
                "given_nk": given_nk,
            }
            cloud_fn = cloud_schemes.get(cloud_scheme)
            if cloud_fn is not None:
                full_params = {k: jnp.asarray(v) for k, v in theta_median.items()}
                k_cld, _, _ = cloud_fn(state, full_params)
                all_opacities["cloud"] = k_cld
                print(f"  - Cloud: {cloud_scheme}")
        except Exception as e:
            print(f"  - Cloud ({cloud_scheme}) failed: {e}")

    # H- special opacity
    special_scheme = str(getattr(phys, "opac_special", "none")).lower()
    if special_scheme not in ("none", "null", "off"):
        try:
            full_params = {k: jnp.asarray(v) for k, v in theta_median.items()}
            hminus_k = compute_hminus_opacity(state, full_params)
            if jnp.any(hminus_k > 0):
                all_opacities["special_H-"] = hminus_k
                print("  - Special: H-")
        except Exception:
            pass

    # Compute transit depth for each component
    print("[component_spectra] Computing transit depth for each component...")
    geometry = _build_transit_geometry(state)

    spectra = {"wavelength_um": hi_wl}
    for name, kappa in all_opacities.items():
        D = _transit_depth_from_opacity(state, kappa, geometry)
        spectra[name] = np.asarray(D)
        print(f"  - {name}: computed")

    # Compute total opacity and transit depth
    if all_opacities:
        total_opacity = sum(all_opacities.values())
        D_total = _transit_depth_from_opacity(state, total_opacity, geometry)
        spectra["total"] = np.asarray(D_total)
        print("  - total: computed")

    return spectra


def save_spectra(spectra: Dict[str, np.ndarray], output_path: Path):
    """Save component spectra to NPZ file."""
    np.savez_compressed(output_path, **spectra)
    print(f"\n[component_spectra] Saved to {output_path}")
    print(f"[component_spectra] Contents:")
    for key in sorted(spectra.keys()):
        if key != "wavelength_um":
            print(f"  - {key}: {spectra[key].shape}")


def main():
    ap = argparse.ArgumentParser(
        description="Calculate per-species/component transmission spectra"
    )
    ap.add_argument(
        "--config", required=True,
        help="Path to YAML config used for the retrieval"
    )
    ap.add_argument(
        "--csv", type=str, default=None,
        help="Path to nested_samples.csv (alternative to posterior.nc)"
    )
    ap.add_argument(
        "--output", type=str, default="component_spectra.npz",
        help="Output filename (default: component_spectra.npz)"
    )
    args = ap.parse_args()

    spectra = compute_component_spectra(args.config, csv_path=args.csv)

    output_path = Path(args.config).parent / args.output
    save_spectra(spectra, output_path)


if __name__ == "__main__":
    main()
