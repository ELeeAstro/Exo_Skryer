#!/usr/bin/env python3
"""
bestfit_em_plot.py -- emission-spectrum analog of bestfit_plot.py.
Builds the forward-model for a 1D emission retrieval and plots
median/credible bands together with observed HD 189 JWST data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import arviz as az
from types import SimpleNamespace

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

_CONST_CACHE = {}


def _ensure_constants():
    global _CONST_CACHE
    if not _CONST_CACHE:
        from exo_skryer.data_constants import R_jup, R_sun, h, c_light, kb

        _CONST_CACHE = {
            "R_jup": R_jup,
            "R_sun": R_sun,
            "h": h,
            "c_light": c_light,
            "kb": kb,
        }
    return _CONST_CACHE


def _to_ns(x):
    if isinstance(x, list):
        return [_to_ns(v) for v in x]
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    return x


def _read_cfg(path: Path):
    with path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return _to_ns(y)


def _resolve_path_relative(path_str: str, exp_dir: Path) -> Path:
    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return path_obj
    base_dirs = [exp_dir]
    base_dirs.extend(exp_dir.parents)
    for base in base_dirs:
        candidate = (base / path_obj).resolve()
        if candidate.exists():
            return candidate
    return (exp_dir / path_obj).resolve()


def _is_fixed_param(p) -> bool:
    return str(getattr(p, "dist", "")).lower() == "delta" or bool(getattr(p, "fixed", False))


def _fixed_value_param(p):
    val = getattr(p, "value", None)
    if val is not None:
        return float(val)
    init = getattr(p, "init", None)
    if init is not None:
        return float(init)
    return None


def _load_observed(exp_dir: Path, cfg):
    csv_path = exp_dir / "observed_data.csv"
    if csv_path.exists():
        arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=str)
        if arr.ndim == 1:
            arr = arr[None, :]
        lam = arr[:, 0].astype(float)
        dlam = arr[:, 1].astype(float)
        y = arr[:, 2].astype(float)
        dy = arr[:, 3].astype(float)
        resp = arr[:, 4] if arr.shape[1] >= 5 else np.full_like(lam, "boxcar", dtype=object)
        return lam, dlam, y, dy, resp
    data_cfg = getattr(cfg, "data", None)
    obs_path = getattr(data_cfg, "obs", None) if data_cfg is not None else None
    if obs_path is None:
        obs_cfg = getattr(cfg, "obs", None)
        obs_path = getattr(obs_cfg, "path", None) if obs_cfg is not None else None
    if obs_path is None:
        raise FileNotFoundError("Need observed data via observed_data.csv or cfg.data.obs/cfg.obs.path")
    data_path = _resolve_path_relative(obs_path, exp_dir)
    arr = np.loadtxt(data_path, dtype=str)
    if arr.ndim == 1:
        arr = arr[None, :]
    lam = arr[:, 0].astype(float)
    dlam = arr[:, 1].astype(float)
    y = arr[:, 2].astype(float)
    dy = arr[:, 3].astype(float)
    resp = arr[:, 4] if arr.shape[1] >= 5 else np.full_like(lam, "boxcar", dtype=object)
    return lam, dlam, y, dy, resp


def _flatten_param(a: np.ndarray) -> np.ndarray:
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
    raise ValueError(f"Unsupported param shape {arr.shape} for flattening")


def _build_param_draws_from_idata(posterior_ds, params_cfg):
    out: Dict[str, np.ndarray] = {}
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
                raise ValueError(f"Fixed/delta param '{name}' needs value/init in YAML.")
            out[name] = np.full((N_total,), float(val), dtype=float)
        else:
            raise KeyError(f"Free parameter '{name}' missing from posterior.")
    return out, N_total


def _flux_to_brightness_temperature(flux: np.ndarray, lam_um: np.ndarray) -> np.ndarray:
    """Convert spectral flux (hemispheric, per wavelength) to brightness temperature."""
    const = _ensure_constants()
    h = const["h"]
    c_light = const["c_light"]
    kb = const["kb"]
    wl_cm = np.asarray(lam_um, dtype=float) * 1.0e-4
    wl_cm = np.maximum(wl_cm, 1.0e-12)
    B_lambda = np.maximum(flux / np.pi, 1.0e-300)
    prefactor = 2.0 * h * c_light**2 / (wl_cm**5)
    ratio = 1.0 + np.maximum(prefactor / np.maximum(B_lambda, 1.0e-300), 0.0)
    Tb = (h * c_light) / (kb * wl_cm * np.log(ratio))
    return Tb


def _recover_planet_flux(
    flux_ratio: np.ndarray,
    stellar_flux: np.ndarray,
    R_p: float,
    R_s: float,
) -> np.ndarray:
    """Reverse the scaling in RT_em_1D to obtain the top-of-atmosphere flux."""
    const = _ensure_constants()
    R_jup = const["R_jup"]
    R_sun = const["R_sun"]
    R0 = R_p * R_jup
    Rstar = R_s * R_sun
    scale = np.maximum(stellar_flux, 1.0e-30) * (Rstar**2) / np.maximum(R0**2, 1.0e-30)
    return flux_ratio * scale


def plot_emission_band(config_path, outname="model_emission", max_samples=2000, seed=123, show_plot=True):
    cfg_path = Path(config_path).resolve()
    exp_dir = cfg_path.parent
    src_root = (exp_dir / "../../exo_skryer").resolve()
    if src_root.is_dir() and str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from exo_skryer.build_model import build_forward_model
    from exo_skryer.build_opacities import build_opacities, master_wavelength_cut
    from exo_skryer.egistry_bandpass import load_bandpass_registry
    from exo_skryer.read_stellar import read_stellar_spectrum

    cfg = _read_cfg(cfg_path)
    lam_obs, dlam_obs, y_obs, dy_obs, resp_obs = _load_observed(exp_dir, cfg)
    lam_obs = np.asarray(lam_obs, dtype=float)
    dlam_obs = np.asarray(dlam_obs, dtype=float)
    obs = {
        "wl": lam_obs,
        "dwl": dlam_obs,
        "y": np.asarray(y_obs, dtype=float),
        "dy": np.asarray(dy_obs, dtype=float),
        "response_mode": np.asarray(resp_obs, dtype=object),
    }

    build_opacities(cfg, obs, exp_dir)
    lam_cut = np.asarray(master_wavelength_cut(), dtype=float)
    load_bandpass_registry(obs, lam_cut, lam_cut)
    opac_cfg = getattr(cfg, "opac", SimpleNamespace())
    has_ck = bool(getattr(opac_cfg, "ck", []))
    stellar_flux = read_stellar_spectrum(cfg, lam_cut, has_ck, base_dir=exp_dir)
    stellar_flux_np = np.asarray(stellar_flux, dtype=float) if stellar_flux is not None else None
    fm = build_forward_model(cfg, obs, stellar_flux=stellar_flux, return_highres=True)

    posterior_path = exp_dir / "posterior.nc"
    if not posterior_path.exists():
        raise FileNotFoundError("posterior.nc not found. Run the retrieval first.")
    idata = az.from_netcdf(posterior_path)
    params_cfg = getattr(cfg, "params", [])
    draws, N_total = _build_param_draws_from_idata(idata.posterior, params_cfg)
    max_samples = min(int(max_samples), N_total)
    rng = np.random.default_rng(seed)
    max_samples = max(1, max_samples)
    M = min(max_samples, N_total)
    idx = np.sort(rng.choice(N_total, size=M, replace=False))
    lam = lam_obs
    dlam = dlam_obs
    hires = lam_cut
    model_samples = np.empty((M, lam.size))
    hires_samples = np.empty((M, hires.size))
    planet_flux_samples = None
    obs_flux = obs_flux_err = Tb_obs = Tb_obs_lo = Tb_obs_hi = None
    if (
        stellar_flux_np is not None
        and "R_p" in draws
        and "R_s" in draws
    ):
        planet_flux_samples = np.empty((M, hires.size))
        R_p_draws = np.asarray(draws["R_p"], dtype=float)
        R_s_draws = np.asarray(draws["R_s"], dtype=float)
    else:
        R_p_draws = R_s_draws = None

    for i, sel in enumerate(idx):
        theta = {name: arr[sel] for name, arr in draws.items()}
        result = fm(theta)
        hires_samples[i] = np.asarray(result["hires"], dtype=float)
        model_samples[i] = np.asarray(result["binned"], dtype=float)
        if planet_flux_samples is not None:
            R_p = float(theta["R_p"])
            R_s = float(theta["R_s"])
            planet_flux_samples[i] = _recover_planet_flux(hires_samples[i], stellar_flux_np, R_p, R_s)

    q_lo, q_med, q_hi = np.percentile(model_samples, [2.5, 50, 97.5], axis=0)
    h_lo, h_med, h_hi = np.percentile(hires_samples, [2.5, 50, 97.5], axis=0)

    save_payload = {
        "lam": lam,
        "dlam": dlam,
        "depth_p02_5": q_lo,
        "depth_p50": q_med,
        "depth_p97_5": q_hi,
        "lam_hires": hires,
        "depth_hi_p02_5": h_lo,
        "depth_hi_p50": h_med,
        "depth_hi_p97_5": h_hi,
        "draw_idx": idx,
    }

    pf_lo = pf_med = pf_hi = Tb_med = Tb_lo = Tb_hi = None
    if planet_flux_samples is not None:
        pf_lo, pf_med, pf_hi = np.percentile(planet_flux_samples, [2.5, 50, 97.5], axis=0)
        Tb_samples = _flux_to_brightness_temperature(planet_flux_samples, hires)
        Tb_lo, Tb_med, Tb_hi = np.percentile(Tb_samples, [2.5, 50, 97.5], axis=0)
        save_payload.update(
            planet_flux_p02_5=pf_lo,
            planet_flux_p50=pf_med,
            planet_flux_p97_5=pf_hi,
            Tb_p02_5=Tb_lo,
            Tb_p50=Tb_med,
            Tb_p97_5=Tb_hi,
        )
        R_p_med = float(np.median(R_p_draws[idx]))
        R_s_med = float(np.median(R_s_draws[idx]))
        interp_stellar = np.interp(lam, hires, stellar_flux_np)
        obs_flux = _recover_planet_flux(y_obs, interp_stellar, R_p_med, R_s_med)
        if dy_obs is not None:
            unit_scale = _recover_planet_flux(
                np.ones_like(lam),
                interp_stellar,
                R_p_med,
                R_s_med,
            )
            obs_flux_err = np.abs(dy_obs) * unit_scale
        else:
            obs_flux_err = None
        Tb_obs = _flux_to_brightness_temperature(obs_flux, lam)
        if obs_flux_err is not None:
            flux_hi = np.maximum(obs_flux + obs_flux_err, 1.0e-300)
            flux_lo = np.maximum(obs_flux - obs_flux_err, 1.0e-300)
            Tb_obs_hi = _flux_to_brightness_temperature(flux_hi, lam)
            Tb_obs_lo = _flux_to_brightness_temperature(flux_lo, lam)
        else:
            Tb_obs_hi = Tb_obs_lo = None

    np.savez_compressed(exp_dir / f"{outname}_quantiles.npz", **save_payload)

    sns.set(style="whitegrid")
    palette = sns.color_palette("colorblind")

    fig_ratio, ax_ratio = plt.subplots(figsize=(7, 5))
    ax_ratio.plot(hires, h_med, color=palette[0], lw=1, alpha=0.7, label="Median (hi-res)")
    ax_ratio.fill_between(lam, q_lo, q_hi, alpha=0.3, color=palette[1], label="95% credible band")
    ax_ratio.plot(lam, q_med, color=palette[2], lw=2, label="Median model")
    ax_ratio.errorbar(
        lam,
        y_obs,
        yerr=dy_obs,
        xerr=dlam,
        fmt="o",
        color=palette[3],
        capsize=3,
        label="Data",
    )
    ax_ratio.set_xscale("log")
    ax_ratio.set_ylabel("F_p / F_s")
    ax_ratio.set_xlabel("Wavelength (micron)")
    ax_ratio.legend()
    fig_ratio.tight_layout()
    fig_ratio.savefig(exp_dir / f"{outname}.png", dpi=200)
    fig_ratio.savefig(exp_dir / f"{outname}.pdf")

    if planet_flux_samples is not None and pf_lo is not None:
        # Planet flux figure
        fig_flux, ax_flux = plt.subplots(figsize=(7, 5))
        ax_flux.fill_between(hires, pf_lo, pf_hi, alpha=0.2, color=palette[4], label="Planet flux 95%")
        ax_flux.plot(hires, pf_med, color=palette[4], lw=2, label="Planet flux median")
        if obs_flux is not None:
            ax_flux.errorbar(
                lam,
                obs_flux,
                yerr=obs_flux_err,
                xerr=dlam,
                fmt="s",
                color=palette[5],
                capsize=3,
                label="Observed flux",
            )
        ax_flux.set_xscale("log")
        ax_flux.set_yscale("log")
        ax_flux.set_xlabel("Wavelength (micron)")
        ax_flux.set_ylabel("Planet flux [cgs]")
        ax_flux.legend()
        fig_flux.tight_layout()
        fig_flux.savefig(exp_dir / f"{outname}_planet_flux.png", dpi=200)
        fig_flux.savefig(exp_dir / f"{outname}_planet_flux.pdf")

        # Brightness temperature figure
        fig_tb, ax_tb = plt.subplots(figsize=(7, 5))
        ax_tb.fill_between(hires, Tb_lo, Tb_hi, alpha=0.2, color=palette[6], label="T_b 95%")
        ax_tb.plot(hires, Tb_med, color=palette[6], lw=2, label="Brightness T median")
        if Tb_obs is not None and Tb_obs_lo is not None and Tb_obs_hi is not None:
            Tb_err = np.vstack(
                (
                    Tb_obs - Tb_obs_lo,
                    Tb_obs_hi - Tb_obs,
                )
            )
            ax_tb.errorbar(
                lam,
                Tb_obs,
                yerr=Tb_err,
                xerr=dlam,
                fmt="o",
                color=palette[7],
                capsize=3,
                label="Observed T_b",
            )
        ax_tb.set_xscale("log")
        ax_tb.set_xlabel("Wavelength (micron)")
        ax_tb.set_ylabel("Brightness temperature [K]")
        ax_tb.legend()
        fig_tb.tight_layout()
        fig_tb.savefig(exp_dir / f"{outname}_brightness_temperature.png", dpi=200)
        fig_tb.savefig(exp_dir / f"{outname}_brightness_temperature.pdf")

    if show_plot:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Emission best-fit plotter")
    ap.add_argument("--config", required=True)
    ap.add_argument("--outname", default="model_emission")
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    plot_emission_band(
        args.config,
        outname=args.outname,
        max_samples=args.max_samples,
        seed=args.seed,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
