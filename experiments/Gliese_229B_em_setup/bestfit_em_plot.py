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
from scipy.integrate import simpson

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


def _compute_effective_temperature(flux: np.ndarray, lam_um: np.ndarray, R_p: float, D_pc: float = None) -> float:
    """Compute effective temperature from integrated flux using Stefan-Boltzmann law.

    Parameters
    ----------
    flux : np.ndarray
        Spectral flux F_λ in erg/s/cm²/cm (per cm of wavelength) at Earth
    lam_um : np.ndarray
        Wavelength grid in microns
    R_p : float
        Planet/brown dwarf radius in Jupiter radii
    D_pc : float, optional
        Distance in parsecs. If provided, flux is scaled to surface flux.

    Returns
    -------
    T_eff : float
        Effective temperature in Kelvin
    """
    const = _ensure_constants()
    R_jup = const["R_jup"]

    # Stefan-Boltzmann constant in cgs: erg/cm²/s/K⁴
    sigma_sb = 5.670374419e-5

    # Convert wavelength from microns to cm
    # 1 µm = 10^-4 cm
    lam_cm = np.asarray(lam_um, dtype=float) * 1.0e-4

    # Scale flux to surface if distance is provided
    if D_pc is not None:
        # Convert distance from pc to cm: 1 pc = 3.0857e18 cm
        D_cm = D_pc * 3.0857e18
        # Convert radius from Jupiter radii to cm
        R_cm = R_p * R_jup
        # Scale factor: (D/R)²
        scale_factor = (D_cm / R_cm) ** 2
        flux_surface = flux * scale_factor
    else:
        flux_surface = flux

    # Integrate flux over wavelength using Simpson's rule
    # F_λ is in erg/s/cm²/cm, integrating over cm gives erg/s/cm²
    F_total = simpson(flux_surface, x=lam_cm)

    # Stefan-Boltzmann: F = σ T^4
    # T_eff = (F / σ)^(1/4)
    T_eff = (F_total / sigma_sb) ** 0.25

    return T_eff


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
    repo_root = (exp_dir / "../..").resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from exo_skryer.build_model import build_forward_model
    from exo_skryer.build_opacities import build_opacities, master_wavelength_cut
    from exo_skryer.registry_bandpass import load_bandpass_registry
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

    # Compute median parameters for single model evaluation
    theta_median = {name: float(np.median(arr)) for name, arr in draws.items()}

    lam = lam_obs
    dlam = dlam_obs
    hires = lam_cut
    obs_flux = obs_flux_err = Tb_obs = Tb_obs_lo = Tb_obs_hi = None

    # Check if this is a brown dwarf (no stellar flux, direct planet emission)
    phys_cfg = getattr(cfg, "physics", SimpleNamespace())
    emission_mode = str(getattr(phys_cfg, "emission_mode", "planet")).lower()
    is_brown_dwarf = emission_mode == "brown_dwarf" or stellar_flux_np is None

    # Compute median model
    result = fm(theta_median)
    h_med = np.asarray(result["hires"], dtype=float)
    q_med = np.asarray(result["binned"], dtype=float)

    # Compute planet flux and brightness temperature for median model
    pf_med = Tb_med = None
    if "R_p" in theta_median:
        R_p_med = float(theta_median["R_p"])
        if stellar_flux_np is not None and "R_s" in theta_median:
            # Planet with stellar companion - convert ratio to flux
            R_s_med = float(theta_median["R_s"])
            pf_med = _recover_planet_flux(h_med, stellar_flux_np, R_p_med, R_s_med)
        elif is_brown_dwarf:
            # Brown dwarf - hires is already planet flux
            pf_med = h_med

    # Convert to percent (only for planets with stellar companion)
    if not is_brown_dwarf:
        q_med *= 100.0
        h_med *= 100.0

    # Compute brightness temperature from median planet flux
    if pf_med is not None:
        Tb_med = _flux_to_brightness_temperature(pf_med, hires)

        # Process observed data
        if stellar_flux_np is not None and "R_s" in theta_median:
            # Planet with stellar companion
            R_s_med = float(theta_median["R_s"])
            interp_stellar = np.interp(lam, hires, stellar_flux_np)
            obs_flux = _recover_planet_flux(y_obs, interp_stellar, R_p_med, R_s_med)
        else:
            # Brown dwarf - observed data is already in flux units
            obs_flux = y_obs

        if obs_flux is not None:
            Tb_obs = _flux_to_brightness_temperature(obs_flux, lam)

    save_payload = {
        "lam": lam,
        "dlam": dlam,
        "depth_p50": q_med,
        "lam_hires": hires,
        "depth_hi_p50": h_med,
    }

    if pf_med is not None:
        save_payload.update(
            planet_flux_p50=pf_med,
            Tb_p50=Tb_med,
        )

    np.savez_compressed(exp_dir / f"{outname}_quantiles.npz", **save_payload)

    # Match bestfit_plot.py style
    palette = sns.color_palette("colorblind")

    # Skip Fp/Fstar plot for brown dwarfs
    if not is_brown_dwarf:
        # Flux ratio plot
        fig_ratio, ax_ratio = plt.subplots(figsize=(8, 4.5))
        ax_ratio.plot(lam, q_med, lw=1, label="Median model", color=palette[4], alpha=0.7, rasterized=True)
        ax_ratio.plot(lam, y_obs * 100.0, lw=2, label="Observed", color="black", zorder=3)
        ax_ratio.set_xscale("log")
        ax_ratio.set_yscale("log")
        ax_ratio.set_xlabel("Wavelength [µm]", fontsize=14)
        ax_ratio.set_ylabel(r"$F_{\rm p}/F_{\star}$ [%]", fontsize=14)
        tick_locs = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
        ax_ratio.set_xticks(tick_locs)
        ax_ratio.set_xticklabels([f"{t:g}" for t in tick_locs], fontsize=12)
        ax_ratio.tick_params(axis="y", labelsize=12)
        ax_ratio.set_xlim(0.3, 6.0)
        ax_ratio.grid(False)
        ax_ratio.legend()
        fig_ratio.tight_layout()
        fig_ratio.savefig(exp_dir / f"{outname}.png", dpi=300)
        fig_ratio.savefig(exp_dir / f"{outname}.pdf")

    if pf_med is not None:
        # Planet flux figure (median only, no 1-sigma bands)
        fig_flux, ax_flux = plt.subplots(figsize=(8, 4.5))
        ax_flux.plot(hires, pf_med, lw=1, label="Median model", color=palette[4], alpha=0.7, rasterized=True)
        if obs_flux is not None:
            ax_flux.plot(lam, obs_flux, lw=2, label="Observed", color="black", zorder=3)
        ax_flux.set_xscale("log")
        ax_flux.set_yscale("log")
 
        ax_flux.set_xlabel(r"Wavelength [$\mu$m]", fontsize=14)
        ax_flux.set_ylabel(r"Flux [erg s$^{-1}$ cm$^{2}$ cm$^{-1}$]", fontsize=14)
        tick_locs = np.array([0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
        ax_flux.set_xticks(tick_locs)
        ax_flux.set_xticklabels([f"{t:g}" for t in tick_locs], fontsize=12)
        ax_flux.tick_params(axis="y", labelsize=12)
        ax_flux.set_xlim(0.6,5.0)
        # Adjust y-range here by uncommenting and modifying:
        ax_flux.set_ylim(10**-12, 10**-7)  # Example: adjust these values as needed
        ax_flux.grid(False)
        ax_flux.legend()
        fig_flux.tight_layout()
        fig_flux.savefig(exp_dir / f"{outname}_planet_flux.png", dpi=300)
        fig_flux.savefig(exp_dir / f"{outname}_planet_flux.pdf")

        # Brightness temperature figure (median only, no 1-sigma bands)
        fig_tb, ax_tb = plt.subplots(figsize=(8, 4.5))
        ax_tb.plot(hires, Tb_med, lw=1, label="Median model", color=palette[4], alpha=0.7, rasterized=True)
        if Tb_obs is not None:
            ax_tb.plot(lam, Tb_obs, lw=2, label="Observed", color="black", zorder=3)
        ax_tb.set_xscale("log")
        ax_tb.set_xlabel("Wavelength [µm]", fontsize=14)
        ax_tb.set_ylabel("Brightness temperature [K]", fontsize=14)
        tick_locs = np.array([
            0.6, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
        ax_tb.set_xticks(tick_locs)
        ax_tb.set_xticklabels([f"{t:g}" for t in tick_locs], fontsize=12)
        ax_tb.tick_params(axis="y", labelsize=12)
        ax_tb.set_xlim(0.6, 5.0)
        # Adjust y-range here by uncommenting and modifying:
        # ax_tb.set_ylim(400, 1400)  # Example: adjust these values as needed
        ax_tb.grid(False)
        ax_tb.legend()
        fig_tb.tight_layout()
        fig_tb.savefig(exp_dir / f"{outname}_brightness_temperature.png", dpi=300)
        fig_tb.savefig(exp_dir / f"{outname}_brightness_temperature.pdf")

        # Brightness temperature zoom plot removed for Gliese experiment (not needed for 0.3-6 µm range)

        # Compute effective temperature from median high-res spectrum
        print("\n" + "="*60)
        print("EFFECTIVE TEMPERATURE CALCULATION")
        print("="*60)

        # Get distance if available (for brown dwarfs)
        D_pc_median = float(theta_median["D"]) if "D" in theta_median else None

        T_eff = _compute_effective_temperature(pf_med, hires, R_p_med, D_pc_median)

        print(f"Effective temperature from integrated median flux:")
        print(f"  T_eff = {T_eff:.1f} K")
        if D_pc_median is not None:
            print(f"  Distance: {D_pc_median:.2f} pc")
            print(f"  Radius: {R_p_med:.3f} R_Jup")
        print("="*60 + "\n")

        save_payload.update(T_eff=T_eff)
        np.savez_compressed(exp_dir / f"{outname}_quantiles.npz", **save_payload)

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
