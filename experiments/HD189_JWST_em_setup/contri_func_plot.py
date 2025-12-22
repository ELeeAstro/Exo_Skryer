#!/usr/bin/env python3
"""
contri_func_plot.py -- Emission contribution function plotter for HD 189.
Computes and plots the normalized emission contribution function for the median model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import seaborn as sns
import yaml
import arviz as az
from types import SimpleNamespace

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)


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


def plot_contribution_function(config_path, outname="contribution_function", show_plot=True, wavelengths=None):
    """
    Plot emission contribution function for median model.

    Parameters
    ----------
    config_path : str or Path
        Path to retrieval_config.yaml
    outname : str
        Output filename prefix
    show_plot : bool
        Whether to display plots
    wavelengths : list of float, optional
        Specific wavelengths (in microns) to plot 1D profiles for.
        If None, will select wavelengths automatically.
    """
    cfg_path = Path(config_path).resolve()
    exp_dir = cfg_path.parent
    repo_root = (exp_dir / "../..").resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from exo_skryer.build_model import build_forward_model
    from exo_skryer.build_opacities import build_opacities, master_wavelength_cut
    from exo_skryer.registry_bandpass import load_bandpass_registry
    from exo_skryer.read_stellar import read_stellar_spectrum
    from exo_skryer.data_constants import bar

    # Read config and ensure contri_func is enabled
    cfg = _read_cfg(cfg_path)

    # Temporarily enable contribution function
    original_contri_func = getattr(cfg.physics, "contri_func", False)
    cfg.physics.contri_func = True

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
    fm = build_forward_model(cfg, obs, stellar_flux=stellar_flux, return_highres=True)

    posterior_path = exp_dir / "posterior.nc"
    if not posterior_path.exists():
        raise FileNotFoundError("posterior.nc not found. Run the retrieval first.")

    idata = az.from_netcdf(posterior_path)
    params_cfg = getattr(cfg, "params", [])
    draws, N_total = _build_param_draws_from_idata(idata.posterior, params_cfg)

    # Compute median parameters
    theta_median = {name: float(np.median(arr)) for name, arr in draws.items()}

    # Run forward model with median parameters
    result = fm(theta_median)

    if "contrib_func" not in result:
        raise RuntimeError("Contribution function not returned. Check that contri_func=True in config.")

    contrib_func = np.asarray(result["contrib_func"])  # shape: (nlay, nwl)
    p_lay = np.asarray(result["p_lay"]) / bar  # Convert to bar
    hires_wl = lam_cut

    nlay, nwl = contrib_func.shape
    print(f"\nContribution function computed successfully!")
    print(f"  Shape: {contrib_func.shape} (nlay={nlay}, nwl={nwl})")
    print(f"  Wavelength range: {hires_wl.min():.2f} - {hires_wl.max():.2f} µm")
    print(f"  Pressure range: {p_lay.min():.2e} - {p_lay.max():.2e} bar")
    print(f"  Contribution sum per wavelength: min={contrib_func.sum(axis=0).min():.3f}, max={contrib_func.sum(axis=0).max():.3f}")
    print(f"  (should be ~1.0 for normalized contribution function)")

    # Save contribution function data
    np.savez_compressed(
        exp_dir / f"{outname}_data.npz",
        contrib_func=contrib_func,
        p_lay=p_lay,
        wavelength=hires_wl,
    )

    # Select wavelengths for 1D plots if not provided
    if wavelengths is None:
        # Select 6 wavelengths spanning the range
        wl_indices = np.linspace(0, nwl - 1, 6, dtype=int)
        wavelengths = hires_wl[wl_indices]
    else:
        # Find nearest wavelength indices
        wl_indices = [np.argmin(np.abs(hires_wl - wl)) for wl in wavelengths]
        wavelengths = hires_wl[wl_indices]

    palette = sns.color_palette("husl", n_colors=len(wavelengths))

    # Plot 1: 1D contribution function profiles at selected wavelengths
    fig1, ax1 = plt.subplots(figsize=(6, 8))
    for i, (wl_idx, wl) in enumerate(zip(wl_indices, wavelengths)):
        cf = contrib_func[:, wl_idx]
        ax1.plot(cf, p_lay, label=f"{wl:.2f} µm", lw=2, color=palette[i])

    ax1.set_yscale("log")
    ax1.invert_yaxis()
    ax1.set_xlabel("Normalized Contribution Function", fontsize=14)
    ax1.set_ylabel("Pressure [bar]", fontsize=14)
    ax1.set_title("Emission Contribution Function", fontsize=16)
    ax1.legend(fontsize=10, loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)
    fig1.tight_layout()
    fig1.savefig(exp_dir / f"{outname}_1d.png", dpi=300)

    # Plot 2: 2D contour plot (wavelength vs pressure)
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Use pcolormesh for the 2D plot
    wl_mesh, p_mesh = np.meshgrid(hires_wl, p_lay)

    # Plot with fixed logarithmic color scale and smooth filled contours.
    vmin = 1e-4
    vmax = 1.0
    cmap = sns.color_palette("rocket", as_cmap=True)
    cmap.set_bad("white")
    cmap.set_under("white")
    contrib_plot = np.ma.masked_less(contrib_func, vmin)
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 128)
    pcm = ax2.contourf(
        hires_wl,
        p_lay,
        contrib_plot,
        levels=levels,
        cmap=cmap,
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        extend="min",
    )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.invert_yaxis()
    ax2.set_xlabel("Wavelength [µm]", fontsize=14)
    ax2.set_ylabel("Pressure [bar]", fontsize=14)
    ax2.set_title("2D Emission Contribution Function (Log Scale)", fontsize=16)

    # Wavelength ticks: denser in the optical while keeping a log axis.
    # (The log axis is useful to show wide spectral range, but default major ticks
    # are sparse at short wavelengths.)
    tick_locs = np.array(
        [
            0.2, 0.3, 0.4, 0.5, 0.6, 0.8,
            1.0, 1.2, 1.5, 2.0, 3.0, 5.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]
    )
    tick_locs = tick_locs[(tick_locs >= hires_wl.min()) & (tick_locs <= hires_wl.max())]
    ax2.xaxis.set_major_locator(mticker.FixedLocator(tick_locs))
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}"))
    ax2.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax2.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax2.tick_params(axis="x", labelsize=12)
    ax2.tick_params(axis="y", labelsize=12)

    # Add colorbar
    cbar = fig2.colorbar(
        pcm,
        ax=ax2,
        label="Normalized Contribution (log scale)",
        extend="min",
    )
    cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    cbar.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    cbar.ax.tick_params(labelsize=12)

    fig2.tight_layout()
    fig2.savefig(exp_dir / f"{outname}_2d.png", dpi=300)

    # Plot 3: 2D contour plot zoomed to 7-12 µm
    zoom_min, zoom_max = 7.0, 12.0
    wl_mask = (hires_wl >= zoom_min) & (hires_wl <= zoom_max)
    if np.any(wl_mask):
        fig3, ax3 = plt.subplots(figsize=(8, 6))

        wl_zoom = hires_wl[wl_mask]
        cf_zoom = contrib_func[:, wl_mask]
        wl_mesh_zoom, p_mesh_zoom = np.meshgrid(wl_zoom, p_lay)

        # Use fixed log scale for zoom plot (smooth filled contours)
        vmin_zoom = 1e-4
        vmax_zoom = 1.0
        cmap_zoom = sns.color_palette("rocket", as_cmap=True)
        cmap_zoom.set_bad("white")
        cmap_zoom.set_under("white")
        cf_zoom_plot = np.ma.masked_less(cf_zoom, vmin_zoom)
        levels_zoom = np.logspace(np.log10(vmin_zoom), np.log10(vmax_zoom), 128)
        pcm_zoom = ax3.contourf(
            wl_zoom,
            p_lay,
            cf_zoom_plot,
            levels=levels_zoom,
            cmap=cmap_zoom,
            norm=mcolors.LogNorm(vmin=vmin_zoom, vmax=vmax_zoom),
            extend="min",
        )

        ax3.set_yscale("log")
        ax3.invert_yaxis()
        ax3.set_xlabel("Wavelength [µm]", fontsize=14)
        ax3.set_ylabel("Pressure [bar]", fontsize=14)
        ax3.set_title("2D Emission Contribution Function (7-12 µm, Log Scale)", fontsize=16)
        ax3.set_xlim(zoom_min, zoom_max)
        ax3.tick_params(labelsize=12)
        ax3.xaxis.set_major_locator(mticker.FixedLocator([7, 8, 9, 10, 11, 12]))
        ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}"))

        cbar_zoom = fig3.colorbar(
            pcm_zoom,
            ax=ax3,
            label="Normalized Contribution (log scale)",
            extend="min",
        )
        cbar_zoom.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
        cbar_zoom.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
        cbar_zoom.ax.tick_params(labelsize=12)

        fig3.tight_layout()
        fig3.savefig(exp_dir / f"{outname}_2d_zoom.png", dpi=300)

    # Restore original config setting
    cfg.physics.contri_func = original_contri_func

    if show_plot:
        plt.show()

    print(f"\nSaved contribution function plots to:")
    print(f"  - {outname}_1d.png")
    print(f"  - {outname}_2d.png")
    if np.any(wl_mask):
        print(f"  - {outname}_2d_zoom.png")
    print(f"  - {outname}_data.npz")


def main():
    ap = argparse.ArgumentParser(description="Emission contribution function plotter")
    ap.add_argument("--config", required=True, help="Path to retrieval_config.yaml")
    ap.add_argument("--outname", default="contribution_function", help="Output filename prefix")
    ap.add_argument("--no-show", action="store_true", help="Don't display plots")
    ap.add_argument(
        "--wavelengths",
        type=float,
        nargs="+",
        help="Specific wavelengths (µm) for 1D plots (default: auto-select 6)",
    )
    args = ap.parse_args()

    plot_contribution_function(
        args.config,
        outname=args.outname,
        show_plot=not args.no_show,
        wavelengths=args.wavelengths,
    )


if __name__ == "__main__":
    main()
