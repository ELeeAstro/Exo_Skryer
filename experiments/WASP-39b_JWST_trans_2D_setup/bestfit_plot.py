#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict


def _configure_runtime() -> None:
    exp_dir = Path(__file__).resolve().parent
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    mpl_cache_dir = exp_dir / ".mplconfig"
    mpl_cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))


_configure_runtime()

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _build_param_draws_from_posterior(posterior_ds, params_cfg) -> tuple[Dict[str, np.ndarray], int]:
    if "chain" not in posterior_ds.dims or "draw" not in posterior_ds.dims:
        raise ValueError("posterior.nc must have dims ('chain', 'draw').")

    n_total = int(posterior_ds.sizes["chain"]) * int(posterior_ds.sizes["draw"])
    out: Dict[str, np.ndarray] = {}

    for p in params_cfg:
        name = getattr(p, "name", None)
        if not name:
            continue
        dist = str(getattr(p, "dist", "")).lower()
        if name in posterior_ds.data_vars:
            out[name] = np.asarray(posterior_ds[name].values, dtype=float).reshape(-1)
        elif dist == "delta":
            value = float(getattr(p, "value", getattr(p, "init")))
            out[name] = np.full(n_total, value, dtype=float)
        else:
            raise KeyError(f"Free parameter '{name}' missing from posterior.")

    return out, n_total


def _quantiles(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.quantile(samples, 0.16, axis=0),
        np.quantile(samples, 0.50, axis=0),
        np.quantile(samples, 0.84, axis=0),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="retrieval_config.yaml")
    ap.add_argument("--posterior", default="posterior.nc")
    ap.add_argument("--output", default="bestfit_limb_spectra.png")
    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    exp_dir = config_path.parent
    repo_root = (exp_dir / "../..").resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import jax
    jax.config.update("jax_enable_x64", True)

    from exo_skryer.read_yaml import read_yaml
    from exo_skryer.read_obs import read_obs_data, resolve_obs_path
    from exo_skryer.build_opacities import build_opacities, master_wavelength_cut
    from exo_skryer.registry_bandpass import load_bandpass_registry
    from exo_skryer.build_chem import load_nasa9_if_needed
    from exo_skryer.build_model import build_forward_model
    from exo_skryer.read_stellar import read_stellar_spectrum

    cfg = read_yaml(config_path)
    posterior_path = exp_dir / args.posterior
    posterior_ds = az.from_netcdf(posterior_path).posterior
    param_draws, n_total = _build_param_draws_from_posterior(posterior_ds, getattr(cfg, "params", []))

    obs_spec = resolve_obs_path(cfg)
    obs = read_obs_data(obs_spec, base_dir=exp_dir)
    if "wl_east" not in obs or "wl_west" not in obs:
        raise ValueError("bestfit_plot.py expects separate limb observations with east/west wavelength grids.")

    build_opacities(cfg, obs, exp_dir)
    hi_wl = np.asarray(master_wavelength_cut(), dtype=float)
    load_bandpass_registry(obs, hi_wl, hi_wl)
    load_nasa9_if_needed(cfg, exp_dir)
    stellar_flux = read_stellar_spectrum(cfg, hi_wl, bool(cfg.opac.ck), base_dir=exp_dir)
    predict_fn = build_forward_model(cfg, obs, stellar_flux=stellar_flux, return_highres=True)

    rng = np.random.default_rng(args.seed)
    max_samples = None if args.max_samples <= 0 else args.max_samples
    if max_samples is not None and max_samples < n_total:
        idx = np.sort(rng.choice(n_total, size=max_samples, replace=False))
    else:
        idx = np.arange(n_total)

    m = idx.size
    east_binned = np.empty((m, obs["wl_east"].shape[0]), dtype=float)
    west_binned = np.empty((m, obs["wl_west"].shape[0]), dtype=float)
    east_hires = np.empty((m, hi_wl.shape[0]), dtype=float)
    west_hires = np.empty((m, hi_wl.shape[0]), dtype=float)

    east_slice = obs.get("east_slice", slice(None))
    west_slice = obs.get("west_slice", slice(None))

    for k, ii in enumerate(idx):
        params = {name: float(values[ii]) for name, values in param_draws.items()}
        result = predict_fn(params)

        east_hires[k, :] = np.asarray(result.get("hires_east_scaled", 0.5 * result["hires_east"]), dtype=float)
        west_hires[k, :] = np.asarray(result.get("hires_west_scaled", 0.5 * result["hires_west"]), dtype=float)
        east_binned[k, :] = np.asarray(result.get("binned_east_scaled", 0.5 * result["binned_east"]), dtype=float)[east_slice]
        west_binned[k, :] = np.asarray(result.get("binned_west_scaled", 0.5 * result["binned_west"]), dtype=float)[west_slice]

    if not np.all(np.isfinite(east_binned)) or not np.all(np.isfinite(west_binned)):
        raise RuntimeError("Non-finite model values encountered while evaluating posterior draws.")

    east_q16, east_q50, east_q84 = _quantiles(east_binned * 100.0)
    west_q16, west_q50, west_q84 = _quantiles(west_binned * 100.0)
    east_hq16, east_hq50, east_hq84 = _quantiles(east_hires * 100.0)
    west_hq16, west_hq50, west_hq84 = _quantiles(west_hires * 100.0)

    np.savez_compressed(
        exp_dir / "bestfit_limb_spectra_quantiles.npz",
        draw_idx=idx,
        lam_east=obs["wl_east"],
        lam_west=obs["wl_west"],
        dlam_east=obs["dwl_east"],
        dlam_west=obs["dwl_west"],
        east_p16=east_q16,
        east_p50=east_q50,
        east_p84=east_q84,
        west_p16=west_q16,
        west_p50=west_q50,
        west_p84=west_q84,
        lam_hires=hi_wl,
        east_hi_p16=east_hq16,
        east_hi_p50=east_hq50,
        east_hi_p84=east_hq84,
        west_hi_p16=west_hq16,
        west_hi_p50=west_hq50,
        west_hi_p84=west_hq84,
    )

    palette = sns.color_palette("colorblind")
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    panels = [
        (
            axes[0],
            "East limb",
            obs["wl_east"],
            np.asarray(obs["dwl_east"], dtype=float),
            np.asarray(obs["y_east"], dtype=float) * 100.0,
            np.asarray(obs["dy_east"], dtype=float) * 100.0,
            east_q16,
            east_q50,
            east_q84,
            east_hq50,
        ),
        (
            axes[1],
            "West limb",
            obs["wl_west"],
            np.asarray(obs["dwl_west"], dtype=float),
            np.asarray(obs["y_west"], dtype=float) * 100.0,
            np.asarray(obs["dy_west"], dtype=float) * 100.0,
            west_q16,
            west_q50,
            west_q84,
            west_hq50,
        ),
    ]

    for ax, title, wl, dwl, y, dy, q16, q50, q84, hq50 in panels:
        ax.plot(hi_wl, hq50, lw=1.0, alpha=0.7, label="Median (hi-res)", color=palette[4], rasterized=True)
        ax.fill_between(wl, q16, q84, alpha=0.3, color=palette[1], label=r"1$\sigma$")
        ax.plot(wl, q50, lw=2, label="Median", color=palette[1])
        ax.errorbar(
            wl,
            y,
            xerr=dwl,
            yerr=dy,
            fmt="o",
            ms=3,
            lw=1,
            alpha=0.9,
            label="Observed",
            color=palette[0],
            ecolor=palette[0],
            capsize=2,
        )
        ax.set_title(title)
        ax.set_ylabel("Transit Depth [%]", fontsize=14)
        ax.set_xscale("log")
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(False)
        ax.legend()

    tick_locs = np.array([0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])
    axes[1].set_xticks(tick_locs)
    axes[1].set_xticklabels([f"{t:g}" for t in tick_locs], fontsize=12)
    axes[1].set_xlim(0.5, 5.5)
    axes[1].set_xlabel("Wavelength [µm]", fontsize=14)
    fig.tight_layout()

    out_png = exp_dir / args.output
    out_pdf = exp_dir / (Path(args.output).stem + ".pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
