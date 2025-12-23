#!/usr/bin/env python3
"""
test_forward_model.py
=====================
Test the forward model with custom parameters from retrieval_config.yaml.
This runs a single forward model evaluation to verify the setup is working.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Set up JAX
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)


def _to_ns(x):
    """Convert nested dict to nested SimpleNamespace."""
    if isinstance(x, list):
        return [_to_ns(v) for v in x]
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    return x


def _read_cfg(path: Path):
    """Read YAML config and convert to namespace."""
    with path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return _to_ns(y)


def _resolve_path_relative(path_str: str, exp_dir: Path) -> Path:
    """Resolve a path relative to experiment directory."""
    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return path_obj
    # Try experiment dir and its parents
    base_dirs = [exp_dir]
    base_dirs.extend(exp_dir.parents)
    for base in base_dirs:
        candidate = (base / path_obj).resolve()
        if candidate.exists():
            return candidate
    return (exp_dir / path_obj).resolve()


def _is_fixed_param(p) -> bool:
    """Check if parameter is fixed (delta dist)."""
    return str(getattr(p, "dist", "")).lower() == "delta"


def _get_param_value(p):
    """Get parameter value from config."""
    # For fixed params, use 'value'
    if _is_fixed_param(p):
        val = getattr(p, "value", None)
        if val is not None:
            return float(val)
    # For free params, use 'init' as test value
    init = getattr(p, "init", None)
    if init is not None:
        return float(init)
    return None


def _load_observed(exp_dir: Path, cfg):
    """Load observed data from file."""
    # Try CSV first (if retrieval was already run)
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

    # Otherwise load from obs file specified in config
    data_cfg = getattr(cfg, "data", None)
    obs_path_str = getattr(data_cfg, "obs", None) if data_cfg is not None else None

    if obs_path_str is None:
        raise ValueError("No observed data path found in config")

    obs_path = _resolve_path_relative(obs_path_str, exp_dir)

    # Load the file (format: wavelength, half_bandwidth, flux, error, response_mode)
    arr = np.loadtxt(obs_path, dtype=str)
    if arr.ndim == 1:
        arr = arr[None, :]

    lam = arr[:, 0].astype(float)
    dlam = arr[:, 1].astype(float)
    y = arr[:, 2].astype(float)
    dy = arr[:, 3].astype(float)
    resp = arr[:, 4] if arr.shape[1] >= 5 else np.full_like(lam, "boxcar", dtype=object)

    return lam, dlam, y, dy, resp


def main():
    """Run forward model test with custom parameters."""

    # Get experiment directory
    exp_dir = Path(__file__).parent.resolve()
    config_path = exp_dir / "retrieval_config.yaml"

    print("=" * 60)
    print("FORWARD MODEL TEST")
    print("=" * 60)
    print(f"Experiment dir: {exp_dir}")
    print(f"Config file: {config_path}")

    # Read config
    cfg = _read_cfg(config_path)

    # Load observed data
    print("\n[1/5] Loading observed data...")
    lam_obs, dlam_obs, y_obs, dy_obs, resp_obs = _load_observed(exp_dir, cfg)
    print(f"  Found {len(lam_obs)} data points")
    print(f"  Wavelength range: {lam_obs.min():.3f} - {lam_obs.max():.3f} µm")

    # Import exo_skryer modules
    from exo_skryer.build_opacities import build_opacities, master_wavelength_cut
    from exo_skryer.registry_bandpass import load_bandpass_registry
    from exo_skryer.read_stellar import read_stellar_spectrum
    from exo_skryer.build_model import build_forward_model

    # Prepare observational dict
    obs = {
        "wl": np.asarray(lam_obs, dtype=float),
        "dwl": np.asarray(dlam_obs, dtype=float),
        "y": np.asarray(y_obs, dtype=float),
        "dy": np.asarray(dy_obs, dtype=float),
        "response_mode": np.asarray(resp_obs, dtype=object),
    }

    # Build opacities
    print("\n[2/5] Building opacities...")
    build_opacities(cfg, obs, exp_dir)
    lam_cut = np.asarray(master_wavelength_cut(), dtype=float)
    print(f"  Master grid: {len(lam_cut)} wavelength points")

    # Load bandpass registry
    print("\n[3/5] Loading bandpass registry...")
    load_bandpass_registry(obs, lam_cut, lam_cut)

    # Read stellar spectrum (if needed)
    opac_cfg = getattr(cfg, "opac", SimpleNamespace())
    has_ck = bool(getattr(opac_cfg, "ck", False))
    stellar_flux = read_stellar_spectrum(cfg, lam_cut, has_ck, base_dir=exp_dir)

    # Build forward model
    print("\n[4/5] Building forward model...")
    fm = build_forward_model(cfg, obs, stellar_flux=stellar_flux, return_highres=True)
    print("  Forward model built successfully")

    # Extract parameters from config
    params_cfg = getattr(cfg, "params", [])
    theta = {}

    print("\n[5/5] Extracting parameters from config...")
    print("\nParameter values:")
    print("-" * 60)
    for p in params_cfg:
        name = str(getattr(p, "name", ""))
        val = _get_param_value(p)
        if val is not None:
            theta[name] = val
            dist = getattr(p, "dist", "")
            print(f"  {name:20s} = {val:12.6g}  ({dist})")
    print("-" * 60)

    # Run forward model
    print("\nRunning forward model...")
    result = fm(theta)

    # Extract results
    model_binned = np.asarray(result["binned"], dtype=float)
    model_hires = np.asarray(result["hires"], dtype=float)

    print(f"  Success! Generated spectrum with {len(model_binned)} binned points")
    print(f"           and {len(model_hires)} hi-res points")

    # Plot results
    print("\nGenerating plots...")

    palette = sns.color_palette("colorblind")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top panel: Model vs Data
    ax1.plot(lam_cut, model_hires, lw=0.5, alpha=0.5, label="Model (hi-res)", color=palette[4])
    ax1.plot(lam_obs, model_binned, 'o-', lw=2, ms=4, label="Model (binned)", color=palette[1])
    ax1.errorbar(
        lam_obs, y_obs,
        yerr=dy_obs,
        xerr=dlam_obs,
        fmt='o',
        ms=3,
        lw=1,
        alpha=0.7,
        label="Observed",
        color=palette[0],
        capsize=2,
    )
    ax1.set_xlabel("Wavelength [µm]", fontsize=14)
    ax1.set_ylabel("Flux", fontsize=14)
    ax1.set_title("Forward Model Test: Model vs Observed Data", fontsize=16)
    ax1.set_yscale('log')
    ax1.tick_params(labelsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Residuals (model - data) / error
    residuals = (model_binned - y_obs) / dy_obs
    ax2.errorbar(
        lam_obs, residuals,
        xerr=dlam_obs,
        fmt='o',
        ms=4,
        lw=1,
        color=palette[0],
        capsize=2,
    )
    ax2.axhline(0, color='k', linestyle='--', lw=1, alpha=0.5)
    ax2.axhline(1, color='gray', linestyle=':', lw=0.5, alpha=0.5)
    ax2.axhline(-1, color='gray', linestyle=':', lw=0.5, alpha=0.5)
    ax2.set_xlabel("Wavelength [µm]", fontsize=14)
    ax2.set_ylabel("Residuals (σ)", fontsize=14)
    ax2.set_title("Residuals: (Model - Data) / Error", fontsize=14)
    ax2.tick_params(labelsize=12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    # Save figure
    outname = "forward_model_test"
    fig.savefig(exp_dir / f"{outname}.png", dpi=300)
    fig.savefig(exp_dir / f"{outname}.pdf")
    print(f"\nPlots saved:")
    print(f"  {exp_dir / outname}.png")
    print(f"  {exp_dir / outname}.pdf")

    # Save model output
    np.savez_compressed(
        exp_dir / f"{outname}_output.npz",
        lam_obs=lam_obs,
        dlam_obs=dlam_obs,
        y_obs=y_obs,
        dy_obs=dy_obs,
        model_binned=model_binned,
        lam_hires=lam_cut,
        model_hires=model_hires,
        theta=theta,
    )
    print(f"  {exp_dir / outname}_output.npz")

    # Show plot
    plt.show()

    print("\n" + "=" * 60)
    print("FORWARD MODEL TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
