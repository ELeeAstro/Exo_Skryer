#!/usr/bin/env python3
"""
compare_ck_mix_methods.py
=========================

Compare RORR and TRANS correlated-k methods using the forward model.

This script:
1. Builds forward models with each method (RORR, TRANS)
2. Runs them with identical test parameters
3. Compares the resulting spectra
4. Reports timing and differences

Usage:
    python compare_ck_mix_methods.py

Output:
    - ck_mix_comparison.png/pdf - Spectrum comparison plot
    - ck_mix_comparison.npz - Numerical results
"""

from __future__ import annotations

import os
import sys
import time
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


# ============================================================================
# Configuration and Setup
# ============================================================================

def _to_ns(x):
    """Convert dict/list to SimpleNamespace recursively."""
    if isinstance(x, list):
        return [_to_ns(v) for v in x]
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    return x


def _read_cfg(path: Path):
    """Read YAML config file."""
    with path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return _to_ns(y)


def _resolve_path_relative(path_str: str, exp_dir: Path) -> Path:
    """Resolve relative paths against experiment directory."""
    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return path_obj
    for base in [exp_dir] + list(exp_dir.parents):
        candidate = (base / path_obj).resolve()
        if candidate.exists():
            return candidate
    return (exp_dir / path_obj).resolve()


def _bump_opacity_paths(cfg, exp_dir: Path):
    """Resolve opacity file paths."""
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
                    if p:
                        resolved = _resolve_path_relative(str(p), exp_dir)
                        setattr(spec, "path", str(resolved))


def _configure_runtime(cfg) -> None:
    """Set JAX/CUDA environment variables based on config."""
    runtime_cfg = getattr(cfg, "runtime", None)
    if runtime_cfg is None:
        return

    platform = str(getattr(runtime_cfg, "platform", "cpu")).lower()

    if platform == "cpu":
        # Leave CPU runtime environment to JAX defaults.
        return
    else:
        cuda_devices = str(getattr(runtime_cfg, "cuda_visible_devices", ""))
        if cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _load_observed(exp_dir: Path, cfg) -> Dict[str, np.ndarray]:
    """Load observed data from CSV or YAML path."""
    csv_path = exp_dir / "observed_data.csv"
    if csv_path.exists():
        arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=str)
        if arr.ndim == 1:
            arr = arr[None, :]
        lam = arr[:, 0].astype(float)
        dlam = arr[:, 1].astype(float) if arr.shape[1] >= 2 else np.zeros_like(lam)
        # Handle potential string columns (response_mode) by checking dtype
        try:
            y = arr[:, 2].astype(float) if arr.shape[1] >= 3 else None
        except ValueError:
            y = None
        try:
            dy = arr[:, 3].astype(float) if arr.shape[1] >= 4 else None
        except ValueError:
            dy = None
        response_mode = arr[:, 4] if arr.shape[1] >= 5 else None
        return {"wl": lam, "dwl": dlam, "y": y, "dy": dy, "response_mode": response_mode}

    # Try YAML obs path
    data_cfg = getattr(cfg, "data", None)
    obs_path = getattr(data_cfg, "obs", None) if data_cfg else None
    if obs_path is None:
        raise FileNotFoundError("No observed data found")

    data_path = _resolve_path_relative(obs_path, exp_dir)

    # Load as string first to handle mixed numeric/string columns
    arr = np.loadtxt(data_path, dtype=str)
    if arr.ndim == 1:
        arr = arr[None, :]

    lam = arr[:, 0].astype(float)
    response_mode = None
    if arr.shape[1] >= 5:
        # Format: lam, dlam, y, dy, response_mode
        dlam = arr[:, 1].astype(float)
        y = arr[:, 2].astype(float)
        dy = arr[:, 3].astype(float)
        response_mode = arr[:, 4]  # Keep as string array
    elif arr.shape[1] >= 4:
        dlam = arr[:, 1].astype(float)
        y = arr[:, 2].astype(float)
        dy = arr[:, 3].astype(float)
    elif arr.shape[1] == 3:
        dlam = np.zeros_like(lam)
        y = arr[:, 1].astype(float)
        dy = arr[:, 2].astype(float)
    else:
        dlam = np.zeros_like(lam)
        y = arr[:, 1].astype(float) if arr.shape[1] >= 2 else None
        dy = None

    return {"wl": lam, "dwl": dlam, "y": y, "dy": dy, "response_mode": response_mode}


def deep_copy_cfg(cfg):
    """Deep copy a SimpleNamespace config."""
    if isinstance(cfg, SimpleNamespace):
        return SimpleNamespace(**{k: deep_copy_cfg(v) for k, v in vars(cfg).items()})
    elif isinstance(cfg, list):
        return [deep_copy_cfg(item) for item in cfg]
    elif isinstance(cfg, dict):
        return {k: deep_copy_cfg(v) for k, v in cfg.items()}
    else:
        return cfg


# ============================================================================
# Main Comparison
# ============================================================================

def main():
    print("=" * 70)
    print("CK Method Comparison: RORR vs TRANS")
    print("=" * 70)

    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    repo_root = (script_dir / "../..").resolve()

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Load base configuration
    config_path = script_dir / "retrieval_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg_base = _read_cfg(config_path)
    _configure_runtime(cfg_base)

    # Configure JAX
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)

    # Import after setting up environment
    from exo_skryer.build_model import build_forward_model
    from exo_skryer.build_opacities import build_opacities, master_wavelength_cut
    from exo_skryer.registry_bandpass import load_bandpass_registry
    import exo_skryer.registry_ck as registry_ck

    # Load observed data
    obs = _load_observed(script_dir, cfg_base)
    print(f"\nObserved data: {len(obs['wl'])} wavelength points")
    print(f"  Range: {obs['wl'].min():.3f} - {obs['wl'].max():.3f} µm")

    # Test parameters (fixed values for comparison)
    test_params = {
        "R_s": 0.776484,
        "p_bot": 1000.0,
        "p_top": 1e-8,
        "log_10_g": 3.1,
        "R_p": 1.1,
        "T_strat": 1200.0,
        "log_10_f_H2O": -4.0,
        "log_10_k_cld_grey": -3.0,
        "log_10_k_cld_Ray": -3.0,
        "alpha_cld": 4.0,
        "wl_ref_cld": 0.3,
    }

    # Define methods to compare
    methods = ["RORR", "TRANS"]
    results = {}

    for method in methods:
        print(f"\n{'=' * 70}")
        print(f"Building forward model with ck_mix = {method}")
        print("=" * 70)

        # Deep copy config and modify for CK mode
        cfg = deep_copy_cfg(cfg_base)

        # Force contribution function off (TRANS path does not implement it)
        if hasattr(cfg, "physics") and cfg.physics is not None:
            cfg.physics.contri_func = False

        # Enable CK mode
        cfg.opac.ck = True
        cfg.opac.ck_mix = method

        # Use CK opacity files (H2O at R1000)
        cfg.opac.line = [
            SimpleNamespace(
                species="H2O",
                path="../../opac_data/ck/1H2-16O__POKAZATEL__R1000_0.3-50mu.ktable.petitRADTRANS.h5"
            ),
        ]

        # Resolve paths
        _bump_opacity_paths(cfg, script_dir)

        # Clear any cached opacity data from previous method
        registry_ck.reset_registry()

        try:
            # Build opacities
            print(f"  Loading CK opacities...")
            build_opacities(cfg, obs, script_dir)

            # Load bandpass
            hi_wl = np.asarray(master_wavelength_cut(), dtype=float)
            load_bandpass_registry(obs, hi_wl, hi_wl)

            print(f"  Hi-res wavelength grid: {len(hi_wl)} points")

            # Build forward model
            print(f"  Building forward model...")
            t_build_start = time.perf_counter()
            predict_fn = build_forward_model(cfg, obs, return_highres=True)
            t_build = time.perf_counter() - t_build_start
            print(f"  Build time: {t_build:.2f} s")

            # Warm-up run (JIT compilation)
            print(f"  JIT compiling (warm-up run)...")
            t_jit_start = time.perf_counter()
            _ = predict_fn(test_params)
            t_jit = time.perf_counter() - t_jit_start
            print(f"  JIT compilation time: {t_jit:.2f} s")

            # Timed evaluation runs
            n_runs = 5
            print(f"  Running {n_runs} evaluations...")
            times = []
            for i in range(n_runs):
                t_start = time.perf_counter()
                result = predict_fn(test_params)
                t_elapsed = time.perf_counter() - t_start
                times.append(t_elapsed)

            t_mean = np.mean(times)
            t_std = np.std(times)
            print(f"  Evaluation time: {t_mean*1000:.2f} ± {t_std*1000:.2f} ms")

            # Store results
            results[method] = {
                "hires": np.asarray(result["hires"]),
                "binned": np.asarray(result["binned"]),
                "hi_wl": hi_wl,
                "t_build": t_build,
                "t_jit": t_jit,
                "t_eval_mean": t_mean,
                "t_eval_std": t_std,
                "success": True,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[method] = {
                "success": False,
                "error": str(e),
            }

    # ========================================================================
    # Analysis and Plotting
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("Results Summary")
    print("=" * 70)

    successful_methods = [m for m in methods if results[m]["success"]]

    if len(successful_methods) == 0:
        print("No methods succeeded. Cannot produce comparison.")
        return

    # Print timing summary
    print("\nTiming Summary:")
    print("-" * 50)
    print(f"{'Method':<10} {'Build (s)':<12} {'JIT (s)':<12} {'Eval (ms)':<15}")
    print("-" * 50)
    for method in successful_methods:
        r = results[method]
        print(f"{method:<10} {r['t_build']:<12.2f} {r['t_jit']:<12.2f} "
              f"{r['t_eval_mean']*1000:.2f} ± {r['t_eval_std']*1000:.2f}")

    # Compare spectra
    if len(successful_methods) >= 2:
        print("\nSpectrum Differences (relative to RORR):")
        print("-" * 50)

        ref_method = "RORR" if "RORR" in successful_methods else successful_methods[0]
        ref_spectrum = results[ref_method]["binned"]

        for method in successful_methods:
            if method == ref_method:
                continue
            spectrum = results[method]["binned"]
            diff = spectrum - ref_spectrum
            rel_diff = diff / np.maximum(ref_spectrum, 1e-10) * 100

            print(f"{method} vs {ref_method}:")
            print(f"  Max abs diff:  {np.max(np.abs(diff)):.2e}")
            print(f"  Mean abs diff: {np.mean(np.abs(diff)):.2e}")
            print(f"  Max rel diff:  {np.max(np.abs(rel_diff)):.2f}%")
            print(f"  RMS rel diff:  {np.sqrt(np.mean(rel_diff**2)):.4f}%")

    # Create comparison plot
    print("\nCreating comparison plot...")

    palette = sns.color_palette("colorblind")
    colors = {"RORR": palette[0], "PRAS": palette[1], "TRANS": palette[2]}

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

    # Top panel: Spectra
    ax1 = axes[0]
    for i, method in enumerate(successful_methods):
        r = results[method]
        # Plot binned spectrum
        ax1.plot(
            obs["wl"], r["binned"] * 100,
            label=f"{method} (binned)",
            color=colors.get(method, f"C{i}"),
            lw=2,
            alpha=0.8
        )

    # Add observed data if available
    if obs["y"] is not None:
        ax1.errorbar(
            obs["wl"], obs["y"] * 100,
            xerr=obs["dwl"],
            yerr=obs["dy"] * 100 if obs["dy"] is not None else None,
            fmt="o", ms=4, color="black", alpha=0.6,
            label="Observed", capsize=2
        )

    ax1.set_ylabel("Transit Depth [%]", fontsize=12)
    ax1.set_xscale("log")
    ax1.legend(loc="best", fontsize=10)
    ax1.set_title("CK Mixing Method Comparison", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Differences
    ax2 = axes[1]
    if len(successful_methods) >= 2:
        ref_method = "RORR" if "RORR" in successful_methods else successful_methods[0]
        ref_spectrum = results[ref_method]["binned"]

        for i, method in enumerate(successful_methods):
            if method == ref_method:
                continue
            spectrum = results[method]["binned"]
            rel_diff = (spectrum - ref_spectrum) / np.maximum(ref_spectrum, 1e-10) * 100

            ax2.plot(
                obs["wl"], rel_diff,
                label=f"{method} - {ref_method}",
                color=colors.get(method, f"C{i}"),
                lw=1.5
            )

        ax2.axhline(0, color="gray", ls="--", lw=1)
        ax2.set_ylabel(f"Rel. Diff [%]", fontsize=12)
        ax2.legend(loc="best", fontsize=10)
        ax2.grid(True, alpha=0.3)

    ax2.set_xlabel("Wavelength [µm]", fontsize=12)
    ax2.set_xscale("log")

    plt.tight_layout()

    # Save
    output_stem = script_dir / "ck_mix_comparison"
    plt.savefig(f"{output_stem}.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{output_stem}.pdf", bbox_inches="tight")
    print(f"  Saved: {output_stem}.png")
    print(f"  Saved: {output_stem}.pdf")

    # Save numerical results
    save_data = {
        "wl_binned": obs["wl"],
        "dwl": obs["dwl"],
        "test_params": str(test_params),
    }
    for method in successful_methods:
        r = results[method]
        save_data[f"{method}_binned"] = r["binned"]
        save_data[f"{method}_hires"] = r["hires"]
        save_data[f"{method}_hi_wl"] = r["hi_wl"]
        save_data[f"{method}_t_eval"] = r["t_eval_mean"]

    np.savez_compressed(f"{output_stem}.npz", **save_data)
    print(f"  Saved: {output_stem}.npz")

    plt.show()
    print("\nDone!")


if __name__ == "__main__":
    main()
