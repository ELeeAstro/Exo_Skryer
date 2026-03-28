#!/usr/bin/env python3
"""
run_forward_model.py
====================
Generate a transmission spectrum from a forward model configuration.

All parameters in the YAML config must use ``dist: delta`` with explicit values.
The script evaluates the forward model once and saves the spectrum to a text file.

Usage
-----
    # High-res only (no obs file needed):
    python run_forward_model.py --config forward_config.yaml

    # With observational binning:
    python run_forward_model.py --config forward_config.yaml --obs ../../obs_data/WASP-107b_JWST.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def _is_missing_obs_path(value) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "none", "null", "~"}


def main() -> None:

    p = argparse.ArgumentParser(description="Run a forward model and save the spectrum")
    p.add_argument("--config", required=True, help="Path to forward model YAML config")
    p.add_argument(
        "--obs",
        default=None,
        help="Path to observational data file (enables binned output). "
             "Overrides data.obs in the config.",
    )
    p.add_argument(
        "--output-prefix",
        default="forward_spectrum",
        help="Output filename prefix (default: forward_spectrum)",
    )
    args = p.parse_args()

    t_start = time.perf_counter()

    # Resolve paths
    config_path = Path(args.config).resolve()
    exp_dir = config_path.parent

    # Add repo root to sys.path
    repo_root = (exp_dir / "../..").resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Load config
    from exo_skryer.read_yaml import read_yaml
    cfg = read_yaml(config_path)

    # Determine obs path: CLI flag overrides config
    obs_path_raw = args.obs
    if obs_path_raw is None:
        cfg_obs_east = getattr(cfg.data, "obs_east", None)
        cfg_obs_west = getattr(cfg.data, "obs_west", None)
        has_east = not _is_missing_obs_path(cfg_obs_east)
        has_west = not _is_missing_obs_path(cfg_obs_west)
        if has_east or has_west:
            if not has_east or not has_west:
                raise ValueError("Both data.obs_east and data.obs_west must be set for separate limb observations.")
            obs_path_raw = {"east": str(cfg_obs_east), "west": str(cfg_obs_west)}
        else:
            cfg_obs = getattr(cfg.data, "obs", None)
            if not _is_missing_obs_path(cfg_obs):
                obs_path_raw = str(cfg_obs)
    has_obs = obs_path_raw is not None

    # Configure JAX platform
    platform = str(getattr(cfg.runtime, "platform", "cpu")).lower()

    if platform == "gpu":
        cuda_devices = str(getattr(cfg.runtime, "cuda_visible_devices", ""))
        if cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
        print(f"[info] Platform: GPU (CUDA_VISIBLE_DEVICES={cuda_devices})")
    else:
        print("[info] Platform: CPU")

    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)

    import jax
    print(f"[info] JAX backend: {jax.default_backend()}")
    print(f"[info] JAX devices: {jax.local_device_count()} {jax.devices()}")

    # Load observational data (or create dummy for high-res-only mode)
    from exo_skryer.read_obs import read_obs_data

    if has_obs:
        obs = read_obs_data(obs_path_raw, base_dir=exp_dir)
        if obs.get("has_limb_observations", False):
            print(
                f"[info] Obs data loaded: east={len(obs['wl_east'])} bins, "
                f"west={len(obs['wl_west'])} bins"
            )
        else:
            print(f"[info] Obs data loaded: {len(obs['wl'])} bins")
    else:
        print("[info] No obs data — will produce high-res spectrum only")
        obs = None

    # Build opacities
    from exo_skryer.build_opacities import build_opacities, master_wavelength, master_wavelength_cut

    if obs is not None:
        build_opacities(cfg, obs, exp_dir)
    else:
        # For high-res only mode, create a dummy obs spanning the full wl range
        # so build_opacities / init_cut_master_wl can proceed (full_grid: True
        # means the cut grid equals the full grid regardless).
        dummy_obs = {
            "wl": np.array([1.0]),
            "dwl": np.array([0.5]),
            "y": np.array([0.0]),
            "dy": np.array([1.0]),
            "response_mode": np.array(["boxcar"]),
            "offset_group": np.array(["__no_offset__"]),
            "offset_group_idx": np.array([0]),
            "offset_group_names": np.array(["__no_offset__"]),
        }
        build_opacities(cfg, dummy_obs, exp_dir)

    full_grid = np.asarray(master_wavelength(), dtype=float)
    cut_grid = np.asarray(master_wavelength_cut(), dtype=float)
    print(f"[info] Master grid: N={full_grid.size}, "
          f"range=[{full_grid.min():.5f}, {full_grid.max():.5f}] um")
    print(f"[info] Cut grid:    N={cut_grid.size}, "
          f"range=[{cut_grid.min():.5f}, {cut_grid.max():.5f}] um")

    # Load bandpass registry
    from exo_skryer.registry_bandpass import load_bandpass_registry

    if obs is not None:
        load_bandpass_registry(obs, full_grid, cut_grid)
    else:
        # Dummy bandpass for the single dummy bin
        dummy_obs_bp = {
            "wl": np.array([cut_grid.mean()]),
            "dwl": np.array([(cut_grid.max() - cut_grid.min()) / 2.0]),
            "response_mode": np.array(["boxcar"]),
        }
        load_bandpass_registry(dummy_obs_bp, full_grid, cut_grid)

    # Load NASA9 tables if needed (for chemical equilibrium)
    from exo_skryer.build_chem import load_nasa9_if_needed
    load_nasa9_if_needed(cfg, exp_dir)

    # Build forward model
    from exo_skryer.build_model import build_forward_model
    from exo_skryer.read_stellar import read_stellar_spectrum

    stellar_flux = read_stellar_spectrum(cfg, cut_grid, bool(cfg.opac.ck), base_dir=exp_dir)

    # Use the real obs for binning, or the dummy for high-res-only
    fm_obs = obs if obs is not None else dummy_obs_bp
    fm_fnc = build_forward_model(cfg, fm_obs, stellar_flux=stellar_flux, return_highres=True)

    # Evaluate forward model (all params are delta → pass empty dict)
    print("[info] Evaluating forward model ...")
    t0 = time.perf_counter()
    result = fm_fnc({})
    t1 = time.perf_counter()
    print(f"[info] Forward model evaluation took {t1 - t0:.3f} s (includes JIT compile)")

    if "hires_east" in result and "hires_west" in result:
        hires_east = np.asarray(result["hires_east"], dtype=float)
        hires_west = np.asarray(result["hires_west"], dtype=float)
        hires_east_scaled = np.asarray(result.get("hires_east_scaled", 0.5 * hires_east), dtype=float)
        hires_west_scaled = np.asarray(result.get("hires_west_scaled", 0.5 * hires_west), dtype=float)

        hires_east_output = np.column_stack([cut_grid, hires_east])
        hires_east_path = exp_dir / f"{args.output_prefix}_east_highres.txt"
        np.savetxt(
            hires_east_path,
            hires_east_output,
            header="wavelength_um  transit_depth_east",
            fmt="%.10e",
        )
        print(f"[info] East high-res spectrum saved to: {hires_east_path}")

        hires_east_scaled_output = np.column_stack([cut_grid, hires_east_scaled])
        hires_east_scaled_path = exp_dir / f"{args.output_prefix}_east_highres_scaled.txt"
        np.savetxt(
            hires_east_scaled_path,
            hires_east_scaled_output,
            header="wavelength_um  transit_depth_east_scaled",
            fmt="%.10e",
        )
        print(f"[info] East scaled high-res spectrum saved to: {hires_east_scaled_path}")

        hires_west_output = np.column_stack([cut_grid, hires_west])
        hires_west_path = exp_dir / f"{args.output_prefix}_west_highres.txt"
        np.savetxt(
            hires_west_path,
            hires_west_output,
            header="wavelength_um  transit_depth_west",
            fmt="%.10e",
        )
        print(f"[info] West high-res spectrum saved to: {hires_west_path}")
        hires_west_scaled_output = np.column_stack([cut_grid, hires_west_scaled])
        hires_west_scaled_path = exp_dir / f"{args.output_prefix}_west_highres_scaled.txt"
        np.savetxt(
            hires_west_scaled_path,
            hires_west_scaled_output,
            header="wavelength_um  transit_depth_west_scaled",
            fmt="%.10e",
        )
        print(f"[info] West scaled high-res spectrum saved to: {hires_west_scaled_path}")
    else:
        D_hires = np.asarray(result["hires"], dtype=float)
        hires_output = np.column_stack([cut_grid, D_hires])
        hires_path = exp_dir / f"{args.output_prefix}_highres.txt"
        np.savetxt(
            hires_path, hires_output,
            header="wavelength_um  transit_depth",
            fmt="%.10e",
        )
        print(f"[info] High-res spectrum saved to: {hires_path}")

    # Save binned spectrum (only when real obs data was provided)
    if obs is not None:
        if "binned_east" in result and "binned_west" in result:
            if obs.get("has_limb_observations", False):
                east_slice = obs["east_slice"]
                west_slice = obs["west_slice"]
                east_wl = obs["wl_east"]
                east_dwl = obs["dwl_east"]
                west_wl = obs["wl_west"]
                west_dwl = obs["dwl_west"]
            else:
                east_slice = slice(None)
                west_slice = slice(None)
                east_wl = obs["wl"]
                east_dwl = obs["dwl"]
                west_wl = obs["wl"]
                west_dwl = obs["dwl"]

            D_binned_east = np.asarray(result["binned_east"], dtype=float)[east_slice]
            D_binned_east_scaled = np.asarray(result.get("binned_east_scaled", 0.5 * result["binned_east"]), dtype=float)[east_slice]
            east_output = np.column_stack([east_wl, east_dwl, D_binned_east])
            east_path = exp_dir / f"{args.output_prefix}_east_binned.txt"
            np.savetxt(
                east_path,
                east_output,
                header="wavelength_um  half_bin_width_um  transit_depth_east",
                fmt="%.10e",
            )
            print(f"[info] East binned spectrum saved to: {east_path}")
            east_scaled_output = np.column_stack([east_wl, east_dwl, D_binned_east_scaled])
            east_scaled_path = exp_dir / f"{args.output_prefix}_east_binned_scaled.txt"
            np.savetxt(
                east_scaled_path,
                east_scaled_output,
                header="wavelength_um  half_bin_width_um  transit_depth_east_scaled",
                fmt="%.10e",
            )
            print(f"[info] East scaled binned spectrum saved to: {east_scaled_path}")

            D_binned_west = np.asarray(result["binned_west"], dtype=float)[west_slice]
            D_binned_west_scaled = np.asarray(result.get("binned_west_scaled", 0.5 * result["binned_west"]), dtype=float)[west_slice]
            west_output = np.column_stack([west_wl, west_dwl, D_binned_west])
            west_path = exp_dir / f"{args.output_prefix}_west_binned.txt"
            np.savetxt(
                west_path,
                west_output,
                header="wavelength_um  half_bin_width_um  transit_depth_west",
                fmt="%.10e",
            )
            print(f"[info] West binned spectrum saved to: {west_path}")
            west_scaled_output = np.column_stack([west_wl, west_dwl, D_binned_west_scaled])
            west_scaled_path = exp_dir / f"{args.output_prefix}_west_binned_scaled.txt"
            np.savetxt(
                west_scaled_path,
                west_scaled_output,
                header="wavelength_um  half_bin_width_um  transit_depth_west_scaled",
                fmt="%.10e",
            )
            print(f"[info] West scaled binned spectrum saved to: {west_scaled_path}")
        else:
            D_binned = np.asarray(result["binned"], dtype=float)
            binned_output = np.column_stack([obs["wl"], obs["dwl"], D_binned])
            binned_path = exp_dir / f"{args.output_prefix}_binned.txt"
            np.savetxt(
                binned_path, binned_output,
                header="wavelength_um  half_bin_width_um  transit_depth",
                fmt="%.10e",
            )
            print(f"[info] Binned spectrum saved to: {binned_path}")

    t_end = time.perf_counter()
    print(f"[info] Total runtime: {t_end - t_start:.1f} s")


if __name__ == "__main__":
    main()
