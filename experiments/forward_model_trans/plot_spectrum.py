#!/usr/bin/env python3
"""
plot_spectrum.py
================
Plot a forward model spectrum saved by ``run_forward_model.py``.

Usage
-----
    # Plot high-res spectrum only:
    python plot_spectrum.py --spectrum forward_spectrum_highres.txt

    # Overlay observational data:
    python plot_spectrum.py --spectrum forward_spectrum_highres.txt --obs ../../obs_data/WASP-107b_JWST.txt

    # Plot binned spectrum with obs overlay:
    python plot_spectrum.py --spectrum forward_spectrum_binned.txt --obs ../../obs_data/WASP-107b_JWST.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_forward_spectrum(
    spectrum_path: str,
    obs_path: str | None = None,
    outname: str = "forward_spectrum",
    show_plot: bool = True,
) -> None:

    exp_dir = Path(spectrum_path).resolve().parent
    data = np.loadtxt(spectrum_path)

    # Auto-detect format: 2 columns = high-res, 3 columns = binned
    is_binned = data.shape[1] >= 3

    fig, ax = plt.subplots(figsize=(10, 5))

    if is_binned:
        wl = data[:, 0]
        dwl = data[:, 1]
        depth = data[:, 2] * 100.0  # Convert to percent
        ax.errorbar(
            wl, depth, xerr=dwl,
            fmt="o-", ms=3, lw=1.2, capsize=2,
            color="C1", label="Binned model",
        )
    else:
        wl = data[:, 0]
        depth = data[:, 1] * 100.0
        ax.plot(
            wl, depth,
            lw=0.4, alpha=0.8, color="C1",
            label="High-res model", rasterized=True,
        )

    # Overlay observational data if provided
    if obs_path is not None:
        obs_raw = np.genfromtxt(obs_path, dtype=str, comments="#")
        obs_wl = obs_raw[:, 0].astype(float)
        obs_dwl = obs_raw[:, 1].astype(float)
        obs_depth = obs_raw[:, 2].astype(float) * 100.0
        obs_err = obs_raw[:, 3].astype(float) * 100.0
        ax.errorbar(
            obs_wl, obs_depth, xerr=obs_dwl, yerr=obs_err,
            fmt="o", ms=3, lw=1, alpha=0.8,
            color="C0", ecolor="C0", capsize=2,
            label="Observed",
        )

    ax.set_xlabel("Wavelength [um]", fontsize=14)
    ax.set_ylabel("Transit Depth [%]", fontsize=14)
    ax.set_xscale("log")

    # Wavelength tick formatting
    tick_locs = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0])
    wl_range = wl.min(), wl.max()
    visible = tick_locs[(tick_locs >= wl_range[0] * 0.8) & (tick_locs <= wl_range[1] * 1.2)]
    if len(visible) > 1:
        ax.set_xticks(visible)
        ax.set_xticklabels([f"{t:g}" for t in visible])

    ax.legend()
    fig.tight_layout()

    fig.savefig(exp_dir / f"{outname}.png", dpi=300)
    fig.savefig(exp_dir / f"{outname}.pdf")
    print(f"[plot] Saved: {exp_dir / outname}.png, .pdf")

    if show_plot:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Plot forward model spectrum")
    ap.add_argument("--spectrum", required=True, help="Path to spectrum txt file")
    ap.add_argument("--obs", default=None, help="Path to observational data txt file")
    ap.add_argument("--outname", default="forward_spectrum", help="Output filename stem")
    ap.add_argument("--no-show", action="store_true", help="Suppress plot window")
    args = ap.parse_args()

    plot_forward_spectrum(args.spectrum, args.obs, args.outname, not args.no_show)


if __name__ == "__main__":
    main()
