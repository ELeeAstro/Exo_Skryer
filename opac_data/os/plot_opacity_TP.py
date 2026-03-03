#!/usr/bin/env python3
"""
plot_opacity_TP.py
==================

Plot opacity (cross section) from an npz file at a specific temperature and pressure point.

Usage:
    python plot_opacity_TP.py <filename.npz> --T <temp_K> --P <pressure_bar>

Example:
    python plot_opacity_TP.py H2O_R10000.npz --T 1500 --P 1e-3
    python plot_opacity_TP.py CO2_dnu_1.npz --T 1000 --P 0.1
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def find_nearest_index(array, value):
    """Find index of nearest value in array."""
    idx = np.argmin(np.abs(array - value))
    return idx


def plot_opacity_at_TP(filename, T_target, P_target, wl_range=None, log_scale=True):
    """
    Plot opacity from npz file at specified temperature and pressure.

    Parameters
    ----------
    filename : str
        Path to npz file
    T_target : float
        Target temperature in Kelvin
    P_target : float
        Target pressure in bars
    wl_range : tuple of float, optional
        Wavelength range (min, max) in microns. If None, plot full range.
    log_scale : bool
        Use log scale for y-axis (default: True)
    """
    # Load data
    data = np.load(filename)

    molecule = str(data['molecule'])
    T_grid = data['temperature']  # Kelvin
    P_grid = data['pressure']     # bars
    wl = data['wavelength']       # microns
    xsec = 10.0**data['cross_section']  # (nT, nP, nwl) in cm^2

    # Find nearest T and P indices
    iT = find_nearest_index(T_grid, T_target)
    iP = find_nearest_index(P_grid, P_target)

    T_actual = T_grid[iT]
    P_actual = P_grid[iP]

    print(f"File: {Path(filename).name}")
    print(f"Molecule: {molecule}")
    print(f"Temperature grid: {T_grid.min():.0f} - {T_grid.max():.0f} K ({len(T_grid)} points)")
    print(f"Pressure grid: {P_grid.min():.2e} - {P_grid.max():.2e} bar ({len(P_grid)} points)")
    print(f"Wavelength grid: {wl.min():.3f} - {wl.max():.3f} μm ({len(wl)} points)")
    print()
    print(f"Requested: T = {T_target:.0f} K, P = {P_target:.2e} bar")
    print(f"Nearest:   T = {T_actual:.0f} K, P = {P_actual:.2e} bar")

    # Extract cross section at this T, P
    xsec_TP = xsec[iT, iP, :]  # (nwl,)
    #xsec_TP = xsec_TP[::-1]

    # Apply wavelength range filter if specified
    if wl_range is not None:
        wl_min, wl_max = wl_range
        mask = (wl >= wl_min) & (wl <= wl_max)
        wl_plot = wl[mask]
        xsec_plot = xsec_TP[mask]
    else:
        wl_plot = wl
        xsec_plot = xsec_TP

    # Statistics
    xsec_nonzero = xsec_plot[xsec_plot > 0]
    if len(xsec_nonzero) > 0:
        print(f"\nCross section stats (cm²):")
        print(f"  Min (nonzero): {xsec_nonzero.min():.3e}")
        print(f"  Max:           {xsec_nonzero.max():.3e}")
        print(f"  Median:        {np.median(xsec_nonzero):.3e}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    if log_scale:
        # Clip zeros for log scale
        xsec_plot_safe = np.where(xsec_plot > 0, xsec_plot, np.nan)
        ax.plot(wl_plot, xsec_plot_safe, linewidth=0.5, alpha=0.8)
        ax.set_yscale('log')
        ax.set_ylabel('Cross section [cm²]', fontsize=12)
    else:
        ax.plot(wl_plot, xsec_plot, linewidth=0.5, alpha=0.8)
        ax.set_ylabel('Cross section [cm²]', fontsize=12)

    ax.set_xlabel('Wavelength [μm]', fontsize=12)
    ax.set_title(f'{molecule} opacity at T = {T_actual:.0f} K, P = {P_actual:.2e} bar',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim(wl_plot.min(), wl_plot.max())

    ax.set_xscale('log')

    plt.tight_layout()

    # Save figure
    outname = f"{Path(filename).stem}_T{T_actual:.0f}K_P{P_actual:.2e}bar.png"
    plt.savefig(outname, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {outname}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot opacity from npz file at specific T and P',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_opacity_TP.py H2O_R10000.npz --T 1500 --P 1e-3
  python plot_opacity_TP.py CO2_dnu_1.npz --T 1000 --P 0.1 --wl-min 1 --wl-max 5
  python plot_opacity_TP.py CH4_dnu_05.npz --T 800 --P 1e-5 --linear
        """
    )

    parser.add_argument('filename', type=str, help='Path to npz opacity file')
    parser.add_argument('--T', type=float, required=True, help='Temperature in Kelvin')
    parser.add_argument('--P', type=float, required=True, help='Pressure in bars')
    parser.add_argument('--wl-min', type=float, default=None, help='Minimum wavelength (μm)')
    parser.add_argument('--wl-max', type=float, default=None, help='Maximum wavelength (μm)')
    parser.add_argument('--linear', action='store_true', help='Use linear y-axis (default: log)')

    args = parser.parse_args()

    # Validate file exists
    if not Path(args.filename).exists():
        print(f"Error: File '{args.filename}' not found")
        return

    # Determine wavelength range
    wl_range = None
    if args.wl_min is not None and args.wl_max is not None:
        wl_range = (args.wl_min, args.wl_max)

    # Plot
    plot_opacity_at_TP(
        args.filename,
        args.T,
        args.P,
        wl_range=wl_range,
        log_scale=not args.linear
    )


if __name__ == '__main__':
    main()
