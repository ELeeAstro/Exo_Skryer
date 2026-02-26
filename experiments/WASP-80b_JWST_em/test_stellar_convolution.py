#!/usr/bin/env python3
"""
Test script to convolve a high-resolution stellar spectrum onto the
opacity-sampled master wavelength grid (R20000).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_stellar_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in stellar spectrum: {path}")
    wl = np.asarray(data[:, 0], dtype=float)
    flux = np.asarray(data[:, 1], dtype=float)
    idx = np.argsort(wl)
    wl = wl[idx]
    flux = flux[idx]
    return wl, flux


def load_two_column_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in spectrum file: {path}")
    wl = np.asarray(data[:, 0], dtype=float)
    flux = np.asarray(data[:, 1], dtype=float)
    idx = np.argsort(wl)
    return wl[idx], flux[idx]


def load_master_wavelength(path: Path) -> np.ndarray:
    # wl_R20000.txt usually has a first line with only N, then "index wavelength".
    values = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) == 1:
                # Header count line (e.g. "92106")
                continue
            values.append(float(parts[-1]))

    if not values:
        raise ValueError(f"No wavelength values parsed from: {path}")

    wl = np.asarray(values, dtype=float)
    idx = np.argsort(wl)
    return wl[idx]


def compute_bin_edges(lam_master: np.ndarray) -> np.ndarray:
    if lam_master.size < 2:
        raise ValueError("Master wavelength grid needs at least 2 points.")
    edges = np.empty(lam_master.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (lam_master[1:] + lam_master[:-1])
    edges[0] = lam_master[0] - 0.5 * (lam_master[1] - lam_master[0])
    edges[-1] = lam_master[-1] + 0.5 * (lam_master[-1] - lam_master[-2])
    return edges


def band_average_fast(
    wl_native: np.ndarray,
    flux_native: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    # Interpolate in log10-space to preserve shape across dynamic range.
    tiny = np.finfo(float).tiny
    log_flux_native = np.log10(np.clip(flux_native, tiny, None))

    left, right = edges[0], edges[-1]
    interior = (wl_native > left) & (wl_native < right)
    x_aug = np.concatenate((edges, wl_native[interior]))
    x_aug = np.unique(x_aug)

    logf_aug = np.interp(
        x_aug,
        wl_native,
        log_flux_native,
        left=log_flux_native[0],
        right=log_flux_native[-1],
    )
    f_aug = 10.0 ** logf_aug

    dx = np.diff(x_aug)
    trap = 0.5 * (f_aug[1:] + f_aug[:-1]) * dx
    cum = np.concatenate(([0.0], np.cumsum(trap)))

    edge_idx = np.searchsorted(x_aug, edges)
    bin_int = cum[edge_idx[1:]] - cum[edge_idx[:-1]]
    return bin_int / np.diff(edges)


def plot_convolution(
    wl_native: np.ndarray,
    flux_native: np.ndarray,
    wl_master: np.ndarray,
    flux_conv: np.ndarray,
    flux_interp: np.ndarray,
    wl_overlay: np.ndarray | None,
    flux_overlay: np.ndarray | None,
    overlay_label: str,
    plot_path: Path,
    show: bool,
) -> None:
    # Downsample native spectrum for plotting speed/readability.
    n_plot_native = min(12000, wl_native.size)
    if n_plot_native < wl_native.size:
        idx = np.linspace(0, wl_native.size - 1, n_plot_native).astype(int)
        wl_native_plot = wl_native[idx]
        flux_native_plot = flux_native[idx]
    else:
        wl_native_plot = wl_native
        flux_native_plot = flux_native

    ratio = flux_conv / np.maximum(flux_interp, 1.0e-300)

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    ax0.plot(wl_native_plot, flux_native_plot, lw=0.6, alpha=0.45, label="Native stellar (downsampled)")
    ax0.plot(wl_master, flux_interp, lw=0.9, alpha=0.85, label="Master-grid interp (log-space)")
    ax0.plot(wl_master, flux_conv, lw=1.1, alpha=0.95, label="Master-grid bin-averaged")
    if wl_overlay is not None and flux_overlay is not None:
        ax0.plot(wl_overlay, flux_overlay, lw=1.0, alpha=0.9, label=overlay_label)
    ax0.set_yscale("log")
    ax0.set_ylabel("Flux")
    ax0.set_title("WASP-80 stellar spectrum convolution onto wl_R20000")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="best", fontsize=9)

    ax1.plot(wl_master, ratio, lw=0.9, color="black")
    ax1.axhline(1.0, lw=0.8, ls="--", color="tab:red")
    ax1.set_xlabel("Wavelength [um]")
    ax1.set_ylabel("Conv/Interp")
    ax1.grid(True, alpha=0.25)

    fig.savefig(plot_path, dpi=180)
    print(f"[ok] Wrote plot: {plot_path}")

    if show:
        try:
            plt.show()
        except Exception as exc:
            print(f"[warn] Could not show plot window ({exc}). Plot was still saved to disk.")
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]

    parser = argparse.ArgumentParser(description="Convolve WASP-80 stellar spectrum onto wl_R20000 grid.")
    parser.add_argument(
        "--stellar",
        type=Path,
        default=script_dir / "WASP80_spectrum_x10.txt",
        help="Path to high-resolution stellar spectrum (2 columns: wavelength[um], flux).",
    )
    parser.add_argument(
        "--master",
        type=Path,
        default=repo_root / "opac_data" / "lbl" / "wl_R20000.txt",
        help="Path to master wavelength grid file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=script_dir / "WASP80_spectrum_on_wl_R20000_convolved.txt",
        help="Output file path for convolved spectrum (2 columns: wavelength[um], flux).",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=script_dir / "WASP80_spectrum_on_wl_R20000_convolved.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        default=script_dir / "WASP-80.txt",
        help="Optional 2-column spectrum to overplot (default: WASP-80.txt).",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable overlay of WASP-80.txt (or --overlay file).",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Show the plot window on screen after saving.",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Do not show plot window; save only.",
    )
    parser.set_defaults(show=True)
    args = parser.parse_args()

    wl_native, flux_native = load_stellar_spectrum(args.stellar.expanduser().resolve())
    wl_master = load_master_wavelength(args.master.expanduser().resolve())
    edges = compute_bin_edges(wl_master)
    flux_conv = band_average_fast(wl_native, flux_native, edges)

    # Optional comparison against plain interpolation.
    log_flux_native = np.log10(np.clip(flux_native, np.finfo(float).tiny, None))
    flux_interp = 10.0 ** np.interp(
        wl_master,
        wl_native,
        log_flux_native,
        left=log_flux_native[0],
        right=log_flux_native[-1],
    )
    rel = np.abs((flux_conv - flux_interp) / np.maximum(np.abs(flux_interp), 1.0e-300))

    out = args.out.expanduser().resolve()
    np.savetxt(
        out,
        np.column_stack((wl_master, flux_conv)),
        fmt="%.15e",
        header="wavelength_um flux_convolved",
    )

    print(f"[ok] Wrote convolved spectrum: {out}")
    print(f"[info] Native stellar grid: N={wl_native.size}, wl=[{wl_native.min():.6f}, {wl_native.max():.6f}] um")
    print(f"[info] Master grid:         N={wl_master.size}, wl=[{wl_master.min():.6f}, {wl_master.max():.6f}] um")
    print(f"[info] Convolved flux:      [{flux_conv.min():.6e}, {flux_conv.max():.6e}]")
    print(f"[info] vs interp rel diff:  median={np.median(rel):.3e}, max={np.max(rel):.3e}")

    wl_overlay = None
    flux_overlay = None
    overlay_label = "WASP-80.txt"
    if not args.no_overlay:
        overlay_path = args.overlay.expanduser().resolve()
        if overlay_path.exists():
            wl_overlay, flux_overlay = load_two_column_spectrum(overlay_path)
            overlay_label = overlay_path.name
            print(
                f"[info] Overlay spectrum:   N={wl_overlay.size}, "
                f"wl=[{wl_overlay.min():.6f}, {wl_overlay.max():.6f}] um, "
                f"path={overlay_path}"
            )
        else:
            print(f"[warn] Overlay file not found, skipping: {overlay_path}")

    plot_path = args.plot.expanduser().resolve()
    plot_convolution(
        wl_native=wl_native,
        flux_native=flux_native,
        wl_master=wl_master,
        flux_conv=flux_conv,
        flux_interp=flux_interp,
        wl_overlay=wl_overlay,
        flux_overlay=flux_overlay,
        overlay_label=overlay_label,
        plot_path=plot_path,
        show=args.show,
    )


if __name__ == "__main__":
    main()
