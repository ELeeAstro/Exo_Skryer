"""
read_stellar.py
================

Overview:
    Helper utilities to read stellar spectra files and interpolate them
    onto the master wavelength grid used by the forward model.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import jax.numpy as jnp
import numpy as np


__all__ = ["read_stellar_spectrum"]


def _resolve_stellar_path(cfg, base_dir: Optional[Path]) -> Path | None:
    data_cfg = getattr(cfg, "data", None)
    stellar_path = getattr(data_cfg, "stellar", None) if data_cfg is not None else None
    if stellar_path is None:
        obs_cfg = getattr(cfg, "obs", None)
        if obs_cfg is not None:
            stellar_path = getattr(obs_cfg, "stellar", None)
    if not stellar_path:
        return None
    path = Path(stellar_path).expanduser()
    if not path.is_absolute():
        if base_dir is not None:
            path = (Path(base_dir) / path).resolve()
        else:
            path = (Path.cwd() / path).resolve()
    return path


def _load_native_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1 or data.shape[1] < 2:
        raise ValueError(f"Stellar spectrum '{path}' must have at least two columns (wl, flux).")
    wavelengths = np.asarray(data[:, 0], dtype=float)
    flux = np.asarray(data[:, 1], dtype=float)
    sort_idx = np.argsort(wavelengths)
    return wavelengths[sort_idx], flux[sort_idx]


def _compute_bin_edges(lam_master: np.ndarray) -> np.ndarray:
    edges = np.zeros(lam_master.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (lam_master[1:] + lam_master[:-1])
    spacing_start = lam_master[1] - lam_master[0]
    spacing_end = lam_master[-1] - lam_master[-2]
    edges[0] = lam_master[0] - 0.5 * spacing_start
    edges[-1] = lam_master[-1] + 0.5 * spacing_end
    return edges


def _band_average(
    wl_native: np.ndarray,
    flux_native: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    out = np.empty(edges.size - 1, dtype=float)
    for i in range(out.size):
        left = edges[i]
        right = edges[i + 1]
        mask = (wl_native >= left) & (wl_native <= right)
        if np.any(mask):
            wl_seg = wl_native[mask]
            fl_seg = flux_native[mask]
        else:
            wl_seg = np.array([left, right], dtype=float)
            fl_seg = np.interp(wl_seg, wl_native, flux_native, left=flux_native[0], right=flux_native[-1])
        out[i] = np.trapezium(fl_seg, wl_seg) / (right - left)
    return out


def read_stellar_spectrum(
    cfg,
    lam_master: Iterable[float],
    ck_mode: bool,
    base_dir: Optional[Path] = None,
) -> jnp.ndarray | None:
    """Read and interpolate the stellar spectrum onto the master grid."""
    path = _resolve_stellar_path(cfg, base_dir)
    if path is None:
        return None
    if not path.exists():
        print(f"[read_stellar] Stellar spectrum not found at {path}; skipping.")
        return None
    wl_native, flux_native = _load_native_spectrum(path)
    lam_master = np.asarray(lam_master, dtype=float)

    if ck_mode:
        edges = _compute_bin_edges(lam_master)
        flux_master = _band_average(wl_native, flux_native, edges)
    else:
        flux_master = np.interp(
            lam_master,
            wl_native,
            flux_native,
            left=flux_native[0],
            right=flux_native[-1],
        )
    return jnp.asarray(flux_master)
