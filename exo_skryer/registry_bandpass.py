"""
instru_bandpass.py
==================

Overview:
    Prepare and cache instrument bandpass / response information for each
    observational bin, and expose JAX-ready padded arrays for fast convolution.

Notes:
    - Assumes obs['wl'], obs['dwl'], cut_grid, and the filter files all use
      the same wavelength units.
    - For each bin, we define weights w(λ) and a normalization
          norm = ∫ w(λ) dλ
      over the *actual* wavelength slice used.
      The later convolution should then compute:
          y_bin = ∫ F(λ) w(λ) dλ / norm
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple, List

import numpy as np
import jax.numpy as jnp


__all__ = [
    "BinConvolutionEntry",
    "reset_bandpass_registry",
    "has_bandpass_data",
    "load_bandpass_registry",
    "bandpass_num_bins",
    "bandpass_bin_edges",
    "bandpass_wavelengths_padded",
    "bandpass_weights_padded",
    "bandpass_indices_padded",
    "bandpass_norms",
]

# --- Dataclass and global registries ---


@dataclass(frozen=True)
class BinConvolutionEntry:
    """
    Holds information needed to convolve a single observational bin at runtime.
    Note: During preprocessing, all arrays are NumPy (CPU)
    They get converted to JAX (device) only at the final cache creation step
    All arrays kept as float64 for maximum accuracy in bandpass convolution
    """
    method: str
    wavelengths: np.ndarray         # NumPy during preprocessing (float64) - Slice of the high-res wavelength grid
    weights: np.ndarray             # NumPy during preprocessing (float64) - Corresponding weights for the slice
    norm: float                     # Pre-calculated normalization constant for the bin
    indices: Tuple[int, int]        # (start, end) index into the CUT wavelength grid
    bin_edges: Tuple[float, float]  # Intended left and right edges of the bin


# Global entries and JAX caches
_BAND_ENTRIES: Tuple[BinConvolutionEntry, ...] = ()

_BAND_WL_PAD_CACHE: jnp.ndarray | None = None
_BAND_W_PAD_CACHE: jnp.ndarray | None = None
_BAND_IDX_PAD_CACHE: jnp.ndarray | None = None
_BAND_NORM_CACHE: jnp.ndarray | None = None

# Map instrument modes to filter filenames
_MODE_TO_FILE = {
    "S36": "Spitzer_irac1_bandpass.dat",
    "S45": "Spitzer_irac2_bandpass.dat",
}


# --- Internal helpers ---


def _clear_cache():
    """Clear all lru_cache-powered helper functions."""
    bandpass_num_bins.cache_clear()
    bandpass_bin_edges.cache_clear()
    bandpass_wavelengths_padded.cache_clear()
    bandpass_weights_padded.cache_clear()
    bandpass_indices_padded.cache_clear()
    bandpass_norms.cache_clear()


def reset_bandpass_registry():
    """
    Reset all bandpass-related registries and caches.
    """
    global _BAND_ENTRIES, _BAND_WL_PAD_CACHE, _BAND_W_PAD_CACHE, _BAND_IDX_PAD_CACHE, _BAND_NORM_CACHE
    _BAND_ENTRIES = ()
    _BAND_WL_PAD_CACHE = None
    _BAND_W_PAD_CACHE = None
    _BAND_IDX_PAD_CACHE = None
    _BAND_NORM_CACHE = None
    _clear_cache()


def has_bandpass_data() -> bool:
    """
    Returns True if the bandpass registry has been initialised.
    """
    return bool(_BAND_ENTRIES)

def get_filter_data(mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and caches the raw transmission data (wavelength, throughput) for a given filter mode.

    Returns
    -------
    wl_filter : np.ndarray
        Wavelengths of the filter transmission curve.
    throughput : np.ndarray
        Corresponding throughput values (dimensionless).
    """
    base_dir = Path(__file__).resolve().parent.parent / "telescope_data"
    if mode not in _MODE_TO_FILE:
        raise FileNotFoundError(f"No transmission file is mapped for filter mode '{mode}'.")
    path = base_dir / _MODE_TO_FILE[mode]
    if not path.exists():
        raise FileNotFoundError(f"Transmission file not found: {path}")

    print(f"[info] Loading filter data for '{mode}' from {path}")

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) < 2:
                continue

            try:
                wl = float(parts[0])
                throughput = float(parts[1])
                rows.append((wl, throughput))
            except ValueError:
                # Skip non-numeric header lines like "Photon counter"
                continue

    if not rows:
        raise ValueError(f"No valid transmission data found in {path}")

    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1]


# --- Main preparation function (NumPy → JAX) ---


def load_bandpass_registry(
    obs: dict,
    full_grid: np.ndarray,
    cut_grid: np.ndarray,
) -> None:
    """
    Build the bandpass registry and JAX-ready padded arrays for each observational bin.

    Parameters
    ----------
    obs : dict
        Observation info, must contain:
            - 'wl' : observed central wavelengths (1D array)
            - 'dwl': half-widths of each bin (1D array)
            - 'response_mode': array of strings / identifiers for each bin
              (e.g. "boxcar", "S36", "S45").
    full_grid : `~numpy.ndarray`
        Full high-resolution wavelength grid (currently unused but kept for API stability).
    cut_grid : `~numpy.ndarray`
        Cut high-resolution wavelength grid on which convolution will be performed.
    """
    global _BAND_ENTRIES, _BAND_WL_PAD_CACHE, _BAND_W_PAD_CACHE, _BAND_IDX_PAD_CACHE, _BAND_NORM_CACHE

    wl_hi = np.asarray(cut_grid, dtype=float)  # high-res grid used for convolution
    wl_obs = np.asarray(obs["wl"], dtype=float)
    dwl_obs = np.asarray(obs["dwl"], dtype=float)
    response_modes = np.asarray(obs["response_mode"])

    nobs = wl_obs.shape[0]
    entries: List[BinConvolutionEntry] = []

    # --- First pass: build per-bin entries on irregular slices (NumPy) ---
    for idx in range(nobs):
        # Mode for this bin
        if response_modes.size:
            mode = str(response_modes[idx]).strip()
        else:
            mode = "boxcar"

        final_method = "boxcar" if (not mode or mode.lower() == "boxcar") else mode

        # Pre-load filter data if needed; fallback to boxcar on failure
        filter_wl = filter_throughput = None
        if final_method.lower() != "boxcar":
            try:
                filter_wl, filter_throughput = get_filter_data(final_method)
            except FileNotFoundError as e:
                print(f"[warn] {e}. Skipping bin {idx} and treating as boxcar.")
                final_method = "boxcar"

        # Bin edges
        center = wl_obs[idx]
        half_width = dwl_obs[idx]
        low, high = center - half_width, center + half_width

        # Index range in high-res grid
        start_idx = np.searchsorted(wl_hi, low, side="left")
        end_idx = np.searchsorted(wl_hi, high, side="right")

        # Fallback: if the bin is empty or completely outside the grid, use nearest point
        if start_idx >= end_idx:
            nearest = np.abs(wl_hi - center).argmin()
            start_idx = nearest
            end_idx = nearest + 1

        wl_slice = wl_hi[start_idx:end_idx]

        if wl_slice.size == 0:
            # Very defensive; shouldn't happen given the fallback above.
            print(f"[warn] Empty wavelength slice for bin {idx}, using nearest grid point.")
            nearest = np.abs(wl_hi - center).argmin()
            wl_slice = wl_hi[nearest:nearest + 1]
            start_idx, end_idx = nearest, nearest + 1

        # Weights
        if final_method.lower() == "boxcar":
            weights_slice = np.ones_like(wl_slice)
        else:
            weights_slice = np.interp(
                wl_slice, filter_wl, filter_throughput, left=0.0, right=0.0
            )

        # Norm = ∫ w(λ) dλ over the actual slice; keeps numerator/denominator consistent
        if wl_slice.size > 1:
            norm = float(np.trapezoid(weights_slice, x=wl_slice))
        else:
            norm = 1.0

        if norm <= 0.0:
            print(
                f"[warn] Non-positive norm ({norm}) for bin {idx} with method "
                f"{final_method!r}. Falling back to norm=1.0."
            )
            norm = 1.0

        entry = BinConvolutionEntry(
            method=final_method,
            wavelengths=wl_slice.astype(np.float64),     # NumPy (float64)
            weights=weights_slice.astype(np.float64),    # NumPy (float64)
            norm=norm,
            indices=(int(start_idx), int(end_idx)),
            bin_edges=(float(low), float(high)),
        )
        entries.append(entry)

    _BAND_ENTRIES = tuple(entries)

    if not _BAND_ENTRIES:
        # Nothing to do, clear everything
        reset_bandpass_registry()
        return

    # --- Second pass: build padded rectangular arrays (NumPy) and convert to JAX ---

    n_bins = len(_BAND_ENTRIES)
    max_len = max(int(e.wavelengths.shape[0]) for e in _BAND_ENTRIES)

    padded_wl = np.zeros((n_bins, max_len), dtype=float)
    padded_w = np.zeros((n_bins, max_len), dtype=float)
    padded_idx = np.zeros((n_bins, max_len), dtype=int)
    norms_np = np.zeros((n_bins,), dtype=float)

    for i, e in enumerate(_BAND_ENTRIES):
        wl = np.asarray(np.array(e.wavelengths), dtype=float)
        w = np.asarray(np.array(e.weights), dtype=float)
        start, end = e.indices
        length = wl.size

        idxs = np.arange(start, end, dtype=int)

        # Fill valid part
        padded_wl[i, :length] = wl
        padded_w[i, :length] = w
        padded_idx[i, :length] = idxs

        # Pad tail: copy last wavelength, set weights=0, repeat last index
        if length < max_len:
            padded_wl[i, length:] = wl[-1]
            padded_w[i, length:] = 0.0
            padded_idx[i, length:] = idxs[-1]

        norms_np[i] = float(e.norm)

    # ============================================================================
    # CRITICAL: Convert NumPy arrays to JAX arrays here (ONE transfer to device)
    # ============================================================================
    # All preprocessing is done in NumPy (CPU). Now we send the final data
    # to the device (GPU/CPU as configured) for use in JIT-compiled forward model.
    # All arrays kept as float64 for maximum accuracy in bandpass convolution.
    # ============================================================================

    print(f"[Bandpass] Transferring {n_bins} bins to device...")

    _BAND_WL_PAD_CACHE = jnp.asarray(padded_wl, dtype=jnp.float64)
    _BAND_W_PAD_CACHE = jnp.asarray(padded_w, dtype=jnp.float64)
    _BAND_IDX_PAD_CACHE = jnp.asarray(padded_idx, dtype=jnp.int32)
    _BAND_NORM_CACHE = jnp.asarray(norms_np, dtype=jnp.float64)

    print(f"[Bandpass] Wavelength cache: {_BAND_WL_PAD_CACHE.shape} (dtype: {_BAND_WL_PAD_CACHE.dtype})")
    print(f"[Bandpass] Weights cache: {_BAND_W_PAD_CACHE.shape} (dtype: {_BAND_W_PAD_CACHE.dtype})")
    print(f"[Bandpass] Index cache: {_BAND_IDX_PAD_CACHE.shape} (dtype: {_BAND_IDX_PAD_CACHE.dtype})")
    print(f"[Bandpass] Norm cache: {_BAND_NORM_CACHE.shape} (dtype: {_BAND_NORM_CACHE.dtype})")

    # Estimate memory usage
    wl_mb = _BAND_WL_PAD_CACHE.size * _BAND_WL_PAD_CACHE.itemsize / 1024**2
    w_mb = _BAND_W_PAD_CACHE.size * _BAND_W_PAD_CACHE.itemsize / 1024**2
    idx_mb = _BAND_IDX_PAD_CACHE.size * _BAND_IDX_PAD_CACHE.itemsize / 1024**2
    norm_mb = _BAND_NORM_CACHE.size * _BAND_NORM_CACHE.itemsize / 1024**2
    total_mb = wl_mb + w_mb + idx_mb + norm_mb
    print(f"[Bandpass] Estimated device memory: {total_mb:.3f} MB (wl: {wl_mb:.3f}, w: {w_mb:.3f}, idx: {idx_mb:.3f}, norm: {norm_mb:.3f})")

    _clear_cache()


# --- lru_cache helper accessors (JAX-ready) ---


@lru_cache(None)
def bandpass_num_bins() -> int:
    """
    Number of observational bins in the bandpass registry.
    """
    return len(_BAND_ENTRIES)


@lru_cache(None)
def bandpass_bin_edges() -> jnp.ndarray:
    """
    Bin edges as an array of shape (n_bins, 2): [λ_low, λ_high] for each bin.
    """
    if not _BAND_ENTRIES:
        raise RuntimeError("Bandpass registry empty; call load_bandpass_registry() first.")
    edges = np.array([e.bin_edges for e in _BAND_ENTRIES], dtype=float)
    return jnp.asarray(edges)


@lru_cache(None)
def bandpass_wavelengths_padded() -> jnp.ndarray:
    """
    Padded wavelength grid for each bin, shape (n_bins, max_len).
    """
    if _BAND_WL_PAD_CACHE is None:
        raise RuntimeError("Bandpass padded arrays not built; call load_bandpass_registry() first.")
    return _BAND_WL_PAD_CACHE


@lru_cache(None)
def bandpass_weights_padded() -> jnp.ndarray:
    """
    Padded weights for each bin, shape (n_bins, max_len).
    """
    if _BAND_W_PAD_CACHE is None:
        raise RuntimeError("Bandpass padded arrays not built; call load_bandpass_registry() first.")
    return _BAND_W_PAD_CACHE


@lru_cache(None)
def bandpass_indices_padded() -> jnp.ndarray:
    """
    Padded index array into the high-res spectrum grid, shape (n_bins, max_len).
    """
    if _BAND_IDX_PAD_CACHE is None:
        raise RuntimeError("Bandpass padded arrays not built; call load_bandpass_registry() first.")
    return _BAND_IDX_PAD_CACHE


@lru_cache(None)
def bandpass_norms() -> jnp.ndarray:
    """
    Normalisation constants for each bin, shape (n_bins,).
    """
    if _BAND_NORM_CACHE is None:
        raise RuntimeError("Bandpass norms not built; call load_bandpass_registry() first.")
    return _BAND_NORM_CACHE

