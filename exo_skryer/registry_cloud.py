"""
registry_cloud.py
=================
"""

from typing import Dict, NamedTuple, Optional, Tuple
from pathlib import Path
import numpy as np
import jax.numpy as jnp


__all__ = [
    "compute_kk_grid_cache",
    "get_or_create_kk_cache",
    "clear_kk_cache",
    "get_kk_cache_stats",
    "set_cloud_nk_data",
    "get_cloud_nk_data",
    "clear_cloud_nk_data",
    "has_cloud_nk_data",
    "cloud_nk_wavelength",
    "cloud_nk_n",
    "cloud_nk_k",
    "load_cloud_nk_data",
]


class KKGridCache(NamedTuple):
    """
    Precomputed grid-dependent quantities for Kramers-Kronig relation.

    This cache stores only O(N) quantities to avoid memory issues with large grids.
    The O(N²) alpha_inv matrix is computed on-the-fly, which is still fast with JAX
    and avoids storing GBs of data for fine grids.

    For N=33219: storing alpha_inv would need ~8.8 GB!
    This design uses only ~0.5 MB instead.

    Note: Arrays are stored as JAX arrays (device) for use in JIT-compiled functions.
    All preprocessing is done in NumPy (CPU), then transferred to device once.
    All arrays kept as float64 for maximum accuracy in KK calculations.
    """
    nu: jnp.ndarray            # (N,) original wavenumber grid [cm⁻¹] - JAX array (float64)
    trap_weights: jnp.ndarray  # (N,) trapezoid integration weights - JAX array (float64)


# Global registry: array id -> KKGridCache
_KK_GRID_REGISTRY: Dict[int, KKGridCache] = {}


def compute_kk_grid_cache(nu: jnp.ndarray) -> KKGridCache:
    """
    Precompute grid-dependent quantities for Kramers-Kronig relation.

    This caches only O(N) quantities (trapezoid weights) to avoid memory issues.
    The O(N²) alpha_inv matrix is NOT cached - it's computed on-the-fly in the
    KK function, which is still fast with JAX and avoids huge memory usage.

    Parameters
    ----------
    nu : array (N,)
        Wavenumber grid, strictly increasing (e.g., cm⁻¹)

    Returns
    -------
    cache : KKGridCache
        Named tuple containing:
        - nu: original grid (JAX array, float64)
        - trap_weights: (N,) weights for trapezoid integration (JAX array, float64)
    """
    # Convert input to NumPy for preprocessing (CPU-based)
    nu_np = np.asarray(nu, dtype=np.float64)

    # Trapezoid rule integration weights (NumPy operations on CPU)
    # For f(ν) on grid, integral ≈ sum(f * trap_weights)
    dnu = nu_np[1:] - nu_np[:-1]  # (N-1,) spacing between points
    trap_weights_np = np.zeros(len(nu_np), dtype=np.float64)
    trap_weights_np[:-1] += 0.5 * dnu  # Left endpoints
    trap_weights_np[1:] += 0.5 * dnu   # Right endpoints

    # ============================================================================
    # CRITICAL: Convert NumPy arrays to JAX arrays here (ONE transfer to device)
    # ============================================================================
    # All preprocessing is done in NumPy (CPU). Now we send the final data
    # to the device (GPU/CPU as configured) for use in JIT-compiled KK function.
    # All arrays kept as float64 for maximum accuracy in Kramers-Kronig calculations.
    # ============================================================================

    nu_jax = jnp.asarray(nu_np, dtype=jnp.float64)
    trap_weights_jax = jnp.asarray(trap_weights_np, dtype=jnp.float64)

    return KKGridCache(
        nu=nu_jax,
        trap_weights=trap_weights_jax,
    )


def get_or_create_kk_cache(nu: jnp.ndarray) -> KKGridCache:
    """
    Retrieve cached KK grid data or compute if not exists.

    This function provides transparent caching: on first call with a given
    grid, it computes and caches the expensive O(N²) quantities. Subsequent
    calls with the same grid array return the cached result instantly.

    Parameters
    ----------
    nu : array (N,)
        Wavenumber grid, strictly increasing (e.g., cm⁻¹)

    Returns
    -------
    cache : KKGridCache
        Cached or newly computed grid quantities
    """
    nu_id = id(nu)

    if nu_id not in _KK_GRID_REGISTRY:
        _KK_GRID_REGISTRY[nu_id] = compute_kk_grid_cache(nu)

    return _KK_GRID_REGISTRY[nu_id]


def clear_kk_cache() -> None:
    """
    Clear all cached KK grid data.

    Useful for memory management in long-running processes or when
    switching between different retrieval configurations with different grids.
    """
    global _KK_GRID_REGISTRY
    _KK_GRID_REGISTRY.clear()


def get_kk_cache_stats() -> dict:
    """
    Get statistics about the current KK cache state.

    Returns
    -------
    stats : dict
        Dictionary with keys:
        - 'num_cached_grids': Number of unique grids cached
        - 'cache_ids': List of array IDs in the cache
        - 'memory_estimate_mb': Approximate memory usage (if grids available)
    """
    num_grids = len(_KK_GRID_REGISTRY)

    # Estimate memory usage based on cached grids
    memory_mb = 0.0
    if num_grids > 0:
        for cache in _KK_GRID_REGISTRY.values():
            # Each cache has: nu (N), trap_weights (N)
            # All float64 = 8 bytes
            # NOTE: We no longer cache alpha_inv (would be N²) to save memory!
            N = len(cache.nu)
            memory_mb += (N + N) * 8 / 1024**2

    return {
        'num_cached_grids': num_grids,
        'cache_ids': list(_KK_GRID_REGISTRY.keys()),
        'memory_estimate_mb': round(memory_mb, 2),
    }


# ============================================================================
# Cloud n,k data registry
# ============================================================================

# Global registry for cloud refractive index data
_CLOUD_NK_DATA: Dict[str, jnp.ndarray] = {}


def set_cloud_nk_data(wl: jnp.ndarray, n: jnp.ndarray, k: jnp.ndarray) -> None:
    """
    Store cloud refractive index data in the global registry.

    This function caches wavelength-dependent complex refractive index arrays
    (n, k) for use in cloud opacity calculations. The data is stored globally
    and can be accessed by any cloud scheme that consumes cached n,k.

    Parameters
    ----------
    wl : `~jax.numpy.ndarray`, shape (nwl,)
        Wavelength grid in microns. MUST match the model wavelength grid.
    n : `~jax.numpy.ndarray`, shape (nwl,)
        Real part of refractive index (dimensionless).
    k : `~jax.numpy.ndarray`, shape (nwl,)
        Imaginary part of refractive index (dimensionless).
    """
    global _CLOUD_NK_DATA
    _CLOUD_NK_DATA['wl'] = jnp.asarray(wl)
    _CLOUD_NK_DATA['n'] = jnp.asarray(n)
    _CLOUD_NK_DATA['k'] = jnp.asarray(k)


def get_cloud_nk_data() -> Dict[str, jnp.ndarray]:
    """
    Retrieve cached cloud refractive index data.

    Returns
    -------
    nk_data : dict[str, `~jax.numpy.ndarray`]
        Dictionary containing:
        - 'wl': Wavelength grid in microns, shape (nwl,)
        - 'n': Real refractive index, shape (nwl,)
        - 'k': Imaginary refractive index, shape (nwl,)
        - 'conducting': int32 flag (1 if conducting, 0 otherwise) if loaded via load_cloud_nk_data()

    Raises
    ------
    RuntimeError
        If no cloud n,k data has been loaded (call `set_cloud_nk_data` first).
    """
    if not _CLOUD_NK_DATA:
        raise RuntimeError(
            "No cloud n,k data has been loaded. "
            "Call set_cloud_nk_data(wl, n, k) before using cached cloud n,k."
        )
    return _CLOUD_NK_DATA


def clear_cloud_nk_data() -> None:
    """
    Clear cached cloud refractive index data.

    Useful for memory management or when switching between different cloud
    species with different refractive index data.
    """
    global _CLOUD_NK_DATA
    _CLOUD_NK_DATA.clear()


def has_cloud_nk_data() -> bool:
    """Return True if cloud n,k arrays have been cached."""
    return bool(_CLOUD_NK_DATA)


def cloud_nk_wavelength() -> jnp.ndarray:
    return get_cloud_nk_data()["wl"]


def cloud_nk_n() -> jnp.ndarray:
    return get_cloud_nk_data()["n"]


def cloud_nk_k() -> jnp.ndarray:
    return get_cloud_nk_data()["k"]


def load_cloud_nk_data(path: str | Path, wl_master: np.ndarray) -> None:
    """Load a refractive index table from disk, interpolate to wl_master, and cache.

    Expected file format: whitespace-delimited columns with at least 3 columns:
        wavelength[um]  n  k
    Extra columns are ignored.

    The first line must be:
        <nwl> <conducting_flag>
    where conducting_flag is a Fortran-style boolean (e.g. ".True." / ".False.").
    This mirrors the behavior in the legacy Fortran tables:
      - Below the table range: clamp n,k to the lowest-wavelength values.
      - Above the table range:
          * if conducting: log-log extrapolate n,k using a point at 0.7*wl_max
          * else: hold n constant, decrease k ~ 1/wl
      - Within range: log-log interpolate n,k between bracketing grid points.
    """
    resolved = Path(path).expanduser().resolve()

    with resolved.open("r", encoding="utf-8") as f:
        header = f.readline().strip()

    parts = header.split()
    if len(parts) < 2:
        raise ValueError(
            f"Cloud n,k file {resolved} first line must be '<nwl> <.True./.False.>', got: {header!r}"
        )

    try:
        nwl_header = int(parts[0])
    except ValueError as e:
        raise ValueError(f"Cloud n,k file {resolved} invalid nwl in first line: {header!r}") from e

    cond_token = parts[1].strip().lower()
    if cond_token in (".true.", "true", "t", "1"):
        conducting = True
    elif cond_token in (".false.", "false", "f", "0"):
        conducting = False
    else:
        raise ValueError(
            f"Cloud n,k file {resolved} invalid conducting flag in first line: {header!r}"
        )

    data = np.loadtxt(resolved, comments="#", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Cloud n,k file {resolved} must have at least 3 columns (wl, n, k).")

    wl = np.asarray(data[:, 0], dtype=np.float64)
    n = np.asarray(data[:, 1], dtype=np.float64)
    k = np.asarray(data[:, 2], dtype=np.float64)

    if nwl_header != wl.size:
        # Not fatal: some tables may include extra comment rows or have stale headers.
        print(f"[warn] cloud nk header nwl={nwl_header} but read {wl.size} rows from {resolved}")

    # Ensure increasing wavelength for interpolation
    if not np.all(np.diff(wl) > 0):
        order = np.argsort(wl)
        wl = wl[order]
        n = n[order]
        k = k[order]

    wl_master = np.asarray(wl_master, dtype=np.float64)
    if wl_master.ndim != 1 or not np.all(np.diff(wl_master) > 0):
        raise ValueError("wl_master must be a 1D strictly-increasing wavelength grid.")

    nwl = wl.size
    wl_min = wl[0]
    wl_max = wl[-1]

    n_i = np.empty_like(wl_master, dtype=np.float64)
    k_i = np.empty_like(wl_master, dtype=np.float64)

    mask_low = wl_master < wl_min
    mask_high = wl_master > wl_max
    mask_mid = ~(mask_low | mask_high)

    # Below range: clamp to first values
    if np.any(mask_low):
        n_i[mask_low] = n[0]
        k_i[mask_low] = k[0]

    # Within range
    if np.any(mask_mid):
        xlog = np.log10(wl)
        nlog = np.log10(n)
        klog = np.log10(k)
        x_m = np.log10(wl_master[mask_mid])
        n_i[mask_mid] = 10.0 ** np.interp(x_m, xlog, nlog)
        k_i[mask_mid] = 10.0 ** np.interp(x_m, xlog, klog)

    # Above range: scheme-based extrapolation
    if np.any(mask_high):
        x = wl_master[mask_high]
        if not conducting:
            # n held constant; k decreases proportional to 1/wl
            n_i[mask_high] = n[-1]
            k_i[mask_high] = k[-1] * (wl_max / x)
        else:
            wl_ex = 0.7 * wl_max
            iwl_ex = np.searchsorted(wl, wl_ex, side="right") - 1
            # Ensure at least 3 points back from the end and valid index
            if (iwl_ex > nwl - 4) or (iwl_ex < 0):
                iwl_ex = nwl - 4
            fac = np.log10(x / wl_max) / np.log10(wl[iwl_ex] / wl_max)
            n_i[mask_high] = 10.0 ** (np.log10(n[-1]) + fac * np.log10(n[iwl_ex] / n[-1]))
            k_i[mask_high] = 10.0 ** (np.log10(k[-1]) + fac * np.log10(k[iwl_ex] / k[-1]))

    # Match Fortran floor
    n_i = np.maximum(n_i, 1e-99)
    k_i = np.maximum(k_i, 1e-99)

    set_cloud_nk_data(jnp.asarray(wl_master), jnp.asarray(n_i), jnp.asarray(k_i))
    _CLOUD_NK_DATA["conducting"] = jnp.asarray(1 if conducting else 0, dtype=jnp.int32)
