"""
registry_cloud.py
=================

Overview:
    Global cache registry for cloud opacity computations, specifically for
    optimizing Kramers-Kronig relation calculations.

Usage:
    The KK grid cache stores precomputed grid-dependent quantities that are
    reused across multiple evaluations of the KK relation with different
    extinction coefficients but the same wavelength/wavenumber grid.

Key Functions:
    - get_or_create_kk_cache: Retrieves or creates cached KK grid data
    - compute_kk_grid_cache: Precomputes all grid-dependent KK quantities

Notes:
    - The cache uses array identity (id()) as the key
    - This is O(N^2) memory but only computed once per unique grid
    - Provides significant speedup for repeated KK evaluations
"""

from typing import Dict, NamedTuple
import numpy as np
import jax.numpy as jnp


__all__ = [
    "compute_kk_grid_cache",
    "get_or_create_kk_cache",
    "clear_kk_cache",
    "get_kk_cache_stats"
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

    Notes
    -----
    Memory usage: Only ~16N bytes for float64
    - N=1000: ~16 KB
    - N=10000: ~160 KB
    - N=33219: ~531 KB (vs 8.8 GB if we cached alpha_inv!)

    The alpha_inv matrix is computed on-the-fly, which:
    - Saves massive memory for large grids
    - Is still fast due to JAX JIT fusion
    - Maintains full numerical accuracy

    Implementation:
    - All preprocessing done in NumPy (CPU) for efficiency
    - Single transfer to device (JAX) at the end
    - All arrays kept as float64 for maximum accuracy in KK calculations
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

    Notes
    -----
    - Cache key is array identity `id(nu)`, not content hash
    - Works for typical workflows where same grid object is reused
    - For JIT-compiled code, pre-compute cache and pass explicitly
    - Use `clear_kk_cache()` if memory becomes an issue

    Examples
    --------
    >>> nu = jnp.linspace(1000, 10000, 1000)
    >>> cache = get_or_create_kk_cache(nu)  # Computes, stores in registry
    >>> cache2 = get_or_create_kk_cache(nu)  # Returns cached (instant)
    >>> cache is cache2  # True - same object
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

    Notes
    -----
    Each cached grid uses ~8N² bytes. For N=1000, this is ~8 MB per grid.
    Most retrievals only need 1-2 cached grids, so memory is rarely an issue.
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

    Examples
    --------
    >>> from registry_cloud import get_kk_cache_stats
    >>> stats = get_kk_cache_stats()
    >>> print(f"Cached grids: {stats['num_cached_grids']}")
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
