"""
kk_schemes.py
=============

Kramers-Kronig transform functions for computing real refractive index from
imaginary part using causality relations.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from .registry_cloud import KKGridCache, get_or_create_kk_cache

__all__ = [
    "kk_n_from_k_wavenumber_cached",
    "kk_n_from_k_wavenumber_fast",
    "kk_n_from_k_wavenumber",
    "kk_n_from_k_wavelength_um",
]


def kk_n_from_k_wavenumber_cached(
    nu: jnp.ndarray,
    k_nu: jnp.ndarray,
    nu_ref: jnp.ndarray,
    n_ref: jnp.ndarray,
    cache: KKGridCache,
) -> jnp.ndarray:
    """Compute `n(ν)` from `k(ν)` via a singly-subtracted Kramers–Kronig relation.

    This variant is JIT-friendly: the `KKGridCache` is passed explicitly, avoiding
    Python-side cache lookups. Grid-dependent trapezoid weights are reused via
    the cache.

    Parameters
    ----------
    nu : `~jax.numpy.ndarray`, shape (N,)
        Wavenumber grid (strictly increasing), e.g. cm⁻¹.
    k_nu : `~jax.numpy.ndarray`, shape (N,)
        Extinction coefficient on the wavenumber grid (clipped to be non-negative).
    nu_ref : `~jax.numpy.ndarray`
        Reference wavenumber used to anchor the subtraction term.
    n_ref : `~jax.numpy.ndarray`
        Real refractive index at `nu_ref`.
    cache : `~exo_skryer.registry_cloud.KKGridCache`
        Precomputed grid quantities for this `nu` grid (e.g., trapezoid weights).

    Returns
    -------
    n_nu : `~jax.numpy.ndarray`, shape (N,)
        Real refractive index on the wavenumber grid.
    """
    k_nu = jnp.maximum(k_nu, 0.0)

    # Extract cached quantities (only O(N) trapezoid weights)
    trap_weights = cache.trap_weights

    # Compute alpha_inv on-the-fly to save memory
    # For N=33219, storing this would need 8.8 GB!
    # Computing it is fast with JAX JIT fusion
    nu_i = nu[:, None]  # (N,1)
    nu_j = nu[None, :]  # (1,N)
    alpha = nu_j**2 - nu_i**2  # (N,N)
    alpha_inv = jnp.where(alpha != 0.0, 1.0 / alpha, 0.0)

    # k(nu_ref) via interpolation
    k_ref = jnp.interp(nu_ref, nu, k_nu)

    # Key optimization: compute v = nu * k_nu once
    v = nu * k_nu  # (N,)

    # y1[i,j] = (v[j] - v[i]) / alpha[i,j]
    v_diff = v[None, :] - v[:, None]  # (N,N)
    y1 = v_diff * alpha_inv

    # y2[i,j] = (v[j] - nu_ref * k_ref) / beta[j]
    beta = nu**2 - nu_ref**2
    beta_inv = jnp.where(beta != 0.0, 1.0 / beta, 0.0)
    v_ref = nu_ref * k_ref
    y2 = (v[None, :] - v_ref) * beta_inv[None, :]

    # Combined integrand
    y = y1 - y2  # (N,N)

    # Trapezoid integration using precomputed weights
    integ = jnp.sum(y * trap_weights[None, :], axis=1)  # (N,)

    n_nu = n_ref + (2.0 / jnp.pi) * integ
    return n_nu


def kk_n_from_k_wavenumber_fast(
    nu: jnp.ndarray,      # (N,) strictly increasing, e.g. cm^-1
    k_nu: jnp.ndarray,    # (N,) extinction coefficient, >= 0
    nu_ref: jnp.ndarray,  # scalar, same units as nu
    n_ref: jnp.ndarray,   # scalar
    cache: Optional[KKGridCache] = None,
) -> jnp.ndarray:
    """Optimized KK relation using precomputed grid quantities.

    Parameters
    ----------
    nu : `~jax.numpy.ndarray`, shape (N,)
        Wavenumber grid (strictly increasing), e.g. cm⁻¹.
    k_nu : `~jax.numpy.ndarray`, shape (N,)
        Extinction coefficient on the wavenumber grid (clipped to be non-negative).
    nu_ref : `~jax.numpy.ndarray`
        Reference wavenumber used to anchor the subtraction term.
    n_ref : `~jax.numpy.ndarray`
        Real refractive index at `nu_ref`.
    cache : `~exo_skryer.registry_cloud.KKGridCache`, optional
        Precomputed grid quantities for this `nu` grid. If `None`, the cache is
        obtained via `registry_cloud.get_or_create_kk_cache(nu)`.

    Returns
    -------
    n_nu : `~jax.numpy.ndarray`, shape (N,)
        Real refractive index on the wavenumber grid.

    Notes
    -----
    For best performance in JIT-compiled code, precompute the cache and pass it
    explicitly:

        cache = get_or_create_kk_cache(nu)
        n = kk_n_from_k_wavenumber_fast(nu, k_nu, nu_ref, n_ref, cache=cache)
    """
    nu = jnp.asarray(nu)
    k_nu = jnp.maximum(jnp.asarray(k_nu), 0.0)
    nu_ref = jnp.asarray(nu_ref)
    n_ref = jnp.asarray(n_ref)

    # Get cache from registry if not provided
    if cache is None:
        cache = get_or_create_kk_cache(nu)

    return kk_n_from_k_wavenumber_cached(nu, k_nu, nu_ref, n_ref, cache)


def kk_n_from_k_wavenumber(
    nu: jnp.ndarray,      # (N,) strictly increasing, e.g. cm^-1
    k_nu: jnp.ndarray,    # (N,) extinction coefficient, >= 0
    nu_ref: jnp.ndarray,  # scalar, same units as nu
    n_ref: jnp.ndarray,   # scalar
) -> jnp.ndarray:
    """Compute `n(ν)` from `k(ν)` via a singly-subtracted KK relation.

    This is a convenience wrapper around `kk_n_from_k_wavenumber_fast()` that
    looks up the grid cache internally.

    Parameters
    ----------
    nu : `~jax.numpy.ndarray`, shape (N,)
        Wavenumber grid (strictly increasing), e.g. cm⁻¹.
    k_nu : `~jax.numpy.ndarray`, shape (N,)
        Extinction coefficient on the wavenumber grid (clipped to be non-negative).
    nu_ref : `~jax.numpy.ndarray`
        Reference wavenumber used to anchor the subtraction term.
    n_ref : `~jax.numpy.ndarray`
        Real refractive index at `nu_ref`.

    Returns
    -------
    n_nu : `~jax.numpy.ndarray`, shape (N,)
        Real refractive index on the wavenumber grid.
    """
    return kk_n_from_k_wavenumber_fast(nu, k_nu, nu_ref, n_ref, cache=None)


def kk_n_from_k_wavelength_um(
    wl_um: jnp.ndarray,   # (N,) wavelength in micron
    k_wl: jnp.ndarray,    # (N,) extinction coefficient on wl grid
    wl_ref_um: jnp.ndarray,
    n_ref: jnp.ndarray,
    cache: Optional[KKGridCache] = None,
) -> jnp.ndarray:
    """Compute `n(λ)` from `k(λ)` via KK, using wavelength inputs in microns.

    This convenience wrapper converts wavelength to wavenumber via
    `ν[cm⁻¹] = 10⁴ / λ[μm]`, runs `kk_n_from_k_wavenumber_fast()` in wavenumber
    space, and returns `n` on the original wavelength ordering.

    Parameters
    ----------
    wl_um : `~jax.numpy.ndarray`, shape (N,)
        Wavelength grid in microns.
    k_wl : `~jax.numpy.ndarray`, shape (N,)
        Extinction coefficient on the wavelength grid (clipped to be non-negative).
    wl_ref_um : `~jax.numpy.ndarray`
        Reference wavelength in microns used to define `nu_ref`.
    n_ref : `~jax.numpy.ndarray`
        Real refractive index at `wl_ref_um`.
    cache : `~exo_skryer.registry_cloud.KKGridCache`, optional
        Precomputed grid quantities for the wavenumber grid. If `None`, the
        cache is obtained via `registry_cloud.get_or_create_kk_cache(nu)`.

    Returns
    -------
    n_wl : `~jax.numpy.ndarray`, shape (N,)
        Real refractive index on the wavelength grid.
    """
    wl_um = jnp.asarray(wl_um)
    k_wl = jnp.maximum(jnp.asarray(k_wl), 0.0)

    # Safety: avoid division by 0 (physically wl must be > 0 anyway)
    wl_um = jnp.maximum(wl_um, 1e-12)

    # Convert to wavenumber nu [cm^-1]
    nu = 1e4 / wl_um
    nu_ref = 1e4 / jnp.maximum(jnp.asarray(wl_ref_um), 1e-12)

    # Ensure nu is increasing for KK (reverse if needed)
    rev = nu[0] > nu[-1]
    nu_inc = jnp.where(rev, nu[::-1], nu)
    k_inc  = jnp.where(rev, k_wl[::-1], k_wl)

    n_inc = kk_n_from_k_wavenumber_fast(nu_inc, k_inc, nu_ref=nu_ref, n_ref=n_ref, cache=cache)

    # Back to original wl ordering
    n_wl = jnp.where(rev, n_inc[::-1], n_inc)
    return n_wl