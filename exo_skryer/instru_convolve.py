"""
instru_convolve.py
==================
"""

from __future__ import annotations

import jax.numpy as jnp

import jax

from .aux_functions import simpson_padded
from .registry_bandpass import (
    bandpass_num_bins,
    bandpass_wavelengths_padded,
    bandpass_weights_padded,
    bandpass_indices_padded,
    bandpass_norms,
    bandpass_valid_lengths,
    bandpass_is_boxcar,
)

__all__ = [
    "apply_response_functions",
    "get_bandpass_cache",
    "apply_response_functions_cached",
]


def _convolve_spectrum_core(
    spec: jnp.ndarray,
    wl_pad: jnp.ndarray,
    w_pad: jnp.ndarray,
    idx_pad: jnp.ndarray,
    norms: jnp.ndarray,
    valid_lens: jnp.ndarray,
    is_boxcar: jnp.ndarray,
) -> jnp.ndarray:
    """Convolve high-resolution spectrum into observational bins (JIT core).

    This function performs the actual convolution calculation using pre-computed
    padded arrays from the bandpass registry. It uses trapezoidal integration to
    compute the weighted average of the spectrum within each bin.

    For boxcar bins (uniform response), the integration is:
        bin_i = ∫ F(λ) dλ / ∫ dλ

    For non-boxcar bins (filter throughput curves), the integration is photon-weighted:
        bin_i = ∫ F(λ) T(λ) λ dλ / ∫ T(λ) λ dλ

    Parameters
    ----------
    spec : `~jax.numpy.ndarray`, shape (nwl_hi,)
        High-resolution spectrum evaluated on the master wavelength grid.
    wl_pad : `~jax.numpy.ndarray`, shape (nbin, max_len)
        Padded wavelength samples for each bin. Each row contains the wavelength
        points where the response function is sampled, padded to max_len.
    w_pad : `~jax.numpy.ndarray`, shape (nbin, max_len)
        Padded response weights/throughputs for each bin. Each row contains the
        instrument response at the corresponding wavelengths, padded.
    idx_pad : `~jax.numpy.ndarray`, shape (nbin, max_len)
        Padded indices into the high-resolution spectrum array. Maps each
        wavelength sample to its position in `spec`.
    norms : `~jax.numpy.ndarray`, shape (nbin,)
        Normalization factors for each bin:
        - Boxcar: ∫ dλ
        - Non-boxcar: ∫ T(λ) λ dλ
    valid_lens : `~jax.numpy.ndarray`, shape (nbin,)
        Number of valid (non-padded) points for each bin.
    is_boxcar : `~jax.numpy.ndarray`, shape (nbin,)
        Boolean flags indicating which bins are boxcar (True) vs filter curves (False).

    Returns
    -------
    binned_spectrum : `~jax.numpy.ndarray`, shape (nbin,)
        Convolved spectrum in observational bins.
    """
    spec_pad = jnp.take(spec, idx_pad, axis=0)  # (nbin, max_len)

    # Compute numerator with conditional λ-weighting:
    # - Boxcar: ∫ F(λ) w(λ) dλ = ∫ F(λ) dλ  (since w=1)
    # - Non-boxcar: ∫ F(λ) T(λ) λ dλ
    # Use where() to apply λ-weighting only to non-boxcar bins
    lambda_weight = jnp.where(
        is_boxcar[:, None],  # Broadcast to (nbin, 1) then (nbin, max_len)
        1.0,                  # Boxcar: no λ weighting
        wl_pad                # Non-boxcar: multiply by λ
    )

    numerator = jnp.trapezoid(spec_pad * w_pad * lambda_weight, x=wl_pad, axis=1)  # (nbin,)

    return numerator / jnp.maximum(norms, 1e-99)


def get_bandpass_cache() -> dict[str, jnp.ndarray]:
    """Materialize bandpass registry arrays into a single PyTree cache (outside jit)."""
    n_bins = bandpass_num_bins()
    if n_bins == 0:
        empty_f = jnp.zeros((0, 0), dtype=jnp.float64)
        empty_i = jnp.zeros((0, 0), dtype=jnp.int32)
        empty_1 = jnp.zeros((0,), dtype=jnp.float64)
        empty_1i = jnp.zeros((0,), dtype=jnp.int32)
        empty_b = jnp.zeros((0,), dtype=bool)
        return {
            "wl_pad": empty_f,
            "w_pad": empty_f,
            "idx_pad": empty_i,
            "norms": empty_1,
            "valid_lens": empty_1i,
            "is_boxcar": empty_b,
        }

    return {
        "wl_pad": bandpass_wavelengths_padded(),
        "w_pad": bandpass_weights_padded(),
        "idx_pad": bandpass_indices_padded(),
        "norms": bandpass_norms(),
        "valid_lens": bandpass_valid_lengths(),
        "is_boxcar": bandpass_is_boxcar(),
    }


def apply_response_functions_cached(spectrum: jnp.ndarray, cache: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Convolve spectrum using a provided bandpass cache (jit-friendly)."""
    norms = cache["norms"]
    if norms.size == 0:
        return jnp.zeros((0,), dtype=spectrum.dtype)

    return _convolve_spectrum_core(
        spec=spectrum,
        wl_pad=cache["wl_pad"],
        w_pad=cache["w_pad"],
        idx_pad=cache["idx_pad"],
        norms=cache["norms"],
        valid_lens=cache["valid_lens"],
        is_boxcar=cache["is_boxcar"],
    )


def apply_response_functions(spectrum: jnp.ndarray) -> jnp.ndarray:
    """Apply instrument response functions to convolve spectrum onto observational bins.

    This function takes a high-resolution model spectrum and convolves it with
    pre-loaded instrument response functions to produce a binned spectrum matching
    the observational wavelength grid. The response functions (boxcar, Gaussian,
    filter throughput curves, etc.) are retrieved from the bandpass registry.

    For boxcar bins, the integration is simple averaging:
        F_bin[i] = ∫ F(λ) dλ / ∫ dλ

    For filter curve bins (non-boxcar), the integration is photon-weighted:
        F_bin[i] = ∫ F(λ) T(λ) λ dλ / ∫ T(λ) λ dλ

    Parameters
    ----------
    spectrum : `~jax.numpy.ndarray`, shape (nwl_hi,)
        High-resolution model spectrum evaluated on the master wavelength grid.
        This should be the output from a radiative transfer calculation.

    Returns
    -------
    binned_spectrum : `~jax.numpy.ndarray`, shape (nbin,)
        Convolved spectrum in observational bins.

        If no bins are registered (nbin=0), returns an empty array with the
        same dtype as `spectrum`.
    """
    return apply_response_functions_cached(spectrum, get_bandpass_cache())
