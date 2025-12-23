"""
instru_convolve.py
==================
"""

from __future__ import annotations

import jax.numpy as jnp

from .registry_bandpass import (
    bandpass_num_bins,
    bandpass_wavelengths_padded,
    bandpass_weights_padded,
    bandpass_indices_padded,
    bandpass_norms,
)

__all__ = [
    "apply_response_functions"
]


def _convolve_spectrum_core(
    spec: jnp.ndarray,
    wl_pad: jnp.ndarray,
    w_pad: jnp.ndarray,
    idx_pad: jnp.ndarray,
    norms: jnp.ndarray,
) -> jnp.ndarray:
    """Convolve high-resolution spectrum into observational bins (JIT core).

    This function performs the actual convolution calculation using pre-computed
    padded arrays from the bandpass registry. It uses trapezoidal integration to
    compute the weighted average of the spectrum within each bin.

    Parameters
    ----------
    spec : `~jax.numpy.ndarray`, shape (nwl_hi,)
        High-resolution spectrum evaluated on the master wavelength grid.
    wl_pad : `~jax.numpy.ndarray`, shape (nbin, max_len)
        Padded wavelength samples for each bin. Each row contains the wavelength
        points where the response function is sampled, zero-padded to max_len.
    w_pad : `~jax.numpy.ndarray`, shape (nbin, max_len)
        Padded response weights/throughputs for each bin. Each row contains the
        instrument response at the corresponding wavelengths, zero-padded.
    idx_pad : `~jax.numpy.ndarray`, shape (nbin, max_len)
        Padded indices into the high-resolution spectrum array. Maps each
        wavelength sample to its position in `spec`.
    norms : `~jax.numpy.ndarray`, shape (nbin,)
        Normalization factors for each bin, typically the integrated throughput:
        ∫ w(λ) dλ over the bin's wavelength range.

    Returns
    -------
    binned_spectrum : `~jax.numpy.ndarray`, shape (nbin,)
        Convolved spectrum in observational bins. Each element represents:
        bin_i = ∫ F(λ) w(λ) dλ / ∫ w(λ) dλ

    Notes
    -----
    The convolution is computed as:
    1. Extract spectrum values at sampled wavelengths using `idx_pad`
    2. Multiply by response weights: F(λ) × w(λ)
    3. Integrate using trapezoidal rule (or simple weighted value for single-point bins)
    4. Normalize by integrated throughput

    For bins with only one wavelength point, trapezoidal integration is not applicable.
    Instead, we use F(λ) × w(λ) directly. Single-point bins are detected when the
    first two wavelengths in a row are identical (indicating padding).

    The normalization denominator is clamped to 1e-99 to prevent division by zero.
    """
    spec_pad = jnp.take(spec, idx_pad, axis=0)  # (nbin, max_len)

    # Detect single-point bins: if first two wavelengths are equal, it's a padded single point
    is_single_point = (wl_pad[:, 0] == wl_pad[:, 1])  # (nbin,) boolean array

    # Single-point bins: use first spectrum value times first weight
    single_point_result = spec_pad[:, 0] * w_pad[:, 0]  # (nbin,)

    # Multi-point bins: use trapezoidal integration
    multi_point_result = jnp.trapezoid(spec_pad * w_pad, x=wl_pad, axis=1)  # (nbin,)

    # Combine results based on bin type
    numerator = jnp.where(is_single_point, single_point_result, multi_point_result)

    return numerator / jnp.maximum(norms, 1e-99)


def apply_response_functions(spectrum: jnp.ndarray) -> jnp.ndarray:
    """Apply instrument response functions to convolve spectrum onto observational bins.

    This function takes a high-resolution model spectrum and convolves it with
    pre-loaded instrument response functions to produce a binned spectrum matching
    the observational wavelength grid. The response functions (boxcar, Gaussian,
    filter throughput curves, etc.) are retrieved from the bandpass registry.

    Parameters
    ----------
    spectrum : `~jax.numpy.ndarray`, shape (nwl_hi,)
        High-resolution model spectrum evaluated on the master wavelength grid.
        This should be the output from a radiative transfer calculation.

    Returns
    -------
    binned_spectrum : `~jax.numpy.ndarray`, shape (nbin,)
        Convolved spectrum in observational bins. Each element represents the
        flux integrated over the corresponding instrument response:
        F_bin[i] = ∫ F(λ) R_i(λ) dλ / ∫ R_i(λ) dλ

        If no bins are registered (nbin=0), returns an empty array with the
        same dtype as `spectrum`.

    Notes
    -----
    The bandpass registry must be loaded before calling this function using
    `load_bandpass_registry()` from the registry_bandpass module. The registry
    caches pre-computed wavelength samples, response weights, and normalization
    factors for efficient convolution.

    This function is designed to work inside JIT-compiled forward models. The
    bandpass data is fetched once and then passed to the JIT-compiled convolution
    kernel.

    See Also
    --------
    registry_bandpass.load_bandpass_registry : Loads and caches response data
    """
    n_bins = bandpass_num_bins()
    if n_bins == 0:
        # No bins prepared; return empty array with matching dtype
        return jnp.zeros((0,), dtype=spectrum.dtype)

    # Fetch pre-computed JAX arrays from the registry
    wl_pad = bandpass_wavelengths_padded()   # (nbin, max_len)
    w_pad = bandpass_weights_padded()        # (nbin, max_len)
    idx_pad = bandpass_indices_padded()      # (nbin, max_len)
    norms = bandpass_norms()                 # (nbin,)

    return _convolve_spectrum_core(
        spec=spectrum,
        wl_pad=wl_pad,
        w_pad=w_pad,
        idx_pad=idx_pad,
        norms=norms,
    )
