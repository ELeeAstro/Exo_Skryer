"""
instru_convolve.py
==================
"""

from __future__ import annotations

import jax.numpy as jnp

from .registry_bandpass import (
    bandpass_num_bins,
    bandpass_indices_padded,
    bandpass_coefficients_padded,
)

__all__ = [
    "apply_response_functions",
    "get_bandpass_cache",
    "apply_response_functions_cached",
]


def _convolve_spectrum_core(
    spec: jnp.ndarray,
    idx_pad: jnp.ndarray,
    coeff_pad: jnp.ndarray,
) -> jnp.ndarray:
    """Convolve high-resolution spectrum into observational bins.

    The bandpass registry precomputes the trapezoidal quadrature coefficients,
    including response weights, optional wavelength weighting, normalization,
    and the single-point-bin fallback. The hot path is therefore a gather and a
    weighted sum.

    Parameters
    ----------
    spec : `~jax.numpy.ndarray`, shape (nwl_hi,)
        High-resolution spectrum evaluated on the master wavelength grid.
    idx_pad : `~jax.numpy.ndarray`, shape (nbin, max_len)
        Padded indices into the high-resolution spectrum array. Maps each
        wavelength sample to its position in `spec`.
    coeff_pad : `~jax.numpy.ndarray`, shape (nbin, max_len)
        Padded linear coefficients for each bin.

    Returns
    -------
    binned_spectrum : `~jax.numpy.ndarray`, shape (nbin,)
        Convolved spectrum in observational bins.
    """
    spec_pad = jnp.take(spec, idx_pad, axis=0)  # (nbin, max_len)
    return jnp.sum(spec_pad * coeff_pad, axis=1)


def get_bandpass_cache() -> dict[str, jnp.ndarray]:
    """Materialize bandpass registry arrays into a single PyTree cache (outside jit)."""
    n_bins = bandpass_num_bins()
    if n_bins == 0:
        empty_f = jnp.zeros((0, 0), dtype=jnp.float64)
        empty_i = jnp.zeros((0, 0), dtype=jnp.int32)
        return {
            "idx_pad": empty_i,
            "coeff_pad": empty_f,
        }

    return {
        "idx_pad": bandpass_indices_padded(),
        "coeff_pad": bandpass_coefficients_padded(),
    }


def apply_response_functions_cached(spectrum: jnp.ndarray, cache: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Convolve spectrum using a provided bandpass cache (jit-friendly)."""
    coeff_pad = cache["coeff_pad"]
    if coeff_pad.size == 0:
        return jnp.zeros((0,), dtype=spectrum.dtype)

    return _convolve_spectrum_core(
        spec=spectrum,
        idx_pad=cache["idx_pad"],
        coeff_pad=coeff_pad,
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
