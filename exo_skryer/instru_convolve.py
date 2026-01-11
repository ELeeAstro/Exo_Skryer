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
    valid_lens: jnp.ndarray,
) -> jnp.ndarray:
    """Convolve high-resolution spectrum into observational bins (JIT core).

    This function performs the actual convolution calculation using pre-computed
    padded arrays from the bandpass registry. It uses Simpson's rule integration to
    compute the weighted average of the spectrum within each bin.

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
        Normalization factors for each bin, typically the integrated throughput:
        ∫ w(λ) dλ over the bin's wavelength range.
    valid_lens : `~jax.numpy.ndarray`, shape (nbin,)
        Number of valid (non-padded) points for each bin.

    Returns
    -------
    binned_spectrum : `~jax.numpy.ndarray`, shape (nbin,)
        Convolved spectrum in observational bins. Each element represents:
        bin_i = ∫ F(λ) w(λ) dλ / ∫ w(λ) dλ
    """
    spec_pad = jnp.take(spec, idx_pad, axis=0)  # (nbin, max_len)

    # # Integrate using Simpson's rule with vmap over bins
    # numerator = jax.vmap(simpson_padded, in_axes=(0, 0, 0))(
    #     spec_pad * w_pad,  # y: (nbin, max_len)
    #     wl_pad,            # x: (nbin, max_len)
    #     valid_lens,        # n_valid: (nbin,)
    # )

    # Alternative: use trapezoidal integration (commented out)
    numerator = jnp.trapezoid(spec_pad * w_pad, x=wl_pad, axis=1)  # (nbin,)

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
    valid_lens = bandpass_valid_lengths()    # (nbin,)

    return _convolve_spectrum_core(
        spec=spectrum,
        wl_pad=wl_pad,
        w_pad=w_pad,
        idx_pad=idx_pad,
        norms=norms,
        valid_lens=valid_lens,
    )
