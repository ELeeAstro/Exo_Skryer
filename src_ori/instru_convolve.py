from __future__ import annotations

import jax
from jax import jit
import jax.numpy as jnp

# Adjust the import path/module name to wherever you placed instru_bandpass.py
from registry_bandpass import (
    bandpass_num_bins,
    bandpass_wavelengths_padded,
    bandpass_weights_padded,
    bandpass_indices_padded,
    bandpass_norms,
)

@jit
def _convolve_spectrum_core(
    spec: jnp.ndarray,
    wl_pad: jnp.ndarray,
    w_pad: jnp.ndarray,
    idx_pad: jnp.ndarray,
    norms: jnp.ndarray,
) -> jnp.ndarray:
    """
    JIT-compiled core for convolving a single spectrum.

    All inputs are explicit JAX arrays, making this a pure function.
    """
    n_bins = norms.shape[0]

    def convolve_bin(carry, i):
        idx_row = idx_pad[i]
        wl_row = wl_pad[i]
        w_row = w_pad[i]
        spec_slice = spec[idx_row]
        numerator = jnp.trapezoid(spec_slice * w_row, x=wl_row)
        value = numerator / jnp.maximum(norms[i], 1e-99)
        return carry, value

    _, binned = jax.lax.scan(convolve_bin, None, jnp.arange(n_bins))
    return binned


def apply_response_functions(
    wl: jnp.ndarray,          # currently unused, kept for API consistency
    spectrum: jnp.ndarray,
) -> jnp.ndarray:
    """
    Convolve hi-res spectrum with per-bin response functions, using the
    pre-built bandpass registry and padded JAX arrays.
    """

    n_bins = bandpass_num_bins()
    if n_bins == 0:
        # No bins prepared; return empty array of the right dtype
        return jnp.zeros((0,), dtype=spectrum.dtype)

    # Fetch the JAX arrays once, outside the jitted core
    wl_pad = bandpass_wavelengths_padded()   # (n_bins, max_len)
    w_pad = bandpass_weights_padded()        # (n_bins, max_len)
    idx_pad = bandpass_indices_padded()      # (n_bins, max_len)
    norms = bandpass_norms()                 # (n_bins,)

    return _convolve_spectrum_core(
        spec=spectrum,
        wl_pad=wl_pad,
        w_pad=w_pad,
        idx_pad=idx_pad,
        norms=norms,
    )
