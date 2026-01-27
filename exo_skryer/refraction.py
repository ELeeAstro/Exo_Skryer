"""
refraction.py
=============

Approximate refraction support for transmission spectroscopy.

Current implementation: "cutoff" mode (option A) that applies a refractive
boundary (fully opaque below a wavelength-dependent impact parameter) without
curved-ray optical-depth integration.
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from . import registry_ray as XR
from .data_constants import AU, kb, amu

__all__ = ["refraction_cutoff_mask"]


def refraction_cutoff_mask(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opac: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Return a boolean mask for impact parameters blocked by refraction.

    The mask is defined on the same "impact parameter grid" used by the current
    transit RT kernels: `b ≈ R0 + z_lay` (layer midpoints).

    For each (layer, wavelength), we estimate the bending angle using the
    exponential-atmosphere approximation:

        alpha(b, λ) ≈ (n(b, λ) - 1) * sqrt(2π b / H)

    and mark a ray as blocked if:

        alpha(b, λ) > theta_star,    theta_star = asin(R_s / a)

    where `a = a_sm * AU`.

    Parameters
    ----------
    state : dict
        Must contain `R0`, `R_s`, `z_lay`, `T_lay`, `mu_lay`, `nd_lay`, and `vmr_lay`.
    params : dict
        Must contain `log_10_g` and `a_sm` (AU).
    opac : dict
        Must contain `ray_nm1_table` and `ray_nd_ref` aligned with
        `registry_ray.ray_species_names()`.

    Returns
    -------
    mask : jnp.ndarray, shape (nlay, nwl), dtype bool
        True where refraction blocks stellar rays (treat as fully opaque).
    """
    nwl = state["wl"].shape[0]
    nlay = state["nd_lay"].shape[0]

    if "ray_nm1_table" not in opac or "ray_nd_ref" not in opac:
        raise RuntimeError("Refraction requested but Rayleigh refractivity tables are missing from opac cache.")

    # Stellar angular radius at the planet
    a_cm = params["a_sm"] * AU
    theta_star = jnp.arcsin(jnp.clip(state["R_s"] / a_cm, 0.0, 1.0))  # scalar

    # Scale height at each layer midpoint (include spherical gravity correction)
    R0 = state["R0"]
    z_lay = state["z_lay"]
    b = R0 + z_lay  # (nlay,)

    g0 = 10.0 ** params["log_10_g"]
    g_z = g0 * (R0 / b) ** 2
    H = (kb * state["T_lay"]) / (state["mu_lay"] * amu * g_z)  # (nlay,)

    # Build (n-1)(layer, wl) from STP refractivities + ideal-gas scaling with number density.
    nm1_ref = opac["ray_nm1_table"].astype(jnp.float64)  # (nspec, nwl)
    nd_ref = opac["ray_nd_ref"].astype(jnp.float64)      # (nspec,)

    species_names = XR.ray_species_names()
    vmr_lay = state["vmr_lay"]
    nd_lay = state["nd_lay"].astype(jnp.float64)  # (nlay,)

    nm1_coeff = jnp.zeros((nlay, nwl), dtype=jnp.float64)
    for i, name in enumerate(species_names):
        vmr_i = jnp.broadcast_to(vmr_lay[name], (nlay,)).astype(jnp.float64)  # (nlay,)
        coeff_i = nm1_ref[i][None, :] / nd_ref[i]  # (1, nwl)
        nm1_coeff = nm1_coeff + vmr_i[:, None] * coeff_i

    nm1_layer = nd_lay[:, None] * nm1_coeff  # (nlay, nwl)
    nm1_layer = jnp.maximum(nm1_layer, 0.0)

    # Exponential-atmosphere bending approximation
    alpha = nm1_layer * jnp.sqrt(2.0 * jnp.pi * b[:, None] / jnp.maximum(H[:, None], 1.0))

    return alpha > theta_star

